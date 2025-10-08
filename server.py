# -------------------------------------------
# ESP8266 PV Inference API (Real Model) + Telegram Alert
# - No voltage rules. Use model only.
# - Alert when label != "ปกติ"
# - /dashboard /recent /stats for viewing results
# -------------------------------------------

import os
import pickle
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator

# ============== Logging ==============
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ============== FastAPI ==============
app = FastAPI(title="ESP8266 PV Inference API", version="2.0")

# ============== Config ==============
# แจ้งเตือนเมื่อไม่ปกติเท่านั้น
ALERT_LABELS = {1, 2}        # 1=สกปรก, 2=แตก (ให้ตรงกับโมเดลของคุณ)
ALERT_PROBA  = 0.50          # ตั้งขั้นต่ำสำหรับ notification (ปรับได้ตามจริง)

# ตั้งจาก ENV ใน Render (Settings → Environment)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# เก็บผลล่าสุดไว้ดูบน dashboard
MAX_LOG = 500
PRED_LOG: List[dict] = []

# ============== Model / Scaler / Label Encoder ==============
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)

def _safe_load_pickle(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log.warning("Cannot load pickle %s: %s", path, e)
        return None

SCALER = _safe_load_pickle("scaler.pkl")
LABEL_ENCODER = _safe_load_pickle("label_encoder.pkl")

MODEL = SimpleMLP(in_dim=9, hidden=64, out_dim=3)
if os.path.exists("clf_tested.pt"):
    try:
        state = torch.load("clf_tested.pt", map_location="cpu")
        MODEL.load_state_dict(state, strict=False)
        log.info("Loaded model weights from clf_tested.pt")
    except Exception as e:
        log.error("Load clf_tested.pt failed: %s", e)
else:
    log.warning("clf_tested.pt not found. Using randomly initialized model.")

MODEL.eval()

# ============== Schemas ==============
class FeaturePacket(BaseModel):
    # data: 9 หรือ 12 ฟีเจอร์
    data: List[float] = Field(..., description="Feature vector (9 หรือ 12 ค่า)")
    v: Optional[float] = None
    i: Optional[float] = None
    p: Optional[float] = None

    @validator("data")
    def check_len(cls, v):
        if len(v) not in (9, 12):
            raise ValueError("data must have 9 or 12 elements")
        return v

class PredictIn(BaseModel):
    features: FeaturePacket

class PredictOut(BaseModel):
    label_idx: int
    label_text: str
    proba: float
    v: Optional[float] = None
    i: Optional[float] = None
    p: Optional[float] = None

# ============== Utils ==============
def _map_12_to_9(arr12: np.ndarray) -> np.ndarray:
    """
    Map 12 → 9 ให้ตรงกับโมเดลที่เทรนไว้
    12 = [v_rms, i_rms, p_rms, v_zc, i_zc, p_zc, v_ssc, i_ssc, p_ssc, v_mean, i_mean, p_mean]
    9  = [v_rms, i_rms, p_rms, v_zc, i_zc, v_ssc, i_ssc, v_mean, i_mean]
    """
    idx = [0, 1, 2, 3, 4, 6, 7, 9, 10]
    return arr12[idx]

def _prepare_input(x: List[float]) -> np.ndarray:
    arr = np.array(x, dtype=np.float32)
    if arr.shape[0] == 12:
        arr = _map_12_to_9(arr)
    elif arr.shape[0] != 9:
        raise ValueError("Expect 9 or 12 features")
    arr = arr.reshape(1, -1)
    if SCALER is not None:
        try:
            arr = SCALER.transform(arr)
        except Exception as e:
            log.warning("Scaler.transform error: %s", e)
    return arr

def _label_text(idx: int) -> str:
    if LABEL_ENCODER is not None:
        try:
            return str(LABEL_ENCODER.inverse_transform([idx])[0])
        except Exception:
            pass
    return {0: "ปกติ", 1: "สกปรก", 2: "แตก"}.get(idx, str(idx))

async def _send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        log.warning("Telegram error: %s", e)

def _append_log(rec: dict):
    PRED_LOG.append(rec)
    if len(PRED_LOG) > MAX_LOG:
        del PRED_LOG[: len(PRED_LOG) - MAX_LOG]

# ============== Endpoints ==============
@app.get("/", tags=["root"])
def root():
    return {"ok": True, "msg": "Server is running successfully!"}

@app.post("/predict", response_model=PredictOut, tags=["inference"])
async def predict(req: PredictIn, request: Request):
    ip = request.client.host if request and request.client else "?"
    feats = req.features.data
    v, i, p = req.features.v, req.features.i, req.features.p

    log.info("Req from %s data=%s v=%s i=%s p=%s", ip, np.round(feats, 5).tolist(), v, i, p)

    try:
        arr = _prepare_input(feats)
        with torch.no_grad():
            t = torch.from_numpy(arr)
            y = MODEL(t)
            probs = y.numpy()[0]
            label_idx = int(np.argmax(probs))
            proba = float(probs[label_idx])
        label_txt = _label_text(label_idx)
    except Exception as e:
        log.exception("Infer error: %s", e)
        raise HTTPException(status_code=400, detail="infer failed")

    # เก็บ log ไว้ดูบน dashboard
    rec = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip": ip,
        "label_idx": label_idx,
        "label_text": label_txt,
        "proba": round(proba, 6),
        "v": v, "i": i, "p": p
    }
    _append_log(rec)

    # แจ้งเตือนเฉพาะเมื่อไม่ปกติ
    if (label_idx in ALERT_LABELS) and (proba >= ALERT_PROBA):
        msg = f"⚠️ พบแผงโซล่าเซลล์ประเภท “{label_idx}” {label_txt} (p={proba:.2f})"
        if any(x is not None for x in (v, i, p)):
            msg += f"\nV={v if v is not None else '-'}  I={i if i is not None else '-'}  P={p if p is not None else '-'}"
        await _send_telegram(msg)
        log.info("Telegram sent.")

    return PredictOut(**rec)

@app.get("/recent", response_class=JSONResponse)
def recent():
    return PRED_LOG[-50:][::-1]  # ล่าสุด 50 รายการ

@app.get("/stats", response_class=JSONResponse)
def stats():
    counts = {"ปกติ": 0, "สกปรก": 0, "แตก": 0}
    for r in PRED_LOG:
        counts[r["label_text"]] += 1
    total = sum(counts.values())
    return {"total": total, "counts": counts}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    def badge(lbl: str) -> str:
        if lbl == "ปกติ": return "🟩 ปกติ"
        if lbl == "สกปรก": return "🟨 สกปรก"
        if lbl == "แตก": return "🟥 แตก"
        return lbl

    rows = ""
    for r in PRED_LOG[::-1]:
        rows += f"""
        <tr>
          <td>{r['ts']}</td>
          <td>{r['ip']}</td>
          <td>{badge(r['label_text'])}</td>
          <td>{r['proba']}</td>
          <td>{'-' if r['v'] is None else r['v']}</td>
          <td>{'-' if r['i'] is None else r['i']}</td>
          <td>{'-' if r['p'] is None else r['p']}</td>
        </tr>"""

    html = f"""
    <html>
    <head>
      <meta http-equiv="refresh" content="10">
      <title>PV Dashboard</title>
      <style>
        body {{ background:#111; color:#fff; font-family:Arial, sans-serif; }}
        table {{ border-collapse:collapse; width:100%; }}
        th,td {{ border:1px solid #444; padding:6px; text-align:center; }}
        th {{ background:#333; }}
        tr:hover {{ background:#222; }}
        h2 {{ margin:10px 0 16px 0; }}
        a {{ color:#9cf; }}
      </style>
    </head>
    <body>
      <h2>PV Dashboard (ล่าสุด {len(PRED_LOG)} รายการ)</h2>
      <div>API: <a href="/recent">/recent</a> · <a href="/stats">/stats</a></div>
      <table>
        <tr><th>เวลา</th><th>IP</th><th>ประเภท</th><th>p</th><th>V</th><th>I</th><th>P</th></tr>
        {rows}
      </table>
    </body>
    </html>
    """
    return HTMLResponse(html)

# ============== Uvicorn boot (local run) ==============
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
