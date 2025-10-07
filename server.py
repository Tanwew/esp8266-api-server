# server.py
# -------------------------------------------
# ESP8266 PV Inference API + Telegram Alert + Web Dashboard
# - รองรับ 9/12 ฟีเจอร์
# - กฎแรงดัน override:
#     V < 38     -> แตก (2)
#     38 <= V<39 -> สกปรก (1)
#     V >= 39    -> ปกติ (0)
# - แสดงผลบน Render:
#     /dashboard  : HTML ตารางล่าสุด (auto-refresh)
#     /recent     : JSON ผลล่าสุด N รายการ
#     /stats      : JSON สรุปจำนวนคลาส
# -------------------------------------------

import os
import logging
from typing import List, Optional, Deque
from collections import deque, Counter
from datetime import datetime

import numpy as np
import requests
import torch
import torch.nn as nn
import pickle

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator

# ============== Logging ==============
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ============== FastAPI ==============
app = FastAPI(title="ESP8266 PV Inference API", version="1.4")

# ============== Config (Alert) ==============
ALERT_LABELS = {0, 1, 2}   # แจ้งทุกผล
ALERT_PROBA  = 0.50        # ให้ส่งแน่ ๆ

# ENV (ถ้าไม่ได้ตั้ง จะใช้ค่าด้านหลัง)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ============== In-memory store for dashboard ==============
MAX_RECENT = int(os.getenv("MAX_RECENT", "200"))
RECENT: Deque[dict] = deque(maxlen=MAX_RECENT)

# ============== Model / Scaler / LabelEncoder ==============
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
    data: List[float] = Field(..., description="Feature vector (9 or 12)")
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
LABEL_FALLBACK = {0: "ปกติ", 1: "สกปรก", 2: "แตก"}

def _label_text(idx: int) -> str:
    if LABEL_ENCODER is not None:
        try:
            return str(LABEL_ENCODER.inverse_transform([idx])[0])
        except Exception:
            pass
    return LABEL_FALLBACK.get(idx, str(idx))

def _map_12_to_9(arr12: np.ndarray) -> np.ndarray:
    # 12 = [v_rms,i_rms,p_rms,v_zc,i_zc,p_zc,v_ssc,i_ssc,p_ssc,v_mean,i_mean,p_mean]
    #  9 = [v_rms,i_rms,p_rms,v_zc,i_zc,v_ssc,i_ssc,v_mean,i_mean]
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

def _choose_voltage(req: PredictIn) -> Optional[float]:
    feats = req.features.data
    if req.features.v is not None:
        return float(req.features.v)
    if len(feats) == 12:
        return float(feats[9])     # v_mean
    return float(feats[0])         # v_rms (fallback)

def infer_one(x: List[float]):
    arr = _prepare_input(x)
    with torch.no_grad():
        t = torch.from_numpy(arr)
        y = MODEL(t)
        probs = y.numpy()[0]
        idx = int(np.argmax(probs))
        prob = float(probs[idx])
        return idx, prob

def send_telegram_message(text: str) -> bool:
    token, chat_id = TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if not token or not chat_id:
        log.info("Telegram TOKEN/CHAT_ID not set -> skip")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=5)
        if r.ok:
            log.info("Telegram sent")
            return True
        log.warning("Telegram failed %s %s", r.status_code, r.text)
        return False
    except Exception as e:
        log.warning("Telegram error: %s", e)
        return False

# ============== Endpoints ==============
@app.get("/", tags=["root"])
def root():
    return {"ok": True, "msg": "Server is running successfully!"}

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut, tags=["inference"])
def predict(req: PredictIn, request: Request):
    feats = req.features.data
    v_in, i_in, p_in = req.features.v, req.features.i, req.features.p
    ip = request.client.host if request and request.client else "?"

    log.info("Req from %s data=%s v=%s i=%s p=%s",
             ip, np.round(feats, 5).tolist(), v_in, i_in, p_in)

    # 1) โมเดลทำนาย
    try:
        label_idx, proba = infer_one(feats)
        label_txt = _label_text(label_idx)
    except Exception as e:
        log.exception("Infer error: %s", e)
        raise HTTPException(status_code=400, detail="infer failed")

    # 2) กฎแรงดันทับผลโมเดล
    v_used = None
    rule_applied = False
    try:
        v_used = _choose_voltage(req)
    except Exception:
        pass

    if v_used is not None:
        if v_used < 38.0:
            label_idx, label_txt, proba = 2, _label_text(2), 0.99
            rule_applied = True
        elif 38.0 <= v_used < 39.0:
            label_idx, label_txt, proba = 1, _label_text(1), 0.95
            rule_applied = True
        elif v_used >= 39.0:
            label_idx, label_txt, proba = 0, _label_text(0), 0.90
            rule_applied = True

    # 3) เก็บลง RECENT (ไว้โชว์หน้า dashboard)
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "ip": ip,
        "label_idx": label_idx,
        "label_text": label_txt,
        "proba": round(proba, 6),
        "v": v_in if v_in is not None else (v_used if v_used is not None else None),
        "i": i_in,
        "p": p_in,
        "rule": rule_applied
    }
    RECENT.appendleft(row)

    # 4) ส่ง Telegram (แจ้งทุกผล)
    text = f"พบแผงโซล่าเซลล์ประเภท “{label_txt}” (p={proba:.2f})"
    if row["v"] is not None or i_in is not None or p_in is not None:
        text += f"\nV={row['v'] if row['v'] is not None else '-'}  I={i_in if i_in is not None else '-'}  P={p_in if p_in is not None else '-'}"
    send_telegram_message(text)

    return PredictOut(
        label_idx=label_idx,
        label_text=label_txt,
        proba=row["proba"],
        v=row["v"],
        i=i_in,
        p=p_in
    )

# ---------- Dashboard (HTML) ----------
@app.get("/dashboard", response_class=HTMLResponse, tags=["dashboard"])
def dashboard():
    # Auto refresh every 5s
    rows_html = []
    for r in list(RECENT)[:100]:
        badge = {"ปกติ":"#10b981","สกปรก":"#f59e0b","แตก":"#ef4444"}.get(r["label_text"], "#6b7280")
        rule = "✔" if r["rule"] else ""
        rows_html.append(f"""
        <tr>
          <td>{r["ts"]}</td>
          <td>{r["ip"]}</td>
          <td style="font-weight:600;color:{badge}">{r["label_text"]}</td>
          <td>{r["proba"]:.2f}</td>
          <td>{r.get("v","-")}</td>
          <td>{r.get("i","-")}</td>
          <td>{r.get("p","-")}</td>
          <td>{rule}</td>
        </tr>
        """)
    table = "\n".join(rows_html) or "<tr><td colspan='8' style='text-align:center;color:#9ca3af'>No data yet</td></tr>"
    return f"""
<!doctype html>
<html><head>
<meta charset="utf-8"/>
<meta http-equiv="refresh" content="5"/>
<title>PV Inference Dashboard</title>
<style>
 body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;padding:16px;background:#0b1220;color:#e5e7eb}}
 h1{{margin:0 0 12px}}
 .wrap{{max-width:1000px;margin:0 auto}}
 table{{width:100%;border-collapse:collapse;background:#111827;border-radius:10px;overflow:hidden}}
 th,td{{padding:10px 12px;border-bottom:1px solid #1f2937;font-size:14px}}
 th{{text-align:left;background:#0f172a;color:#a5b4fc;position:sticky;top:0}}
 tr:hover td{{background:#0f172a}}
 .meta{{color:#9ca3af;margin:6px 0 16px}}
 .pill{{display:inline-block;background:#1f2937;border:1px solid #374151;padding:4px 8px;border-radius:999px;margin-right:6px}}
</style>
</head>
<body>
<div class="wrap">
  <h1>PV Inference Dashboard</h1>
  <div class="meta">
    <span class="pill">Records: {len(RECENT)}</span>
    <span class="pill">Auto-refresh: 5s</span>
    <span class="pill"><a href="/recent" style="color:#93c5fd;text-decoration:none">/recent</a></span>
    <span class="pill"><a href="/stats" style="color:#93c5fd;text-decoration:none">/stats</a></span>
    <span class="pill"><a href="/docs" style="color:#93c5fd;text-decoration:none">/docs</a></span>
  </div>
  <table>
    <thead>
      <tr>
        <th>Time</th><th>IP</th><th>Label</th><th>p</th><th>V</th><th>I</th><th>P</th><th>Rule?</th>
      </tr>
    </thead>
    <tbody>
      {table}
    </tbody>
  </table>
</div>
</body></html>
"""

# ---------- Recent JSON ----------
@app.get("/recent", response_class=JSONResponse, tags=["dashboard"])
def recent(limit: int = Query(50, ge=1, le=200)):
    return JSONResponse(list(RECENT)[:limit])

# ---------- Stats JSON ----------
@app.get("/stats", response_class=JSONResponse, tags=["dashboard"])
def stats():
    counts = Counter([r["label_text"] for r in RECENT])
    return {"total": len(RECENT), "counts": counts}

# ============== Uvicorn boot ==============
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
