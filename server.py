# -------------------------------------------
# ESP8266 PV Inference API + Dashboard + Telegram
# Majority Vote (โมเดลจริง) + Voltage Helper (เพิ่มคะแนน 1 แต้มเฉพาะรอบนั้น)
# -------------------------------------------
import os
import logging
from typing import List, Optional
from collections import deque, Counter
from datetime import datetime, timezone

import numpy as np
import requests
import torch
import torch.nn as nn
import pickle

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator

# ============== Logging ==============
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ============== FastAPI ==============
app = FastAPI(title="ESP8266 PV Inference API", version="2.1-majority+vhelper")

# ============== Config ==============
# แจ้งเตือนเฉพาะ “ไม่ปกติ” (label_idx != 0)
ONLY_ALERT_ABNORMAL = True
SHOW_VIP = True  # แนบค่า V/I/P ในข้อความแจ้งเตือน

# ขนาดหน้าต่างสำหรับ majority vote (ตั้ง ENV MAJ_WINDOW ได้)
MAJ_WINDOW = int(os.getenv("MAJ_WINDOW", "5"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8091687691:AAHRnXog3_BEFTOdbmPXlSkCXPaRSt9eCE4")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "8279950843")

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
            nn.Softmax(dim=1),
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
    data: List[float] = Field(..., description="Feature vector (9 หรือ 12)")
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
    rule: bool   # ใช้ voltage helper หรือไม่

# ============== Helpers ==============
def _map_12_to_9(arr12: np.ndarray) -> np.ndarray:
    # 12 -> 9: [v_rms, i_rms, p_rms, v_zc, i_zc, p_zc, v_ssc, i_ssc, p_ssc, v_mean, i_mean, p_mean]
    # 9  ->     [v_rms, i_rms, p_rms, v_zc, i_zc, v_ssc, i_ssc, v_mean, i_mean]
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

def send_telegram_message(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=5)
        return bool(r.ok)
    except Exception as e:
        log.warning("Telegram error: %s", e)
        return False

def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

# ============== Stores ==============
HISTORY_MAX = 500
HISTORY = deque(maxlen=HISTORY_MAX)
COUNTS = {"ปกติ": 0, "สกปรก": 0, "แตก": 0}

# เก็บ “ผลดิบจากโมเดล” เพื่อ majority
PRED_WINDOW = deque(maxlen=MAJ_WINDOW)

def record_result(ip, label_idx, label_text, proba, v, i, p, rule):
    entry = {
        "ts": _now_iso(),
        "ip": ip,
        "label_idx": int(label_idx),
        "label_text": str(label_text),
        "proba": float(proba),
        "v": None if v is None else float(v),
        "i": None if i is None else float(i),
        "p": None if p is None else float(p),
        "rule": bool(rule),
    }
    HISTORY.appendleft(entry)
    if entry["label_text"] in COUNTS:
        COUNTS[entry["label_text"]] += 1

def _render_dashboard_html(rows):
    def _fmt(x): return "-" if x is None else x
    head = """
<!doctype html><html lang="th"><meta charset="utf-8">
<title>PV Dashboard</title>
<style>
body{background:#0d0d0d;color:#eee;font-family:system-ui,Arial,sans-serif;padding:24px}
h1{font-size:20px;margin:0 0 12px}
table{width:100%;border-collapse:collapse;background:#121212}
th,td{padding:8px 10px;border-bottom:1px solid #222;text-align:left;font-size:14px}
th{background:#1b1b1b}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-weight:600}
.badge-0{background:#1b3f1b;color:#a6f3a6}
.badge-1{background:#3f311b;color:#ffd18a}
.badge-2{background:#3f1b1b;color:#ff9a9a}
small{color:#aaa}
</style>
<h1>PV Dashboard <small>(ล่าสุด {n} รายการ)</small></h1>
<table><thead><tr>
<th>เวลา</th><th>IP</th><th>ประเภท</th><th>p</th><th>V</th><th>I</th><th>P</th><th>rule?</th>
</tr></thead><tbody>
""".replace("{n}", str(len(rows)))
    body = []
    for r in rows:
        badge = {0:"badge-0",1:"badge-1",2:"badge-2"}.get(r["label_idx"], "badge-0")
        body.append(
            f"<tr>"
            f"<td>{r['ts']}</td>"
            f"<td>{_fmt(r['ip'])}</td>"
            f"<td><span class='badge {badge}'>{r['label_text']}</span></td>"
            f"<td>{_fmt(round(r['proba'],3))}</td>"
            f"<td>{_fmt(r['v'])}</td>"
            f"<td>{_fmt(r['i'])}</td>"
            f"<td>{_fmt(r['p'])}</td>"
            f"<td>{'✓' if r['rule'] else '-'}</td>"
            f"</tr>"
        )
    tail = "</tbody></table></html>"
    return head + "\n".join(body) + tail

# ============== Endpoints ==============
@app.get("/", tags=["root"])
def root():
    return {"ok": True, "msg": "Server is running successfully!"}

@app.get("/recent")
def recent(n: int = 50):
    n = max(1, min(int(n), HISTORY_MAX))
    return list(HISTORY)[:n]

@app.get("/stats")
def stats():
    total = sum(COUNTS.values())
    return {"total": total, "counts": COUNTS}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html = _render_dashboard_html(list(HISTORY)[:100])
    return HTMLResponse(content=html, status_code=200)

@app.get("/window")
def window():
    cnt = Counter(PRED_WINDOW)
    return {"size": len(PRED_WINDOW), "window": list(PRED_WINDOW), "counts": dict(cnt)}

# ============== Predict core ==============
def infer_one(x: List[float]):
    arr = _prepare_input(x)
    with torch.no_grad():
        t = torch.from_numpy(arr)
        y = MODEL(t)
        probs = y.numpy()[0]
        idx = int(np.argmax(probs))
        prob = float(probs[idx])
        return idx, prob

def voltage_helper_vote(v: Optional[float]) -> Optional[int]:
    """
    ตัวช่วยแรงดัน (ใช้แค่เพิ่ม 1 คะแนนในการตัดสิน 'รอบนี้' เท่านั้น)
      v < 37.0      -> 2 'แตก'
      37.0 ≤ v < 39.0 -> 1 'สกปรก'
      v ≥ 39.0      -> 0 'ปกติ'
    """
    if v is None:
        return None
    if v < 37.0:
        return 2
    elif v < 39.0:
        return 1
    else:
        return 0

def majority_with_helper(raw_idx: int, helper_idx: Optional[int]):
    """
    - เก็บผลดิบจากโมเดลลง PRED_WINDOW
    - ใช้คะแนนนับจริงจาก PRED_WINDOW มาหา majority
    - ถ้ามี helper_idx ให้ 'บวกเพิ่ม 1 คะแนน' ให้คลาสนั้นเฉพาะการคำนวณครั้งนี้
      (ไม่เขียนค่า helper ลง PRED_WINDOW เพื่อไม่ให้ลากค่าประวัติ)
    - คืน (final_idx, confidence) โดย confidence = max_count / (len(window)+ (1 ถ้ามี helper))
    """
    PRED_WINDOW.append(raw_idx)

    base_cnt = Counter(PRED_WINDOW)
    extra = 1 if helper_idx is not None else 0
    if helper_idx is not None:
        base_cnt[helper_idx] += 1

    final_idx, max_count = max(base_cnt.items(), key=lambda kv: kv[1])
    denom = len(PRED_WINDOW) + extra
    conf = max_count / denom if denom > 0 else 1.0
    return final_idx, float(conf), helper_idx is not None

@app.post("/predict", response_model=PredictOut, tags=["inference"])
def predict(req: PredictIn, request: Request):
    feats = req.features.data
    v, i, p = req.features.v, req.features.i, req.features.p
    ip = request.client.host if request and request.client else "?"

    log.info("Req from %s data=%s v=%s i=%s p=%s",
             ip, np.round(feats, 5).tolist(), v, i, p)

    # 1) ทำนายผลดิบจากโมเดล
    try:
        raw_idx, _ = infer_one(feats)
    except Exception as e:
        log.exception("Infer error: %s", e)
        raise HTTPException(status_code=400, detail="infer failed")

    # 2) ตัวช่วยแรงดัน (ถ้ามี V)
    hint_idx = voltage_helper_vote(v)

    # 3) Majority + helper (helper เพิ่ม 1 คะแนนเฉพาะรอบนี้)
    label_idx, maj_conf, used_helper = majority_with_helper(raw_idx, hint_idx)
    label_txt = _label_text(label_idx)

    log.info("Vote: raw=%s, helper=%s -> final=%s (conf=%.2f, window=%d)",
             _label_text(raw_idx),
             ("-" if hint_idx is None else _label_text(hint_idx)),
             label_txt, maj_conf, len(PRED_WINDOW))

    # 4) บันทึกผลไว้ดู
    record_result(ip, label_idx, label_txt, maj_conf, v, i, p, used_helper)

    # 5) แจ้งเตือน
    should_alert = (not ONLY_ALERT_ABNORMAL) or (label_idx != 0)
    if should_alert:
        msg = f'พบแผงโซล่าเซลล์ประเภท “{label_idx}” {label_txt} (p={maj_conf:.2f})'
        if SHOW_VIP and any(x is not None for x in (v, i, p)):
            msg += f"\nV={v if v is not None else '-'}  I={i if i is not None else '-'}  P={p if p is not None else '-'}"
        if send_telegram_message(msg):
            log.info("Telegram sent.")
        else:
            log.info("Telegram skipped/failed.")

    return PredictOut(
        label_idx=label_idx,
        label_text=label_txt,
        proba=round(maj_conf, 6),
        v=v, i=i, p=p,
        rule=used_helper
    )

# ============== Uvicorn boot ==============
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
