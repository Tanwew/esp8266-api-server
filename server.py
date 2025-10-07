# -------------------------------------------
# ESP8266 PV Inference API + Telegram Alert
# รองรับฟีเจอร์ 9 หรือ 12 ตัวอัตโนมัติ
# - ถ้าได้ 12 ฟีเจอร์ จะ map เหลือ 9 ตัวให้ตรงกับโมเดล
# -------------------------------------------

import os
import json
import math
import logging
from typing import List, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
import pickle

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator

# ============== Logging ==============
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ============== FastAPI ==============
app = FastAPI(title="ESP8266 PV Inference API", version="1.2")

# ============== Config ==============
# เงื่อนไขการส่งแจ้งเตือน
ALERT_LABELS = {1, 2}        # 1=สกปรก, 2=แตก (ปรับให้ตรงกับโมเดลของคุณ)
ALERT_PROBA  = 0.80          # ความมั่นใจขั้นต่ำ

# ENV (ถ้าไม่ได้ตั้ง จะใช้ค่าด้านหลัง)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8091687691:AAHRnXog3_BEFTOdbmPXlSkCXPaRSt9eCE4")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "8279950843")

# ============== Model / Scaler / Label Encoder ==============
class SimpleMLP(nn.Module):
    """โครง MLP ง่าย ๆ ไว้รับ 9 ฟีเจอร์ และทำนาย 3 คลาส (ปกติ/สกปรก/แตก)"""
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
        MODEL.load_state_dict(state, strict=False)  # กัน key mismatch บางกรณี
        log.info("Loaded model weights from clf_tested.pt")
    except Exception as e:
        log.error("Load clf_tested.pt failed: %s", e)
else:
    log.warning("clf_tested.pt not found. Using randomly initialized model.")

MODEL.eval()

# ============== Schemas ==============
class FeaturePacket(BaseModel):
    # data: 9 หรือ 12 ฟีเจอร์
    data: List[float] = Field(..., description="Feature vector (9 หรือ 12 ตัว)")
    # ช่องเหล่านี้ optional (ถ้าอยากส่งซ้ำมาให้แสดงใน log/alert)
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
def send_telegram_message(text: str) -> bool:
    token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    if not token or not chat_id:
        # ไม่ตั้งค่าไว้ก็ข้ามได้
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        r = requests.post(url, json=payload, timeout=5)
        return bool(r.ok)
    except Exception as e:
        log.warning("Telegram error: %s", e)
        return False

def _map_12_to_9(arr12: np.ndarray) -> np.ndarray:
    """
    Map 12 ฟีเจอร์ → 9 ฟีเจอร์ที่โมเดลต้องการ
    12 = [v_rms, i_rms, p_rms, v_zc, i_zc, p_zc, v_ssc, i_ssc, p_ssc, v_mean, i_mean, p_mean]
    9  = [v_rms, i_rms, p_rms, v_zc, i_zc, v_ssc, i_ssc, v_mean, i_mean]
    """
    idx = [0, 1, 2, 3, 4, 6, 7, 9, 10]
    return arr12[idx]

def _prepare_input(x: List[float]) -> np.ndarray:
    """รับ list (9 หรือ 12) → คืน numpy (1, 9) พร้อม normalize ถ้ามี SCALER"""
    arr = np.array(x, dtype=np.float32)

    # ถ้าได้ 12 → map ให้เหลือ 9 ก่อน
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
    # ถ้ามี LabelEncoder
    if LABEL_ENCODER is not None:
        try:
            return str(LABEL_ENCODER.inverse_transform([idx])[0])
        except Exception:
            pass
    # Fallback
    return {0: "ปกติ", 1: "สกปรก", 2: "แตก"}.get(idx, str(idx))

def infer_one(x: List[float]):
    """คืน (label_idx, proba)"""
    arr = _prepare_input(x)
    with torch.no_grad():
        t = torch.from_numpy(arr)
        y = MODEL(t)
        probs = y.numpy()[0]
        idx = int(np.argmax(probs))
        prob = float(probs[idx])
        return idx, prob

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
    v, i, p = req.features.v, req.features.i, req.features.p
    ip = request.client.host if request and request.client else "?"

    # Log inbound
    log.info("Req from %s data=%s v=%s i=%s p=%s", ip, np.round(feats, 5).tolist(), v, i, p)

    try:
        label_idx, proba = infer_one(feats)
        label_txt = _label_text(label_idx)
    except Exception as e:
        log.exception("Infer error: %s", e)
        raise HTTPException(status_code=400, detail="infer failed")

    # Alert rule
    if (label_idx in ALERT_LABELS) and (proba >= ALERT_PROBA):
        text = f"⚠️ แจ้งเตือน: พบสัญญาณ “{label_txt}” (prob={proba:.2f})"
        if any(x is not None for x in (v, i, p)):
            text += f"\nV={v if v is not None else '-'}  I={i if i is not None else '-'}  P={p if p is not None else '-'}"
        if send_telegram_message(text):
            log.info("Telegram sent successfully.")
        else:
            log.info("Telegram skipped or failed.")

    return PredictOut(
        label_idx=label_idx,
        label_text=label_txt,
        proba=round(proba, 6),
        v=v, i=i, p=p
    )

# ============== Uvicorn boot ==============
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
