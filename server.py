# server.py
# -------------------------------------------
# ESP8266 PV Inference API + Telegram Alert
# -------------------------------------------

import os
import io
import json
import math
import logging
from typing import List, Optional

import requests
import torch
import torch.nn as nn
import numpy as np
import pickle

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator

# ============== Logging ==============
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ============== FastAPI ==============
app = FastAPI(title="ESP8266 PV Inference API", version="1.1")

# ============== Config ==============
# เงื่อนไขการส่งแจ้งเตือน
ALERT_LABELS = {1, 2}        # 1=สกปรก, 2=แตก (ปรับให้ตรงกับโมเดลของคุณ)
ALERT_PROBA  = 0.80          # ความมั่นใจขั้นต่ำ

# เอาค่าจาก ENV ถ้ามี (ถ้าไม่มีจะใช้ค่าที่ใส่ fallback ในวงเล็บ)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8091687691:AAHRnXog3_BEFTOdbmPX1SkCXPaRst9eCE4")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "8279950843")

# ============== Model / Scaler / Label Encoder ==============
# โมเดลง่าย ๆ (ต้องมีจำนวน input/out ให้ตรงกับที่เทรนไว้)
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

# โหลดไฟล์เสริม (ถ้ามี)
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

# โหลดโมเดล .pt
MODEL = SimpleMLP(in_dim=9, hidden=64, out_dim=3)
if os.path.exists("clf_tested.pt"):
    try:
        state = torch.load("clf_tested.pt", map_location="cpu")
        MODEL.load_state_dict(state, strict=False)  # เผื่อ key mismatch บางกรณี
        log.info("Loaded model weights from clf_tested.pt")
    except Exception as e:
        log.error("Load clf_tested.pt failed: %s", e)
else:
    log.warning("clf_tested.pt not found. Using randomly initialized model.")

MODEL.eval()

# ============== Pydantic Schemas ==============
class FeaturePacket(BaseModel):
    data: List[float] = Field(..., description="feature vector ใช้สำหรับ infer (เช่น 9 ค่า)")
    # ข้อมูลจาก ESP (optional, ไว้โชว์ใน log/monitor)
    v: Optional[float] = Field(None, description="Voltage (opt)")
    i: Optional[float] = Field(None, description="Current (opt)")
    p: Optional[float] = Field(None, description="Power (opt)")

    @validator("data")
    def check_len(cls, v):
        if len(v) == 0:
            raise ValueError("data must not be empty")
        # ถ้าโมเดลคุณเทรน 9 inputs ให้คุมยาว 9 ไว้ (ปรับได้)
        if len(v) != 9:
            raise ValueError("data must have 9 elements")
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
def url_encode(s: str) -> str:
    # encode น้อย ๆ ให้พอส่งข้อความไทยได้
    return requests.utils.quote(s, safe="")

def send_telegram_message(text: str) -> bool:
    token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    if not token or not chat_id:
        log.info("Telegram TOKEN/CHAT_ID not set → skip sending.")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        r = requests.post(url, json=payload, timeout=5)
        if r.ok:
            log.info("Telegram sent OK")
            return True
        log.warning("Telegram send fail: %s %s", r.status_code, r.text)
        return False
    except Exception as e:
        log.warning("Telegram error: %s", e)
        return False


def infer_one(x: List[float]):
    """ ทำ normalize (ถ้ามี scaler), infer, คืน (label_idx, proba) """
    arr = np.array(x, dtype=np.float32).reshape(1, -1)
    if SCALER is not None:
        try:
            arr = SCALER.transform(arr)
        except Exception as e:
            log.warning("Scaler.transform error: %s", e)
    with torch.no_grad():
        t = torch.from_numpy(arr)
        y = MODEL(t)
        probs = y.numpy()[0]
        idx = int(np.argmax(probs))
        prob = float(probs[idx])
        return idx, prob


def label_text_from_idx(idx: int) -> str:
    # ถ้ามี LABEL_ENCODER จะลองกลับเป็น string
    if LABEL_ENCODER is not None:
        try:
            return str(LABEL_ENCODER.inverse_transform([idx])[0])
        except Exception:
            pass
    # fallback คำไทยแบบเดิม
    mapping = {
        0: "ปกติ",
        1: "สกปรก",
        2: "แตก",
    }
    return mapping.get(idx, str(idx))


# ============== Endpoints ==============
@app.get("/", tags=["root"])
def root():
    return {"ok": True, "msg": "Server is running successfully!"}


@app.post("/predict", response_model=PredictOut, tags=["inference"])
def predict(req: PredictIn, request: Request):
    ip = request.client.host if request and request.client else "?"
    feats = req.features.data

    v = req.features.v
    i = req.features.i
    p = req.features.p

    log.info("Req from %s  data=%s  v=%s i=%s p=%s", ip, np.round(feats, 4).tolist(), v, i, p)

    try:
        label_idx, proba = infer_one(feats)
        label_txt = label_text_from_idx(label_idx)
    except Exception as e:
        log.exception("Infer error: %s", e)
        raise HTTPException(status_code=500, detail="infer failed")

    # --- ALERT rule ---
    if (label_idx in ALERT_LABELS) and (proba >= ALERT_PROBA):
        text = f"⚠️ แจ้งเตือน: พบสัญญาณ “{label_txt}” (prob={proba:.2f})"
        if v is not None or i is not None or p is not None:
            text += f"\nV={v if v is not None else '-'}  I={i if i is not None else '-'}  P={p if p is not None else '-'}"
        send_telegram_message(text)

    return PredictOut(
        label_idx=label_idx,
        label_text=label_txt,
        proba=round(proba, 6),
        v=v, i=i, p=p
    )


# ============== Uvicorn boot ==============
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))   # Render จะเซ็ต PORT ให้
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
