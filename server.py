# -------------------------------------------
# ESP8266 PV Inference API + Telegram Alert
# Compatible with FastAPI 0.115.x (Pydantic v2)
# -------------------------------------------

import os
import math
import pickle
import logging
from typing import List, Optional

import requests
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

# ============== Logging ==============
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ============== FastAPI ==============
app = FastAPI(title="ESP8266 PV Inference API", version="1.2")

# ============== Config ==============
ALERT_LABELS = {1, 2}        # 1=สกปรก, 2=แตก
ALERT_PROBA  = 0.80          # แจ้งเมื่อ proba >= 0.8

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8091687691:AAHRnXog3_BEFTOdbmPXlSkCXPaRSt9eCE4")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "8279950843")

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
        log.warning(f"Cannot load {path}: {e}")
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
        log.error(f"Load clf_tested.pt failed: {e}")
else:
    log.warning("clf_tested.pt not found → using random weights")

MODEL.eval()

# ============== Schemas ==============
class FeaturePacket(BaseModel):
    data: List[float] = Field(..., description="Feature vector (9 ค่า)")
    v: Optional[float] = Field(None, description="Voltage (optional)")
    i: Optional[float] = Field(None, description="Current (optional)")
    p: Optional[float] = Field(None, description="Power (optional)")

    @field_validator("data")
    @classmethod
    def validate_data(cls, v: List[float]):
        if len(v) != 9:
            raise ValueError("data ต้องมี 9 ค่าเท่านั้น")
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
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Telegram not configured, skip send.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=10
        )
        if r.ok:
            log.info("Telegram sent successfully.")
            return True
        log.warning(f"Telegram send failed: {r.status_code} {r.text}")
    except Exception as e:
        log.warning(f"Telegram error: {e}")
    return False


def infer_one(x: List[float]):
    arr = np.array(x, dtype=np.float32).reshape(1, -1)
    if SCALER is not None:
        try:
            arr = SCALER.transform(arr)
        except Exception as e:
            log.warning(f"Scaler error: {e}")

    with torch.no_grad():
        t = torch.from_numpy(arr)
        y = MODEL(t)
        probs = y.numpy()[0]
        idx = int(np.argmax(probs))
        prob = float(probs[idx])
        return idx, prob


def label_text_from_idx(idx: int) -> str:
    if LABEL_ENCODER is not None:
        try:
            return str(LABEL_ENCODER.inverse_transform([idx])[0])
        except Exception:
            pass
    mapping = {0: "ปกติ", 1: "สกปรก", 2: "แตก"}
    return mapping.get(idx, f"label_{idx}")


# ============== Endpoints ==============
@app.get("/", tags=["root"])
def root():
    return {"ok": True, "msg": "Server is running successfully!"}


@app.post("/predict", response_model=PredictOut, tags=["inference"])
def predict(req: PredictIn, request: Request):
    ip = request.client.host if request and request.client else "?"
    feats = req.features.data
    v, i, p = req.features.v, req.features.i, req.features.p

    log.info(f"Req from {ip} data={np.round(feats,4).tolist()} v={v} i={i} p={p}")

    try:
        label_idx, proba = infer_one(feats)
        label_text = label_text_from_idx(label_idx)
    except Exception as e:
        log.exception(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="infer failed")

    # ----- Alert -----
    if (label_idx in ALERT_LABELS) and (proba >= ALERT_PROBA):
        msg = f"⚠️ แจ้งเตือน: พบสัญญาณ '{label_text}' (prob={proba:.2f})"
        if v or i or p:
            msg += f"\nV={v or '-'}  I={i or '-'}  P={p or '-'}"
        send_telegram_message(msg)

    return PredictOut(
        label_idx=label_idx,
        label_text=label_text,
        proba=round(proba, 6),
        v=v, i=i, p=p
    )


# ============== Run (for local test only) ==============
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
