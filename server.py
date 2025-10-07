# server.py
# -------------------------------------------
# ESP8266 PV Inference API + Telegram Alert
# - รองรับ 9 หรือ 12 ฟีเจอร์อัตโนมัติ
# - มีกฎ Override ตามแรงดัน:
#     V < 38     -> แตก (label 2)
#     38 <= V<39 -> สกปรก (label 1)
#     V >= 39    -> ปกติ (label 0)
# -------------------------------------------

import os
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
app = FastAPI(title="ESP8266 PV Inference API", version="1.3")

# ============== Config ==============
# แจ้งเตือนทุกผลลัพธ์ (ตามที่คุยไว้)
ALERT_LABELS = {0, 1, 2}    # ปกติ/สกปรก/แตก -> แจ้งทุกแบบ
ALERT_PROBA  = 0.50         # ตั้งค่าต่ำไว้เพื่อให้แจ้งแน่ ๆ

# ENV (ถ้าไม่ได้ตั้ง จะใช้ค่าด้านหลัง)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8091687691:AAHRnXog3_BEFTOdbmPXlSkCXPaRSt9eCE4")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "8279950843")

# ============== Model / Scaler / Label Encoder ==============
class SimpleMLP(nn.Module):
    """MLP ง่าย ๆ รับ 9 ฟีเจอร์ ออก 3 คลาส"""
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
    # รับ 9 หรือ 12 ฟีเจอร์
    data: List[float] = Field(..., description="Feature vector (9 or 12)")
    # ค่าส่งประกอบ (optional)
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
    12 -> 9 mapping:
    12 = [v_rms, i_rms, p_rms, v_zc, i_zc, p_zc, v_ssc, i_ssc, p_ssc, v_mean, i_mean, p_mean]
    9  = [v_rms, i_rms, p_rms, v_zc, i_zc, v_ssc, i_ssc, v_mean, i_mean]
    """
    idx = [0, 1, 2, 3, 4, 6, 7, 9, 10]
    return arr12[idx]

def _prepare_input(x: List[float]) -> np.ndarray:
    """รับ list (9/12) -> numpy (1,9) และ normalize ถ้ามี SCALER"""
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
    # ถ้ามี LabelEncoder
    if LABEL_ENCODER is not None:
        try:
            return str(LABEL_ENCODER.inverse_transform([idx])[0])
        except Exception:
            pass
    return {0: "ปกติ", 1: "สกปรก", 2: "แตก"}.get(idx, str(idx))

def _choose_voltage(req: PredictIn) -> Optional[float]:
    """
    เลือกค่าแรงดันที่จะใช้กับกฎ:
    1) req.features.v ถ้ามี
    2) ถ้า data=12 ใช้ v_mean index 9
    3) อย่างอื่น fallback เป็น v_rms index 0
    """
    feats = req.features.data
    if req.features.v is not None:
        return float(req.features.v)
    if len(feats) == 12:
        return float(feats[9])     # v_mean
    # fallback: v_rms (สำหรับ DC จะใกล้เคียงค่า V)
    return float(feats[0])

def infer_one(x: List[float]):
    """คืน (label_idx, proba) จากโมเดล"""
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
    v_in, i_in, p_in = req.features.v, req.features.i, req.features.p
    ip = request.client.host if request and request.client else "?"

    log.info("Req from %s data=%s v=%s i=%s p=%s",
             ip, np.round(feats, 5).tolist(), v_in, i_in, p_in)

    # 1) ทำนายด้วยโมเดลก่อน
    try:
        label_idx, proba = infer_one(feats)
        label_txt = _label_text(label_idx)
    except Exception as e:
        log.exception("Infer error: %s", e)
        raise HTTPException(status_code=400, detail="infer failed")

    # 2) เลือกแรงดันมาใช้กับกฎ
    try:
        v_used = _choose_voltage(req)
    except Exception:
        v_used = None

    # 3) กฎแรงดันทับผลโมเดล (Override)
    rule_applied = False
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

    if rule_applied:
        log.info("Voltage rule applied: V=%.3f => label=%s (p=%.2f)",
                 v_used, label_txt, proba)
    else:
        log.info("Model used (no rule override). label=%s (p=%.2f)", label_txt, proba)

    # 4) ส่ง Telegram (แจ้งทุกผล)
    text = f"พบแผงโซล่าเซลล์ประเภท “{label_txt}” (p={proba:.2f})"
    if v_in is not None or i_in is not None or p_in is not None:
        text += f"\nV={v_in if v_in is not None else '-'}  I={i_in if i_in is not None else '-'}  P={p_in if p_in is not None else '-'}"
    elif v_used is not None:
        text += f"\nV={v_used:.3f}"

    if label_idx in ALERT_LABELS and proba >= ALERT_PROBA:
        ok = send_telegram_message(text)
        log.info("Telegram sent: %s", ok)

    return PredictOut(
        label_idx=label_idx,
        label_text=label_txt,
        proba=round(proba, 6),
        v=v_in if v_in is not None else (v_used if v_used is not None else None),
        i=i_in,
        p=p_in
    )

# ============== Uvicorn boot ==============
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
