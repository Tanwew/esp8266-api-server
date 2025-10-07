# =====================================================
# ESP8266 PV Inference API + Telegram Alert (v1.3)
# =====================================================

import os, json, torch, pickle, logging, numpy as np, requests
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
from typing import List, Optional

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI(title="ESP8266 PV Inference API", version="1.3")

# ---------------- Config ----------------
ALERT_LABELS = {1, 2}        # 1=สกปรก, 2=แตก
ALERT_PROBA  = 0.80          # ความมั่นใจขั้นต่ำ
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8091687691:AAHRnXog3_BEFTOdbmPXlSkCXPaRSt9eCE4")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "8279950843")

# ---------------- Model ----------------
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
    def forward(self, x): return self.net(x)

def _safe_load_pickle(path):
    if not os.path.exists(path): return None
    try:
        with open(path, "rb") as f: return pickle.load(f)
    except: return None

SCALER = _safe_load_pickle("scaler.pkl")
LABEL_ENCODER = _safe_load_pickle("label_encoder.pkl")

MODEL = SimpleMLP()
if os.path.exists("clf_tested.pt"):
    state = torch.load("clf_tested.pt", map_location="cpu")
    MODEL.load_state_dict(state, strict=False)
    log.info("✅ Loaded clf_tested.pt")
else:
    log.warning("⚠️ No clf_tested.pt found, using random weights.")
MODEL.eval()

# ---------------- Schemas ----------------
class FeaturePacket(BaseModel):
    data: List[float]
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

# ---------------- Utils ----------------
def send_telegram_message(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        return bool(r.ok)
    except Exception as e:
        log.warning("Telegram error: %s", e)
        return False

def _map_12_to_9(arr):
    idx = [0,1,2,3,4,6,7,9,10]
    return arr[idx]

def _prepare_input(x):
    arr = np.array(x, dtype=np.float32)
    if arr.shape[0] == 12: arr = _map_12_to_9(arr)
    elif arr.shape[0] != 9: raise ValueError("Expect 9 or 12 features")
    arr = arr.reshape(1,-1)
    if SCALER is not None:
        try: arr = SCALER.transform(arr)
        except: pass
    return arr

def _label_text(idx):
    if LABEL_ENCODER is not None:
        try: return str(LABEL_ENCODER.inverse_transform([idx])[0])
        except: pass
    return {0:"ปกติ",1:"สกปรก",2:"แตก"}.get(idx,str(idx))

def infer_one(x):
    arr = _prepare_input(x)
    with torch.no_grad():
        y = MODEL(torch.from_numpy(arr))
        probs = y.numpy()[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])

# ---------------- API ----------------
@app.get("/")
def root(): return {"ok":True,"msg":"Server running"}

@app.post("/predict", response_model=PredictOut)
def predict(req: PredictIn, request: Request):
    feats = req.features.data
    v,i,p = req.features.v, req.features.i, req.features.p
    ip = request.client.host if request.client else "?"

    log.info(f"Req from {ip} data={np.round(feats,4).tolist()} v={v} i={i} p={p}")
    try:
        label_idx, proba = infer_one(feats)
        label_txt = _label_text(label_idx)
    except Exception as e:
        log.exception("Infer failed: %s", e)
        raise HTTPException(status_code=422, detail="Infer failed")

    if label_idx in ALERT_LABELS and proba >= ALERT_PROBA:
        text = f"⚠️ พบสัญญาณ “{label_txt}” (p={proba:.2f})"
        if v or i or p:
            text += f"\nV={v} I={i} P={p}"
        if send_telegram_message(text):
            log.info("Telegram sent successfully.")
        else:
            log.info("Telegram skipped or failed.")

    return PredictOut(label_idx=label_idx, label_text=label_txt, proba=round(proba,4), v=v, i=i, p=p)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT",10000)))
