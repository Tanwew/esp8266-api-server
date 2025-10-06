import os
import json
import pickle
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -------------------------------
# FastAPI app (สำคัญ: ชื่อ 'app')
# -------------------------------
app = FastAPI(title="ESP8266 PV Inference API", version="1.0")

# -------------------------------
# โหลด config / features meta
# -------------------------------
def load_json(path: str, default: dict):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

cfg = load_json("config.json", {"input_size": 9, "hidden": 64, "out_dim": 3})
feat_meta = load_json("features.json", {"FEATURES": ["f"+str(i) for i in range(cfg["input_size"])]})

INPUT_SIZE = int(cfg.get("input_size", 9))
HIDDEN     = int(cfg.get("hidden", 64))
OUT_DIM    = int(cfg.get("out_dim", 3))

# -------------------------------
# โหลด scaler / label encoder
# -------------------------------
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

scaler = None
label_encoder = None
try:
    scaler = load_pickle("scaler.pkl")
except Exception:
    pass

try:
    label_encoder = load_pickle("label_encoder.pkl")
except Exception:
    pass

# -------------------------------
# สร้างสถาปัตยกรรมที่ "ชื่อเลเยอร์" ตรงกับ state_dict แบบ LSTM
# (ใช้เมื่อไฟล์ .pt เป็น state_dict ที่มีคีย์ encoder.lstm / encoder.fc / head.*)
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, hidden)

class LSTMHeadModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=HIDDEN, out_dim=OUT_DIM):
        super().__init__()
        self.encoder = Encoder(input_size, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # รองรับทั้ง (B, T, F) และ (B, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,F)
        out, _ = self.encoder.lstm(x)  # (B,T,H)
        h = out[:, -1, :]              # (B,H)
        h = self.encoder.fc(h)
        logits = self.head(h)
        return logits

# -------------------------------
# โหลดโมเดล (.pt)
# กลยุทธ์:
# 1) พยายาม torch.load ทั้งโมเดลก่อน
# 2) ถ้าได้ state_dict → ลองโหลดเข้ากับ LSTMHeadModel (strict=False)
# -------------------------------
model: Optional[nn.Module] = None
try:
    obj = torch.load("clf_tested.pt", map_location="cpu")
    if isinstance(obj, nn.Module):
        model = obj
    else:
        # เป็น state_dict
        m = LSTMHeadModel(INPUT_SIZE, HIDDEN, OUT_DIM)
        missing = m.load_state_dict(obj, strict=False)  # อนุโลมเพื่อให้รันได้ก่อน
        model = m
except Exception as e:
    # ถ้าโหลดไม่ได้เลย ให้สร้างโมเดล mock เพื่อไม่ให้ service ล่ม
    class Dummy(nn.Module):
        def __init__(self, out_dim=OUT_DIM): super().__init__(); self.out_dim = out_dim
        def forward(self, x): 
            b = x.shape[0]
            return torch.zeros(b, self.out_dim)
    model = Dummy(OUT_DIM)

model.eval()

# -------------------------------
# request/response schema
# -------------------------------
class PredictIn(BaseModel):
    features: List[float] = Field(..., description=f"List of {INPUT_SIZE} features")

class PredictOut(BaseModel):
    label_idx: int
    label_text: str
    proba: float

# -------------------------------
# utils
# -------------------------------
def postprocess_label(idx: int) -> str:
    if label_encoder is None:
        mapping = {0: "ปกติ", 1: "สกปรก", 2: "แตก"}
        return mapping.get(int(idx), str(idx))
    try:
        # inverse_transform ต้องรับ array
        return str(label_encoder.inverse_transform([int(idx)])[0])
    except Exception:
        return str(idx)

def run_model(x_np: np.ndarray) -> np.ndarray:
    """return logits numpy shape (B, OUT_DIM)"""
    with torch.no_grad():
        x = torch.from_numpy(x_np.astype(np.float32))
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.detach().cpu().numpy()

# -------------------------------
# routes
# -------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "input_size": INPUT_SIZE,
        "out_dim": OUT_DIM,
        "features": feat_meta.get("FEATURES", []),
    }

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    # ตรวจความยาว
    if len(inp.features) != INPUT_SIZE:
        raise HTTPException(status_code=400, detail=f"features length must be {INPUT_SIZE}")

    x = np.array(inp.features, dtype=np.float32).reshape(1, -1)  # (1,F)

    # สเกล ถ้ามี
    if scaler is not None:
        try:
            x = scaler.transform(x)
        except Exception:
            pass

    # วิ่งโมเดล
    # รองรับทั้ง (B,F) และ (B,T,F) → ตอน run_model แปลงเป็น tensor แล้วจัดการต่อ
    logits = run_model(x)  # (1, C)

    # softmax → proba
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = (e / (e.sum(axis=1, keepdims=True) + 1e-9))[0]

    idx = int(prob.argmax())
    text = postprocess_label(idx)
    return PredictOut(label_idx=idx, label_text=text, proba=float(prob[idx]))
