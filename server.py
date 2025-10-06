import json, joblib, torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ==== Load artifacts ====
with open("config.json","r",encoding="utf-8") as f:
    cfg = json.load(f)   # e.g. {"WINDOW":128,"input_size":9}
with open("features.json","r",encoding="utf-8") as f:
    feat = json.load(f)  # {"features":[...9 items...], "class_names":["0","1","2"]}

WINDOW = cfg["WINDOW"]
INPUT_SIZE = cfg["input_size"]
FEATURE_ORDER = feat["features"]
CLASS_NAMES   = feat["class_names"]

scaler = joblib.load("scaler.pkl")
lblenc = joblib.load("label_encoder.pkl")

class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=3, p=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(p),
            torch.nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleMLP(in_dim=INPUT_SIZE, hidden=64, out_dim=len(CLASS_NAMES), p=0.3)
model.load_state_dict(torch.load("clf_tested.pt", map_location="cpu"))
model.eval()

app = FastAPI(title="PV Panel Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class InPayload(BaseModel):
    features: list[float]  # length must be INPUT_SIZE
    ref_hint: str | None = None  # optional: "ปกติ" / "สกปรก" / "แตก"

# map model string labels to Thai text
LABEL_MAP = {"0":"ปกติ", "1":"สกปรก", "2":"แตก"}

def apply_ref_bias(prob: np.ndarray, pred_idx: int, ref_hint: str|None):
    """On close calls, bias prediction toward provided ref hint."""
    if not ref_hint:
        return pred_idx
    ref_to_idx = {"ปกติ":0,"สกปรก":1,"แตก":2}
    if ref_hint not in ref_to_idx:
        return pred_idx
    # if margin between top-1 and top-2 < 0.05 → pick REF
    top1 = int(np.argmax(prob))
    sorted_prob = np.sort(prob)
    margin = sorted_prob[-1] - sorted_prob[-2] if prob.size >= 2 else 1.0
    if margin < 0.05:
        return ref_to_idx[ref_hint]
    return pred_idx

@app.get("/health")
def health():
    return {"ok": True, "window": WINDOW, "input_size": INPUT_SIZE}

@app.post("/predict")
def predict(inp: InPayload):
    if len(inp.features) != INPUT_SIZE:
        raise HTTPException(status_code=400, detail=f"Need {INPUT_SIZE} features, got {len(inp.features)}")

    X = np.array(inp.features, dtype=np.float32).reshape(1, -1)
    Xs = scaler.transform(X)

    with torch.no_grad():
        logits = model(torch.from_numpy(Xs))
        prob = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(prob))

    pred_idx = apply_ref_bias(prob, pred_idx, inp.ref_hint)
    label_str = CLASS_NAMES[pred_idx]  # e.g. '0','1','2'
    label_text = LABEL_MAP.get(label_str, label_str)
    return {"label": label_str, "label_text": label_text, "proba": float(prob[pred_idx])}
