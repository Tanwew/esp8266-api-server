from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SensorData(BaseModel):
    voltage: float
    current: float
    power: float

@app.get("/")
def root():
    return {"message": "ESP8266 API Server is running!"}

@app.post("/predict")
def predict(data: SensorData):
    # ตัวอย่าง: คำนวณกำลังจากค่า v และ i (สมมติ)
    energy = data.voltage * data.current
    return {
        "voltage": data.voltage,
        "current": data.current,
        "power": data.power,
        "predicted_energy": energy
    }
