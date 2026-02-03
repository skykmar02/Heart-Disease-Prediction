import pickle
import uvicorn
from fastapi import FastAPI
from typing import Dict, Any

app = FastAPI(title = 'health-prediction')




model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(patient):
    prob = pipeline.predict_proba(patient)[0, 1]
    return float(prob)

@app.post("/predict")
def predict(patient: Dict[str, Any]):
    health = predict_single(patient)
    return {
        "health_probability":health,
        "health": bool(health<=0.4)
    }


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=True)
