import pickle
import uvicorn
from fastapi import FastAPI
from typing import Dict, Any

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # categorical
    sex: Literal["male", "female"]
    cp: Literal[
        "asymptomatic",
        "non-anginal",
        "atypical_angina",
        "typical_angina"
    ]
    fbs: Literal["true", "false"]
    restecg: Literal[
        "normal",
        "lv_hypertrophy",
        "st-t_abnormality"
    ]
    exang: Literal["true", "false"]
    slope: Literal["flat", "upsloping", "downsloping"]
    thal: Literal[
        "normal",
        "reversable_defect",
        "fixed_defect"
    ]

    # numerical
    age: int = Field(..., ge=28)
    trestbps: float = Field(..., ge=0)
    chol: float = Field(..., ge=0)
    thalch: float = Field(..., ge=60)
    oldpeak: float = Field(..., ge=-2.6)
    ca: int = Field(..., ge=0)

class PredictResponse(BaseModel):
    health_probability:float
    health: bool


app = FastAPI(title = 'health-prediction')




model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(patient:dict):
    prob = pipeline.predict_proba([patient])[0, 1]
    return float(prob)

@app.post("/predict")
def predict(patient: Patient) -> PredictResponse: #def predict(patient: Dict[str, Any]):
    health = predict_single(patient.dict())  #health = predict_single(patient)
    return PredictResponse(
        health_probability= health,
        health= bool(health<=0.4)
    )


#if __name__ == "__main__":
   # uvicorn.run("predict:app", host="0.0.0.0", port=9696)
