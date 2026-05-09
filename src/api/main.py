import logging
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Load model
MODEL_PATH = Path("models/fraud_model.pkl")
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded ✅")
else:
    model = None
    logger.warning("Model not found!")


class Transaction(BaseModel):
    features: list[float]  # 30 features (V1-V28 + scaled_amount + scaled_time)


class PredictionResponse(BaseModel):
    is_fraud: bool
    confidence: float
    model_version: str = "v1"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features")

    features = np.array(transaction.features).reshape(1, -1)
    confidence = float(model.predict_proba(features)[0][1])
    is_fraud = confidence > 0.5

    logger.info(f"Prediction: is_fraud={is_fraud}, confidence={confidence:.4f}")

    return PredictionResponse(
        is_fraud=is_fraud,
        confidence=round(confidence, 4)
    )


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running 🚀"}