from pathlib import Path
from typing import Dict
from fastapi import FastAPI, HTTPException, status
import joblib
import pandas as pd

from src.api.schemas import BatchInferenceRequest, BatchInferenceResponse

app = FastAPI(
    title="BTS Flight Delay Severity Inference API",
    version="0.1.0",
)

# baseline model path
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "baseline_model.joblib"

# global model variable
model_pipeline = None


@app.on_event("startup")
def load_model():
    global model_pipeline

    if not MODEL_PATH.exists():
        raise RuntimeError(f"No model found: {MODEL_PATH}")

    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Model read correctly: {MODEL_PATH}")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "model_loaded": str(model_pipeline is not None)}


@app.post(
    "/classify",
    response_model=BatchInferenceResponse,
    status_code=status.HTTP_200_OK,
)

def classify_flights(payload: BatchInferenceRequest) -> BatchInferenceResponse:
    if not payload.instances:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="instances cannot be empty :'("
        )

    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="model not loaded :'("
        )

    try:
        df_input = pd.DataFrame(payload.instances)

        # TO DO
        # to update to predict from model
        predictions = model_pipeline.predict(df_input)

        return BatchInferenceResponse(classifications=list(predictions))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"data processing error :'(: {str(e)}",
        )