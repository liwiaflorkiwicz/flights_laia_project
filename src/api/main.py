from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
import joblib
import pandas as pd
import os
import mlflow.sklearn

from src.api.schemas import BatchInferenceRequest, BatchInferenceResponse
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from fastapi import FastAPI

# paths and variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODEL = PROJECT_ROOT / "models" / "model.joblib"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://10.17.0.190:5050")
MODEL_URI = os.getenv("MODEL_URI", "models:/flight-delay-model/latest")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")  # only for CI testing

model_pipeline = None
load_error = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact into memory on application startup (MLflow Registry or local fallback)."""
    global model_pipeline, load_error

    try:
        if LOCAL_MODEL_PATH:
            # only ci tests
            model_pipeline = joblib.load(LOCAL_MODEL_PATH)
            print(f"MODEL LOADED LOCALLY (CI MODE): {type(model_pipeline)}")
        else:
            # production flow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_pipeline = mlflow.sklearn.load_model(MODEL_URI)
            print(f"MODEL LOADED FROM MLFLOW ({MODEL_URI}): {type(model_pipeline)}")

        load_error = None
    except Exception as e:
        model_pipeline = None
        load_error = repr(e)
        print(f"MODEL LOADING FAILED: {load_error}")

    yield

    # memory cleaning
    model_pipeline = None


app = FastAPI(
    title="BTS Flight Delay Severity Inference API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Check API operational status and model load state."""
    return {
        "status": "ok",
        "model_loaded": str(model_pipeline is not None),
        "debug_error": load_error or "none",
    }


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root path directly to OpenAPI interactive documentation."""
    return RedirectResponse(url="/docs")


@app.post(
    "/classify",
    tags=["Inference"],
    response_model=BatchInferenceResponse,
    status_code=status.HTTP_200_OK,
)
def classify_flights(payload: BatchInferenceRequest) -> BatchInferenceResponse:
    """Predict delay severity classes for a batch of flight records."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    if not payload.instances:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="instances cannot be empty"
        )

    try:
        df_input = pd.DataFrame(payload.instances)
        predictions = model_pipeline.predict(df_input)
        return BatchInferenceResponse(classifications=list(predictions))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"data processing error: {str(e)}",
        )

    import os
    import mlflow.sklearn

    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://10.17.0.190:5050")
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Pobranie najnowszej wersji zarejestrowanego modelu z MLflow
    model = mlflow.sklearn.load_model("models:/flight-delay-model/latest")