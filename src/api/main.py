from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
import joblib
import pandas as pd
from src.api.schemas import BatchInferenceRequest, BatchInferenceResponse

# baseline model path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

model_pipeline = None
load_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, load_error

    # checks
    print(f"MAIN FILE: {Path(__file__).resolve()}")
    print(f"PROJECT ROOT: {PROJECT_ROOT}")
    print(f"MODEL PATH: {MODEL_PATH}")
    print(f"MODEL EXISTS: {MODEL_PATH.exists()}")

    try:
        model_pipeline = joblib.load(MODEL_PATH)
        load_error = None
        print(f"MODEL LOADED: {type(model_pipeline)}")

    except Exception as e:
        model_pipeline = None
        load_error = repr(e)
        print(f"MODEL LOADING FAILED: {repr(e)}")

    yield

app = FastAPI(
    title="BTS Flight Delay Severity Inference API",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "model_loaded": str(model_pipeline is not None),
        "debug_error": load_error or "none"
    }

# to correctly redirect path
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")

@app.post(
    "/classify",
    response_model=BatchInferenceResponse,
    status_code=status.HTTP_200_OK,
)

def classify_flights(payload: BatchInferenceRequest) -> BatchInferenceResponse:
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded :'(",
        )

    if not payload.instances:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="instances cannot be empty :'("
        )

    try:
        df_input = pd.DataFrame(payload.instances)
        predictions = model_pipeline.predict(df_input)
        return BatchInferenceResponse(classifications=list(predictions))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"data processing error :'(: {str(e)}",
        )