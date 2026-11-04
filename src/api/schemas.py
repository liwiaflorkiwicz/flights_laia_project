from typing import Any, Dict, List
from pydantic import BaseModel, Field

# features instances list as input
class BatchInferenceRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(
        ...,
        json_schema_extra={
            "example": [
                {
                    "FL_DATE": "2025-05-15",
                    "OP_UNIQUE_CARRIER": "AA",
                    "ORIGIN": "JFK",
                    "DEST": "LAX",
                    "CRS_DEP_TIME": 1400,
                    "CRS_ARR_TIME": 1730,
                    "CRS_ELAPSED_TIME": 330.0,
                    "DISTANCE": 2475.0,
                }
            ]
        },
    )

# classification output list
class BatchInferenceResponse(BaseModel):
    classifications: List[str] = Field(
        ...,
        json_schema_extra={
            "example": ["on_time", "moderate_delay", "severe_delay"]
        },
    )