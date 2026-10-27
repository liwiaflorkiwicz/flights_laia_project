from typing import Any, Dict, List
from pydantic import BaseModel, Field

# features instances list as input
class BatchInferenceRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(
        #..., example=[{"feature1": 1.0, "feature2": 2.0}, {"feature1": 3.0, "feature2": 4.0}]
    )

# classification output list
class BatchInferenceResponse(BaseModel):
    classifications: List[str] = Field(
        #..., example=["moderate_delay", "on_time"]
    )