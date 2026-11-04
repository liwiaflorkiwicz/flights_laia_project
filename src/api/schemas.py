from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class FlightRecord(BaseModel):
    """
    Schema representing a single flight input instance for model inference.
    Core features are mandatory, while additional BTS metadata fields are optional.
    """

    # Mandatory baseline features required for core prediction logic
    FL_DATE: str = Field(..., description="Flight date in YYYY-MM-DD format")
    OP_UNIQUE_CARRIER: str = Field(
        ..., description="Unique operating carrier code (e.g., 'AA')"
    )
    ORIGIN: str = Field(
        ..., description="Origin airport 3-letter IATA code (e.g., 'JFK')"
    )
    DEST: str = Field(
        ..., description="Destination airport 3-letter IATA code (e.g., 'LAX')"
    )
    CRS_DEP_TIME: int = Field(
        ..., description="Scheduled departure time in HHMM integer format"
    )
    CRS_ARR_TIME: int = Field(
        ..., description="Scheduled arrival time in HHMM integer format"
    )
    CRS_ELAPSED_TIME: float = Field(
        ..., description="Scheduled elapsed flight time in minutes"
    )
    DISTANCE: float = Field(
        ..., description="Flight distance between airports in miles"
    )

    # Optional fields present in full BTS datasets or feature engineering pipelines
    YEAR: Optional[int] = Field(
        default=None, description="Flight year extracted from date"
    )
    MONTH: Optional[int] = Field(
        default=None, description="Flight month (1-12)"
    )
    DAY_OF_MONTH: Optional[int] = Field(
        default=None, description="Day of the month (1-31)"
    )
    DAY_OF_WEEK: Optional[int] = Field(
        default=None, description="Day of the week (1=Monday, 7=Sunday)"
    )
    ORIGIN_STATE_NM: Optional[str] = Field(
        default=None, description="Full state name of origin airport"
    )
    DEST_STATE_NM: Optional[str] = Field(
        default=None, description="Full state name of destination airport"
    )
    DEP_TIME_BLK: Optional[str] = Field(
        default=None, description="Departure time block (e.g., '1200-1259')"
    )

    class Config:
        # Ignore extra fields in payload to avoid HTTP 422 errors on unexpected columns
        extra = "ignore"

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