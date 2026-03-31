from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Comment text to classify")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    text: str
    processed_text: str
    predicted_label: str
    toxic_score: float
    confidence: float
    threshold: float
    recommendation: str
    source_model: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

