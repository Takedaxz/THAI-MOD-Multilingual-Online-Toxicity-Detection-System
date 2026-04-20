from typing import List, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)
    next_path: Optional[str] = Field(default=None, max_length=200)


class AuthStatusResponse(BaseModel):
    authenticated: bool
    username: Optional[str] = None
    protect_analyzer: bool = False


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Comment text to classify")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    request_id: str
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


class ReviewedExampleRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=1, max_length=5000)
    reviewed_label: str = Field(..., pattern="^(neg|neu)$")
    predicted_label: Optional[str] = Field(default=None, max_length=20)
    toxicity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_model: Optional[str] = Field(default=None, max_length=200)


class ReviewedExampleResponse(BaseModel):
    status: str
    reviewed_count: int
    path: str
