from typing import Literal
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    event_value: float
    amount: float
    hour: int = Field(ge=0, le=23)
    is_mobile: int = Field(ge=0, le=1)
    lag_event_value_1: float
    lag_amount_1: float
    rolling_mean_event_value_5: float
    rolling_mean_amount_5: float
    rolling_std_event_value_5: float
    rolling_mobile_rate_5: float = Field(ge=0.0, le=1.0)
    events_seen: int = Field(ge=1)
    model_variant: Literal["champion", "challenger"] = "champion"


class PredictionResponse(BaseModel):
    prediction: int
    probability_positive: float
    model_ready: bool


class OnlineEventRequest(BaseModel):
    user_id: int
    event_value: float
    amount: float
    hour: int = Field(ge=0, le=23)
    is_mobile: int = Field(ge=0, le=1)