from fastapi import FastAPI, HTTPException
import pandas as pd

from api.deps import predictor, registry, online_builder
from api.schemas import OnlineEventRequest, PredictionRequest, PredictionResponse
from core.config import FEATURE_COLUMNS, settings
from core.logging import get_logger, setup_logging
from monitoring.metrics import metrics

from api.deps import (
    champion_predictor,
    challenger_predictor,
    champion_registry,
    challenger_registry,
    online_builder,
)

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title=settings.app_name)


@app.get("/")
def root():
    return {"status": "ok", "service": settings.app_name, "env": settings.app_env}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    champion_predictor.reload_if_exists()
    challenger_predictor.reload_if_exists()
    return {
        "champion_ready": champion_predictor.is_ready(),
        "challenger_ready": challenger_predictor.is_ready(),
        "champion_exists": champion_registry.exists(),
        "challenger_exists": challenger_registry.exists(),
    }


@app.get("/metrics")
def get_metrics():
    return metrics.snapshot()


@app.get("/model_info")
def model_info():
    return {
        "champion": champion_registry.load_metadata(),
        "challenger": challenger_registry.load_metadata(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    champion_predictor.reload_if_exists()
    challenger_predictor.reload_if_exists()

    predictor = champion_predictor if req.model_variant == "champion" else challenger_predictor

    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"{req.model_variant} model is not loaded",
        )

    row = req.model_dump()
    row.pop("model_variant")
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    pred = int(predictor.predict(df)[0])
    prob = float(predictor.predict_proba(df)[0][1])

    metrics.log_prediction(prob)
    logger.info(
        "prediction_request variant=%s prediction=%s probability_positive=%.6f",
        req.model_variant,
        pred,
        prob,
    )

    return PredictionResponse(
        prediction=pred,
        probability_positive=prob,
        model_ready=True,
    )


@app.post("/online_features")
def online_features(req: OnlineEventRequest):
    result = online_builder.update(
        user_id=req.user_id,
        event_value=req.event_value,
        amount=req.amount,
        hour=req.hour,
        is_mobile=req.is_mobile,
    )
    ready = result is not None
    metrics.log_online_update(ready=ready)

    return {
        "ready": ready,
        "features": result,
        "warmup": settings.online_warmup,
    }