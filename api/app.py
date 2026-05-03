from fastapi import FastAPI, HTTPException
import pandas as pd

from api.schemas import OnlineEventRequest, PredictionRequest, PredictionResponse
from core.config import FEATURE_COLUMNS, settings
from core.logging import get_logger, setup_logging
from monitoring.metrics import metrics
from monitoring.drift import detect_drift, load_baseline

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

    champion_ready = champion_predictor.is_ready()
    challenger_ready = challenger_predictor.is_ready()

    return {
        # ✅ backward compatibility for tests
        "ready": champion_ready,

        # ✅ new production fields
        "champion_ready": champion_ready,
        "challenger_ready": challenger_ready,
        "champion_exists": champion_registry.exists(),
        "challenger_exists": challenger_registry.exists(),
    }


@app.get("/metrics")
def get_metrics():
    return metrics.snapshot()


@app.get("/model_info")
def model_info():
    champion_meta = champion_registry.load_metadata()
    challenger_meta = challenger_registry.load_metadata()

    return {
        # ✅ backward compatibility for tests
        "status": champion_meta.get("status", "unknown"),

        # ✅ full production info
        "champion": champion_meta,
        "challenger": challenger_meta,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    champion_predictor.reload_if_exists()
    challenger_predictor.reload_if_exists()

    predictor = (
        champion_predictor
        if req.model_variant == "champion"
        else challenger_predictor
    )

    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"{req.model_variant} model is not loaded",
        )

    row = req.model_dump()
    row.pop("model_variant", None)

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    pred = int(predictor.predict(df)[0])
    prob = float(predictor.predict_proba(df)[0][1])

    metrics.log_prediction(prob)

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

@app.post("/drift")
def drift(req: PredictionRequest):
    row = req.model_dump()
    row.pop("model_variant", None)

    baseline = load_baseline(settings.drift_baseline_path)

    if not baseline:
        raise HTTPException(
            status_code=404,
            detail="Drift baseline not found. Run training pipeline first.",
        )

    drift_detected, drift_report = detect_drift(
        live_features=row,
        baseline=baseline,
        threshold=settings.drift_zscore_threshold,
    )

    return {
        "drift_detected": drift_detected,
        "threshold": settings.drift_zscore_threshold,
        "report": drift_report,
    }