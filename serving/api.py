from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from feature_store.offline_store import load_offline_features
from feature_store.online_store import OnlineFeatureStore
from serving.inference import run_inference
from core.config import TASK_TYPE

app = FastAPI()

offline_df = load_offline_features()
online_store = OnlineFeatureStore()
online_store.put_many(offline_df.to_dict(orient="records"))


class PredictRequest(BaseModel):
    user_id: int = Field(..., example=12)


@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /predict or open /docs"}


@app.post("/predict")
def predict(req: PredictRequest):
    pred = run_inference(online_store, req.user_id)
    if pred is None:
        raise HTTPException(status_code=404, detail="user_id not found")

    return {
        "user_id": req.user_id,
        "task_type": TASK_TYPE,
        "prediction": float(pred) if TASK_TYPE == "regression" else int(pred),
    }