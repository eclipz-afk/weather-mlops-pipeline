from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from weather_forecasting.serving.service import PredictionService


class PredictionRequest(BaseModel):
    """Historical timestamp at which the forecast should be made."""

    timestamp: datetime = Field(
        description="Observed timestamp with timezone, e.g. 2026-09-14T12:00:00Z"
    )


class PredictionResponse(BaseModel):
    timestamp: datetime
    target_timestamp: datetime
    temperature_2m_prediction: float
    unit: str = "°C"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = PredictionService()
    service.load()
    app.state.prediction_service = service
    yield


app = FastAPI(
    title="Weather Temperature Forecast API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    service: PredictionService | None = getattr(request.app.state, "prediction_service", None)
    return {"status": "ok" if service and service.ready else "not_ready", "ready": bool(service and service.ready)}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    try:
        timestamp = pd.Timestamp(payload.timestamp)
        service: PredictionService = request.app.state.prediction_service
        prediction = service.predict_at(timestamp)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    target_timestamp = timestamp + pd.Timedelta(hours=1)
    return PredictionResponse(
        timestamp=timestamp.to_pydatetime(),
        target_timestamp=target_timestamp.to_pydatetime(),
        temperature_2m_prediction=round(prediction, 2),
    )
