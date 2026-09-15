from dataclasses import dataclass

import mlflow.pyfunc
import pandas as pd

from weather_forecasting.config import SERVING_DATASET_ID, SERVING_MODEL_URI
from weather_forecasting.datasets.features import LAG_HOURS, build_features
from weather_forecasting.datasets.storage import read_ml_manifest, read_weather


@dataclass
class PredictionService:
    """Stateful serving facade; the MLflow model is loaded once at startup."""

    model_uri: str = SERVING_MODEL_URI
    dataset_id: str = SERVING_DATASET_ID
    _model: object | None = None
    _feature_columns: list[str] | None = None

    def load(self) -> None:
        """Load the promoted model and its immutable feature schema."""
        manifest = read_ml_manifest(self.dataset_id)
        self._feature_columns = list(manifest["feature_columns"])
        self._model = mlflow.pyfunc.load_model(self.model_uri)

    @property
    def ready(self) -> bool:
        return self._model is not None and self._feature_columns is not None

    def predict_at(self, timestamp: pd.Timestamp) -> float:
        """Predict temperature one hour after an observed UTC timestamp.

        The current observation and all required temperature lags are read
        from the processed object-storage layer. This guarantees identical
        feature construction for training and inference.
        """
        if not self.ready:
            raise RuntimeError("prediction service is not initialized")

        ts = _to_utc(timestamp)
        history = read_weather(
            layer="processed",
            start=ts - pd.Timedelta(hours=max(LAG_HOURS)),
            end=ts,
        )
        featured = build_features(history)
        row = featured.loc[featured["timestamp"] == ts]
        if row.empty:
            raise ValueError(f"no processed weather observation for {ts.isoformat()}")

        X = row.loc[:, self._feature_columns]
        if X.isna().any().any():
            missing = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"insufficient weather history for {ts.isoformat()}: missing {missing}"
            )

        prediction = self._model.predict(X)
        return float(prediction[0])


def _to_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        raise ValueError("timestamp must include a timezone offset, for example Z")
    return timestamp.tz_convert("UTC")
