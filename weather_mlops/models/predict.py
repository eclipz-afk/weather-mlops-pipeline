import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger("weather_mlops.model.predict")


def load_model(path: str | Path):
    """Load a fitted model from disk."""
    return joblib.load(path)


def predict(model, X: pd.DataFrame) -> pd.Series:
    """Predict temperature 1h ahead."""
    return pd.Series(model.predict(X))