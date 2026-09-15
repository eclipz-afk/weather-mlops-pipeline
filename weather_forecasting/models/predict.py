import logging

import pandas as pd

logger = logging.getLogger("weather_forecasting.model.predict")


def predict(model, X: pd.DataFrame) -> pd.Series:
    """Predict temperature 1h ahead."""
    return pd.Series(model.predict(X))
