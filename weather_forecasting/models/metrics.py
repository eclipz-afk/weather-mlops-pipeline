import logging

import numpy as np
import pandas as pd
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

logger = logging.getLogger("weather_forecasting.model.metrics")

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute MAE, RMSE and R^2 for a regression problem.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Predicted values, same length as ``y_true``.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"mae"``, ``"rmse"``, ``"r2"``.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different lengths, or either
        is empty.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    logger.info(
        f"metrics computed mae={mae}, rmse={rmse}, r2={r2}"
    )
    return metrics
