import logging

import numpy as np
import pandas as pd
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

logger = logging.getLogger("weather_mlops.model.metrics")

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
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    if len(y_pred) == 0:
        raise ValueError("y_pred is empty")
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length: "
            f"got {len(y_true)} and {len(y_pred)}"
    )

    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    logger.info(
        f"metrics computed mae={mae}, rmse={rmse}, r2={r2}"
    )
    return metrics