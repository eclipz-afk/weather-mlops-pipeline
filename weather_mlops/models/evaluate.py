import logging
import pandas as pd

from weather_mlops.models.metrics import calculate_metrics

logger = logging.getLogger("weather_mlops.model.evaluate")

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Predict with ``model`` on ``X`` and compute regression metrics.

    Parameters
    ----------
    model:
        Fitted sklearn-compatible regressor with a ``predict`` method.
    X:
        Feature matrix.
    y:
        Ground-truth target aligned with ``X``.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"mae"``, ``"rmse"``, ``"r2"``
    """
    logger.info(f"evaluation started: X shape={X.shape}")
    y_pred = model.predict(X)
    metrics = calculate_metrics(y, y_pred)
    logger.info("evaluation completed")
    return metrics