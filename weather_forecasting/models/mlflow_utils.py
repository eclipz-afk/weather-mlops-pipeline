import logging
import os
from typing import Any

import mlflow
import mlflow.sklearn

from weather_forecasting.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

logger = logging.getLogger("weather_forecasting.model.mlflow_utils")

METRIC_PREFIX = "val_"

MODEL_ARTIFACT_PATH = "model"

def log_to_mlflow(
    model,
    metrics: dict[str, float],
    *,
    run_name: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> str:
    """Log a single training run to MLflow.

    Parameters
    ----------
    model:
        Fitted sklearn-compatible model.
    metrics:
        Metrics dict, e.g. from ``evaluate_model``.
    run_name:
        Optional human-readable run name.
    extra_params:
        Additional parameters to log (beyond ``model.get_params()``).

    Returns
    -------
    str
        The MLflow run ID.

    Raises
    ------
    mlflow.exceptions.MlflowException
        If the tracking server is unreachable or logging fails.
    """
    
    # With Dockerized MinIO, the hostname ``minio`` is resolvable only inside
    # the Compose network. Disable MLflow's direct multipart mode because it
    # would expose presigned URLs with that internal hostname to this Windows
    # client. Ordinary proxy uploads keep the network boundary inside MLflow.
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=run_name) as run:
        params = dict(model.get_params())
        if extra_params:
            params.update(extra_params)
        
        mlflow.log_params(_stringify(params))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name=MODEL_ARTIFACT_PATH)
        
        logger.info(f"mlflow run logged: run_id={run.info.run_id} experiment={MLFLOW_EXPERIMENT_NAME}")
        return run.info.run_id

def _stringify(params: dict[str, Any]) -> dict[str, str]:
    """MLflow expects flat string/number params; convert the rest"""
    return {k: str(v) for k, v in params.items()}

def _prefix_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Prepend `METRIC_PREFIX` to every metric key"""
    return {f"{METRIC_PREFIX}{k}": float(v) for k, v in metrics.items()}
