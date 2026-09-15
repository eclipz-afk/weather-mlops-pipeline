"""Scheduled model retraining and safe promotion through MLflow Registry."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator

from weather_forecasting.models.registry import promote_run
from weather_forecasting.pipelines.train_pipeline import train_and_log


DEFAULT_ARGS = {
    "owner": "weather-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def train_candidate(**context) -> dict[str, object]:
    """Train a candidate model and fail unless its validation MAE beats baseline."""
    params = context["params"]
    result = train_and_log(
        dataset_id=params["dataset_id"],
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        n_jobs=int(params["n_jobs"]),
        run_name=f"airflow-{context['run_id']}",
    )

    if result.model_val_metrics["mae"] >= result.baseline_val_metrics["mae"]:
        raise AirflowFailException(
            "candidate did not outperform the persistence baseline: "
            f"candidate_mae={result.model_val_metrics['mae']:.4f}, "
            f"baseline_mae={result.baseline_val_metrics['mae']:.4f}"
        )

    return asdict(result)


def promote_candidate(**context) -> dict[str, str]:
    """Promote the preceding train task's finished MLflow run to champion."""
    training_result = context["ti"].xcom_pull(
        task_ids="train_candidate",
        key="return_value",
    )
    if not training_result or "mlflow_run_id" not in training_result:
        raise AirflowFailException("train_candidate did not return an MLflow run ID")

    return asdict(promote_run(str(training_result["mlflow_run_id"])))


with DAG(
    dag_id="weather_retraining",
    description="Train a candidate model and promote it only after the baseline gate.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 9, 14),
    schedule="@monthly",
    catchup=False,
    is_paused_upon_creation=True,
    params={
        "dataset_id": "weather-kazan-2000-2026-v1",
        "n_estimators": 100,
        "max_depth": 15,
        "n_jobs": 1,
    },
    tags=["weather", "training", "mlflow"],
) as dag:
    train_model = PythonOperator(
        task_id="train_candidate",
        python_callable=train_candidate,
    )
    promote_model = PythonOperator(
        task_id="promote_candidate",
        python_callable=promote_candidate,
    )

    train_model >> promote_model
