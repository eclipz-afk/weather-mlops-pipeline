"""Scheduled model retraining and safe promotion through MLflow Registry."""

from dataclasses import asdict
from datetime import datetime, timedelta
from hashlib import sha256

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "weather-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def prepare_dataset(**context) -> str:
    """Publish fresh chronological splits, or reuse this run's completed dataset."""
    import pandas as pd

    from weather_forecasting.datasets.storage import read_ml_manifest
    from weather_forecasting.pipelines.prepare_training_data import prepare_training_dataset

    params = context["params"]
    reference = context["data_interval_end"]
    if context["dag_run"].run_type == "manual":
        reference = context["logical_date"]
    cutoff = pd.Timestamp(reference).tz_convert("UTC").floor("D")
    cutoff -= pd.Timedelta(days=int(params["archive_lag_days"]))
    start = pd.Timestamp(params["start_date"], tz="UTC")
    end = cutoff - pd.Timedelta(hours=1)
    if params.get("end_date"):
        requested_end = pd.Timestamp(params["end_date"], tz="UTC")
        requested_end += pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        if requested_end > end:
            raise AirflowFailException("end_date is newer than the archive availability cutoff")
        end = requested_end
    version = sha256(context["run_id"].encode("utf-8")).hexdigest()[:16]
    dataset_id = f"weather-innopolis-{end:%Y%m%d}-{version}"
    expected_source = {
        "layer": "processed",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "train_ratio": 0.70,
        "val_ratio": 0.15,
    }
    try:
        manifest = read_ml_manifest(dataset_id)
    except FileNotFoundError:
        prepare_training_dataset(start=start, end=end, dataset_id=dataset_id)
    else:
        if any(manifest["source"].get(key) != value for key, value in expected_source.items()):
            raise AirflowFailException("existing dataset does not match this run's source interval")
    return dataset_id


def train_candidate(**context) -> dict[str, object]:
    """Train a candidate model and fail unless its validation MAE beats baseline."""
    from weather_forecasting.pipelines.train_pipeline import train_and_log

    params = context["params"]
    dataset_id = context["ti"].xcom_pull(task_ids="prepare_dataset")
    result = train_and_log(
        dataset_id=dataset_id,
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
    from weather_forecasting.models.registry import promote_run

    training_result = context["ti"].xcom_pull(
        task_ids="train_candidate",
        key="return_value",
    )
    return asdict(promote_run(str(training_result["mlflow_run_id"])))


with DAG(
    dag_id="weather_retraining",
    description="Prepare fresh splits, train a candidate and promote it after the baseline gate.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 9, 14),
    schedule="@monthly",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    params={
        "start_date": "2020-01-01",
        "end_date": "",
        "archive_lag_days": 7,
        "n_estimators": 64,
        "max_depth": 12,
        "n_jobs": 1,
    },
    tags=["weather", "training", "mlflow"],
) as dag:
    prepare_data = PythonOperator(
        task_id="prepare_dataset",
        python_callable=prepare_dataset,
    )
    train_model = PythonOperator(
        task_id="train_candidate",
        python_callable=train_candidate,
    )
    promote_model = PythonOperator(
        task_id="promote_candidate",
        python_callable=promote_candidate,
    )

    prepare_data >> train_model >> promote_model
