from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "weather-mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def ingest_completed_day(**context) -> dict[str, str]:
    """Fetch, validate and write the completed UTC data interval to MinIO."""
    from weather_forecasting.datasets.backfill import backfill_weather

    lag = timedelta(days=int(context["params"]["archive_lag_days"]))
    data_interval_start = context["data_interval_start"] - lag
    data_interval_end = context["data_interval_end"] - lag

    start_date = data_interval_start.date()
    end_date = (data_interval_end - timedelta(microseconds=1)).date()
    backfill_weather(start_date.isoformat(), end_date.isoformat())

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }


with DAG(
    dag_id="weather_ingestion",
    description="Fetch, validate and persist one completed day of weather data.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 9, 14),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    params={"archive_lag_days": 7},
    tags=["weather", "ingestion", "minio"],
) as dag:
    ingest_weather = PythonOperator(
        task_id="ingest_completed_day",
        python_callable=ingest_completed_day,
    )
