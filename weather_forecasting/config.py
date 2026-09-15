from pathlib import Path
from dotenv import load_dotenv
import os
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
PROCESSED_WEATHER_DIR = PROJECT_ROOT / "data" / "processed" / "weather"
# Service artifacts (meta, reports, API samples) live OUTSIDE the partition
# tree so that Parquet dataset discovery never trips over non-parquet files
SERVICE_DIR = PROJECT_ROOT / "data" / "raw" / "_service"


# Geographic location (fixed for v1)
LOCATION: dict[str, str | float] = {
    "latitude": 55.752085,
    "longitude": 48.744618,
    "name": "Innopolis, Republic of Tatarstan, Russia",
    "timezone_local": "Europe/Moscow",
    "local_utc_offset_hours": 3,
}

# Hourly variables requested from the API
HOURLY_VARIABLES: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

# Columns that must be present in every validated weather DataFrame
REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "latitude",
    "longitude",
)


# Meteorological columns - nullable
NULLABLE_WEATHER_COLUMNS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
)

MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)
MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "weather-temperature-forecast",
)
MLFLOW_REGISTERED_MODEL_NAME: str = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "weather-temperature-forecaster",
)
MLFLOW_MODEL_ALIAS: str = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
SERVING_MODEL_URI: str = os.getenv(
    "SERVING_MODEL_URI",
    f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}",
)
SERVING_DATASET_ID: str = os.getenv(
    "SERVING_DATASET_ID",
    "weather-kazan-2000-2026-v1",
)

# S3-compatible object storage (MinIO locally, AWS S3 in production)
# Credentials intentionally come only from environment variables.
S3_ENDPOINT_URL: str | None = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET: str = os.getenv("S3_BUCKET", "weather-mlops")
S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
S3_ACCESS_KEY: str | None = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY: str | None = os.getenv("S3_SECRET_KEY")

# Prefixes are configurable so that the same storage module can be used with
# a local MinIO bucket and a production S3 bucket without code changes.
S3_RAW_PREFIX: str = os.getenv("S3_RAW_PREFIX", "raw/weather").strip("/")
S3_PROCESSED_PREFIX: str = os.getenv(
    "S3_PROCESSED_PREFIX", "processed/weather"
).strip("/")
S3_ML_PREFIX: str = os.getenv("S3_ML_PREFIX", "processed/ml").strip("/")

S3_REPORTS_PREFIX: str = os.getenv(
    "S3_REPORTS_PREFIX", "reports"
).strip("/")
