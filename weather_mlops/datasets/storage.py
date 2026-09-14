import io
import logging
from typing import Literal
import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError

from weather_mlops.config import (
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_ML_PREFIX,
    S3_PROCESSED_PREFIX,
    S3_RAW_PREFIX,
    S3_REGION,
    S3_SECRET_KEY,
)

logger = logging.getLogger("weather_mlops.storage")
Layer = Literal["raw", "processed"]

_LAYER_PREFIXES: dict[str, str] = {
    "raw": S3_RAW_PREFIX,
    "processed": S3_PROCESSED_PREFIX,
}

_client = None

def _get_client():
    "Return a cached boto S3 client"
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION,
            config=Config(signature_version="s3v4")
        )
    return _client

def _layer_prefix(layer: Layer) -> str:
    if layer not in _LAYER_PREFIXES:
        raise ValueError(
            f"unknown layer '{layer!r}' expected one of "
            f"{sorted(_LAYER_PREFIXES.keys())}"
        )
    return _LAYER_PREFIXES[layer]

def _day_partition(ts: pd.Timestamp) -> tuple[int, int, int]:
    """Extract the UTC calendar day from a timestamp."""
    ts_utc = ts.tz_convert("UTC") if ts.tz is not None else ts
    return ts_utc.year, ts_utc.month, ts_utc.day



def weather_partition_key(layer: str, timestamp: pd.Timestamp) -> str:
    """Return the object key for one daily weather partition."""
    prefix = _layer_prefix(layer)
    year, month, day = _day_partition(pd.Timestamp(timestamp))
    return f"{prefix}/year={year:04d}/month={month:02d}/day={day:02d}/data.parquet"

def ml_split_key(name: str) -> str:
     """Return the key for an ML split Parquet file (e.g. 'train')"""
     return f"{S3_ML_PREFIX}/{name}.parquet"

def _ensure_bucket() -> None:
    """Create the bucket if it does not exist"""
    client = _get_client()
    try:
        client.head_bucket(Bucket=S3_BUCKET)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchBucket", "NotFound"}:
            client.create_bucket(Bucket=S3_BUCKET)
            logger.info(f"created S3 bucket {S3_BUCKET}")
        else:
            raise


def _df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    return buf.getvalue()

def _put_parquet(df: pd.DataFrame, key: str) -> None:
    client = _get_client()
    body = _df_to_parquet_bytes(df)
    client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/octet-stream"
    )

def write_partition(
    df: pd.DataFrame,
    *,
    layer: Layer,
    skip_existing: bool = False,
) -> list[str]:
    """Group ``df`` by UTC day and write each group as a Parquet partition.

    Parameters
    ----------
    df:
        DataFrame with a tz-aware ``timestamp`` column.
    layer:
        ``"raw"`` or ``"processed"``.
    skip_existing:
        When true, list the layer once and do not overwrite partition keys
        that already exist. This is useful for resuming an interrupted
        backfill.

    Returns
    -------
    list[str]
        Object keys that were written, sorted.

    Notes
    -----
    Re-writing the same key overwrites the object, so the default operation
    is idempotent. With ``skip_existing=True``, an existing object is kept.
    """
    
    if "timestamp" not in df.columns:
        raise ValueError(
            "df must have a 'timestamp' column"
        )
    if df.empty:
        logger.info(
            "write_partition: empty DataFrame, nothing to write"
        )
        return []
    _ensure_bucket()
    existing_keys = (
        set(_list_keys(f"{_layer_prefix(layer)}/")) if skip_existing else set()
    )
    
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df.assign(_partition_day=ts.dt.floor("D"))
    
    written: list[str] = []
    skipped_existing = 0
    
    for day, group in df.groupby("_partition_day", sort=True):
        chunk = (
            group.drop(columns="_partition_day")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        key = weather_partition_key(layer, day)
        if key in existing_keys:
            skipped_existing += 1
            continue
        _put_parquet(chunk, key)
        written.append(key)
        logger.info(
            f"wrote partition: layer={layer} day={day} rows={len(chunk)} key={key}"
        )
    if skipped_existing:
        logger.info(
            "write_partition: skipped %d existing %s partitions",
            skipped_existing,
            layer,
        )
    return sorted(written)

def write_ml_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Persist train/test splits as Parquet objects.

    The target column is appended to each X so that a single file
    contains both features and target, aligned by row.

    Keys::

        processed/ml/train.parquet
        processed/ml/test.parquet
    """
    _ensure_bucket()
    
    train = X_train.copy()
    train[y_train.name] = y_train.values
    test=X_test.copy()
    test[y_test.name] = y_test.values
    
    _put_parquet(train, ml_split_key("train"))
    _put_parquet(test, ml_split_key("test"))
    
    logger.info(
        f"wrote ML split: train={len(train)} rows, test={len(test)} rows"
    )

def _list_keys(prefix: str) -> list[str]:
    client = _get_client()
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            keys.append(obj["Key"])
    return keys

def _read_parquet_key(key: str) -> pd.DataFrame:
    client = _get_client()
    obj = client.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")

def _day_keys_between(
    layer: Layer, start: pd.Timestamp, end: pd.Timestamp
) -> list[str]:
    """Return daily partition keys overlapping the inclusive [start, end] range."""
    start_utc = pd.Timestamp(start).tz_convert("UTC")
    end_utc = pd.Timestamp(end).tz_convert("UTC")
    days = pd.date_range(
        start_utc.floor("D"), end_utc.floor("D"), freq="D", tz="UTC"
    )
    return [weather_partition_key(layer, day) for day in days]

def read_weather(
    *,
    layer: Layer,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Read all weather partitions overlapping [start, end].

    Parameters
    ----------
    layer:
        ``"raw"`` or ``"processed"``.
    start, end:
        Inclusive UTC boundaries.

    Returns
    -------
    pandas.DataFrame
        Concatenated rows, filtered to the exact requested time range, sorted
        by ``timestamp``, RangeIndex reset. Missing partitions are silently
        skipped.
    """
    
    start_utc = pd.Timestamp(start).tz_convert("UTC")
    end_utc = pd.Timestamp(end).tz_convert("UTC")
    keys = _day_keys_between(layer, start_utc, end_utc)
    
    client = _get_client()
    frames: list[pd.DataFrame] = []
    for key in keys:
        try:
            frames.append(_read_parquet_key(key))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                logger.debug(f"missing partition: (skipped): {key}")
                continue
            raise
    
    if not frames:
        logger.info(
            f"read_weather: no partitions found for layer={layer} [{start}, {end}]"
        )
        return pd.DataFrame()
    
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("ns")
    df = df.loc[df["timestamp"].between(start_utc, end_utc, inclusive="both")]
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        f"read_weather: layer={layer} rows={len(df)} partitions={len(frames)}"
    )
    return df

def list_weather_partitions(
    *,
    layer: Layer,
) -> list[str]:
    return sorted(_list_keys(_layer_prefix(layer)))
