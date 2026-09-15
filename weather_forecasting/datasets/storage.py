import io
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Literal, Mapping
import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError

from weather_forecasting.config import (
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_ML_PREFIX,
    S3_PROCESSED_PREFIX,
    S3_RAW_PREFIX,
    S3_REGION,
    S3_REPORTS_PREFIX,
    S3_SECRET_KEY,
)

logger = logging.getLogger("weather_forecasting.storage")
Layer = Literal["raw", "processed"]
MLSPLIT = Literal["train", "val", "test"]
_MISSING_OBJECT_CODES = {"404", "NoSuchKey", "NotFound"}

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

_DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_dataset_id(dataset_id: str) -> str:
    """Validate and return an immutable dataset version identifier."""
    normalized = dataset_id.strip()
    if not _DATASET_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "dataset_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return normalized


def ml_dataset_prefix(dataset_id: str) -> str:
    version = validate_dataset_id(dataset_id)
    return f"{S3_ML_PREFIX}/dataset_id={version}"

def ml_split_key(name: str, *, dataset_id: str | None = None) -> str:
    """Return a Parquet key for a legacy or versioned ML split."""
    if dataset_id is None:
        return f"{S3_ML_PREFIX}/{name}.parquet"
    version = validate_dataset_id(dataset_id)
    return f"{S3_ML_PREFIX}/dataset_id={version}/{name}.parquet"

def ml_manifest_key(dataset_id: str) -> str:
    return f"{ml_dataset_prefix(dataset_id)}/manifest.json"

def ml_success_key(dataset_id: str) -> str:
    return f"{ml_dataset_prefix(dataset_id)}/_SUCCESS"

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

def write_ml_splits(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_id: str,
    source_metadata: Mapping[str, Any],
) -> dict[str, str]:
    """Publish an immutable train/validation/test dataset version.

    Consumers must use a version only when its ``_SUCCESS`` marker exists.
    The marker is written after all Parquet objects and ``manifest.json``.
    """
    version = validate_dataset_id(dataset_id)
    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    _check_split_consistency(splits)

    base_prefix = ml_dataset_prefix(version)
    success_key = ml_success_key(version)
    _ensure_bucket()
    existing_keys = _list_keys(f"{base_prefix}/")
    if existing_keys:
        raise FileExistsError(
            f"dataset version already has objects: dataset_id={version}"
        )

    keys: dict[str, str] = {}
    uploaded_keys: list[str] = []
    try:
        for name, (X, y) in splits.items():
            key = ml_split_key(name, dataset_id=version)
            _put_parquet(_combine_xy(X, y), key)
            keys[name] = key
            uploaded_keys.append(key)

        manifest_key = ml_manifest_key(version)
        manifest = {
            "dataset_id": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_column": y_train.name,
            "feature_columns": list(X_train.columns),
            "splits": {
                name: {"key": keys[name], "rows": len(X)}
                for name, (X, _) in splits.items()
            },
            "source": dict(source_metadata),
        }
        write_json_report(manifest, key=manifest_key)
        keys["manifest"] = manifest_key
        uploaded_keys.append(manifest_key)

        _get_client().put_object(
            Bucket=S3_BUCKET,
            Key=success_key,
            Body=b"",
            ContentType="application/octet-stream",
        )
        keys["success"] = success_key
        return keys
    except Exception:
        _delete_objects_quietly(uploaded_keys)
        raise


def _check_xy_pair(name: str, X: pd.DataFrame, y: pd.Series) -> None:
    """Validate one feature/target pair before it is persisted."""
    if X.empty or y.empty:
        raise ValueError(f"{name} split is empty")
    if len(X) != len(y):
        raise ValueError(f"{name}: X/y length mismatch")
    if not X.index.equals(y.index):
        raise ValueError(f"{name}: X/y indexes are not aligned")
    if y.name is None or not str(y.name):
        raise ValueError(f"{name}: target series must have a name")
    if y.name in X.columns:
        raise ValueError(f"{name}: target column is present in X")
    if X.columns.duplicated().any():
        raise ValueError(f"{name}: X contains duplicate feature columns")
    if X.isna().any().any() or y.isna().any():
        raise ValueError(f"{name}: NaN values found")


def _check_split_consistency(
    splits: Mapping[str, tuple[pd.DataFrame, pd.Series]],
) -> None:
    """Validate all split contracts and their shared schema."""
    for name, (X, y) in splits.items():
        _check_xy_pair(name, X, y)

    reference_columns = list(splits["train"][0].columns)
    target_name = splits["train"][1].name
    for name, (X, y) in splits.items():
        if list(X.columns) != reference_columns:
            raise ValueError(f"{name}: feature columns differ from train")
        if y.name != target_name:
            raise ValueError(f"{name}: target name differs from train")


def _combine_xy(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Return a new frame containing features and the aligned target."""
    frame = X.copy()
    frame[y.name] = y.to_numpy()
    return frame


def _delete_objects_quietly(keys: list[str]) -> None:
    """Best-effort cleanup of a failed version publish without masking errors."""
    if not keys:
        return
    try:
        _get_client().delete_objects(
            Bucket=S3_BUCKET,
            Delete={"Objects": [{"Key": key} for key in keys]},
        )
    except Exception:
        logger.exception("could not clean up partially published ML dataset")

def write_json_report(
    report: Mapping[str, Any],
    *,
    key: str,
) -> str:
    """Serialize a mapping to UTF-8 JSON and upload it to the bucket.

    Parameters
    ----------
    report:
        Any JSON-serializable mapping. Nested dicts, lists, and
        primitive scalars are supported.
    key:
        Full object key inside the bucket, e.g.
        ``reports/data_quality/dataset_id=2026-09-14/report.json``.

    Returns
    -------
    str
        The key that was written (same as ``key``, returned for
        convenience).

    Raises
    ------
    TypeError, ValueError
        If ``report`` contains values that are not valid JSON.
    botocore.exceptions.ClientError
        If the upload fails.
    """
    if not key:
        raise ValueError("key must be a non-empty string")
    
    _ensure_bucket()
    body = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    client = _get_client()
    client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )
    logger.info(f"wrote JSON report: key={key} bytes={len(body.encode('utf-8'))}")
    return key

def data_quality_report_key(dataset_id: str) -> str:
    """Return the object key for a data-quality report.

    Example::

        reports/data_quality/dataset_id=2026-09-14/report.json
    """
    version = validate_dataset_id(dataset_id)
    return f"{S3_REPORTS_PREFIX}/data_quality/dataset_id={version}/report.json"

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

def _object_exists(key: str) -> bool:
    try:
        _get_client().head_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _MISSING_OBJECT_CODES:
            return False
        raise
    return True

def read_ml_manifest(dataset_id: str) -> dict[str, Any]:
    version = validate_dataset_id(dataset_id)
    success_key = ml_success_key(version)
    
    if not _object_exists(success_key):
        raise FileNotFoundError(
            f"ML dataset is not published or does not exist: dataset_id={version}"
        )
    
    manifest_key = ml_manifest_key(version)
    
    try:
        obj = _get_client().get_object(Bucket=S3_BUCKET, Key=manifest_key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _MISSING_OBJECT_CODES:
            raise FileNotFoundError(
                f"published ML dataset has no manifest: dataset_id={version}"
            ) from e
        raise
    
    try:
        manifest = json.loads(obj["Body"].read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(
            f"ML dataset manifest is invalid JSON: dataset: dataset_id={version}"
        ) from e

    if not isinstance(manifest, dict):
        raise ValueError(
            "ML dataset manifest must be a JSON"
        )
    if manifest.get("dataset_id") != version:
        raise ValueError(
            "ML dataset manifest dataset_id does not match requested dataset_id"
        )
    
    target_column = manifest.get("target_column")
    if not isinstance(target_column, str) or not target_column:
        raise ValueError(
            "ML dataset manifest has no valid target_column"
        )
    
    feature_columns = manifest.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(
        isinstance(column, str) for column in feature_columns
    ):
        raise ValueError(
            "ML dataset manifest has no valid feature_columns"
        )
    
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(
            "ML dataset manifest has no splits section"
        )
    
    for split in ("train", "val", "test"):
        split_info = splits.get(split)
        
        if not isinstance(split_info, dict):
            raise ValueError(
                f"ML dataset has no {split} split"
            )
        if split_info.get("key") != ml_split_key(split, dataset_id=version):
            raise ValueError(
                f"ML dataset manifest has invalid {split} key"
            )
    return manifest

def read_ml_split(
    dataset_id: str,
    split: MLSPLIT,
) -> tuple[pd.DataFrame, pd.Series]:
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"unknown ML split: {split}"
        )
    manifest = read_ml_manifest(dataset_id)
    target_column = manifest["target_column"]
    expected_features = manifest["feature_columns"]
    key = manifest["splits"][split]["key"]
    
    try:
        frame = _read_parquet_key(key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _MISSING_OBJECT_CODES:
            raise FileNotFoundError(
                f"ML dataset split is missing: dataset_id={dataset_id}, split={split}"
            ) from e
        raise
    
    if frame.empty:
        raise ValueError(
            f"ML dataset split is empty: dataset_id={dataset_id}"
        )
    if target_column not in frame.columns:
        raise ValueError(
            f"ML dataset split has no target column {target_column!r}: split={split}"
        )
    
    y = frame.pop(target_column)
    if list(frame.columns) != expected_features:
        raise ValueError(
            f"ML dataset split feature schema differs from manifest: split={split}"
        )

    if frame.isna().any().any() or y.isna().any():
        raise ValueError(
            f"ML dataset split contains NaN values: dataset_id={dataset_id}, split={split}"
        )

    y.name = target_column
    return frame, y

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
