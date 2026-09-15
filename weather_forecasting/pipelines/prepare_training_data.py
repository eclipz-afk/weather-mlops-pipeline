import logging
import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd

from weather_forecasting.datasets.audit import(audit_weather, require_acceptable_quality)
from weather_forecasting.datasets.features import build_features
from weather_forecasting.datasets.splitting import split_data
from weather_forecasting.datasets.storage import (
    data_quality_report_key,
    read_weather,
    write_json_report,
    write_ml_splits
)

logger = logging.getLogger("weather_forecasting.prepare_training_data")

@dataclass(frozen=True)
class PrepareDatasetResult:
    dataset_id: str
    source_rows: int
    feature_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    report_key: str
    artifact_keys: dict[str, str]
    
def _to_utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")

def prepare_training_dataset(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dataset_id: str,
    train_ratio: float=0.7,
    val_ratio: float=0.15,
) -> PrepareDatasetResult:
    start_utc = _to_utc_timestamp(start)
    end_utc = _to_utc_timestamp(end)
    
    if start_utc > end_utc:
        raise ValueError(
            "start must be earlier than or equal to end"
        )
    weather_df = read_weather(
        layer="processed",
        start=start_utc,
        end=end_utc
    )
    
    report = audit_weather(weather_df)
    require_acceptable_quality(report)
    
    features_df = build_features(weather_df)
    splits = split_data(
        features_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    report_key = data_quality_report_key(dataset_id)
    
    report_payload = report.to_dict()
    report_payload.update(
        {
            "dataset_id": dataset_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "layer": "processed",
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat(),
            },
        }
    )
    
    write_json_report(report_payload, key=report_key)
    
    artifact_keys = write_ml_splits(
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
        dataset_id=dataset_id,
        source_metadata={
            "layer": "processed",
            "start": start_utc.isoformat(),
            "end": end_utc.isoformat(),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "quality_report_key": report_key,
        },
    )
    
    result = PrepareDatasetResult(
        dataset_id=dataset_id,
        source_rows=len(weather_df),
        feature_rows=len(features_df),
        train_rows=len(splits.X_train),
        val_rows=len(splits.X_val),
        test_rows=len(splits.X_test),
        report_key=report_key,
        artifact_keys=artifact_keys,
    )
    
    logger.info(
        "ML dataset prepared: dataset_id=%s train=%d val=%d test=%d",
        result.dataset_id,
        result.train_rows,
        result.val_rows,
        result.test_rows,
    )
    
    return result

def _parse_cli_dates(
    start_date: str,
    end_date: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start_day = date.fromisoformat(start_date)
        end_day = date.fromisoformat(end_date)
    except ValueError as exc:
        raise ValueError(
            "start-date and end-date must use YYYY-MM-DD format"
        ) from exc

    if start_day > end_day:
        raise ValueError("start-date must not be later than end-date")

    start = pd.Timestamp(start_day, tz="UTC")
    end = (
        pd.Timestamp(end_day, tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(hours=1)
    )

    return start, end

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and publish a versioned weather ML dataset."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Inclusive source date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Inclusive source date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Immutable dataset version, e.g. weather-kazan-2000-2026-v1.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start, end = _parse_cli_dates(args.start_date, args.end_date)

    result = prepare_training_dataset(
        start=start,
        end=end,
        dataset_id=args.dataset_id,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print(f"dataset_id: {result.dataset_id}")
    print(
        "rows: "
        f"source={result.source_rows}, "
        f"features={result.feature_rows}, "
        f"train={result.train_rows}, "
        f"val={result.val_rows}, "
        f"test={result.test_rows}"
    )
    print(f"quality report: {result.report_key}")

    for name, key in result.artifact_keys.items():
        print(f"{name}: {key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
