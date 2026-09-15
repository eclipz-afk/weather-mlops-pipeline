"""Data-quality measurements for cleaned weather datasets."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger("weather_forecasting.audit")


@dataclass(frozen=True)
class GapExample:
    """One interval that is longer than the expected sampling interval."""

    start: str
    end: str
    duration_hours: float
    missing_intervals: int


@dataclass(frozen=True)
class DataQualityReport:
    """Immutable, JSON-ready summary of a weather dataset's quality."""

    rows: int
    unique_timestamps: int
    expected_rows: int
    completeness_ratio: float
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    duplicate_timestamps: int
    missing_hourly_intervals: int
    irregular_intervals: int
    largest_gap_hours: float
    null_counts: dict[str, int] = field(default_factory=dict)
    gap_examples: tuple[GapExample, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        result = asdict(self)
        result["start_timestamp"] = self.start_timestamp.isoformat()
        result["end_timestamp"] = self.end_timestamp.isoformat()
        return result


def _normalize_timestamp(df: pd.DataFrame) -> pd.Series:
    """Return ``timestamp`` as a non-null UTC Series without mutating ``df``."""
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")

    timestamp = df["timestamp"]
    try:
        if pd.api.types.is_datetime64_any_dtype(timestamp):
            if timestamp.dt.tz is None:
                raise ValueError("'timestamp' must be timezone aware, not tz-naive")
            timestamp_utc = timestamp.dt.tz_convert("UTC")
        else:
            timestamp_utc = pd.to_datetime(timestamp, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'timestamp' must be datetime-like: {exc}") from exc

    if timestamp_utc.isna().any():
        raise ValueError("'timestamp' contains NaT values")

    return timestamp_utc.dt.as_unit("ns")


def _frequency_delta(expected_frequency: str) -> pd.Timedelta:
    """Convert a fixed pandas frequency to a positive timedelta."""
    try:
        offset = pd.tseries.frequencies.to_offset(expected_frequency)
        delta = pd.Timedelta(offset.nanos, unit="ns")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"expected_frequency must be a fixed frequency, got {expected_frequency!r}"
        ) from exc

    if delta <= pd.Timedelta(0):
        raise ValueError("expected_frequency must be positive")
    return delta


def _compute_gaps(
    timestamps: pd.Series,
    *,
    expected_delta: pd.Timedelta,
    max_examples: int,
) -> tuple[int, int, float, tuple[GapExample, ...]]:
    """Return missing count, irregular count, largest gap and examples."""
    if len(timestamps) < 2:
        return 0, 0, 0.0, ()

    differences = timestamps.diff().dropna()
    ratios = differences / expected_delta
    rounded_ratios = ratios.round()
    regular_multiple = (ratios - rounded_ratios).abs() < 1e-9
    gap_mask = (ratios > 1) & regular_multiple
    irregular_count = int((~regular_multiple).sum())
    missing_count = int((rounded_ratios[gap_mask].astype(int) - 1).sum())
    largest_gap_hours = float(differences.max() / pd.Timedelta(hours=1))

    examples: list[GapExample] = []
    largest_gaps = ratios[gap_mask].sort_values(ascending=False)
    for index, ratio in largest_gaps.iloc[:max_examples].items():
        examples.append(
            GapExample(
                start=timestamps.iloc[index - 1].isoformat(),
                end=timestamps.iloc[index].isoformat(),
                duration_hours=float(differences.loc[index] / pd.Timedelta(hours=1)),
                missing_intervals=int(round(ratio)) - 1,
            )
        )

    return missing_count, irregular_count, largest_gap_hours, tuple(examples)


def audit_weather(
    df: pd.DataFrame,
    *,
    expected_frequency: str = "1h",
    max_gap_examples: int = 10,
) -> DataQualityReport:
    """Measure data quality for a weather dataset without mutating it."""
    if df.empty:
        raise ValueError("cannot audit an empty DataFrame")
    if max_gap_examples < 0:
        raise ValueError("max_gap_examples must be non-negative")

    timestamp_utc = _normalize_timestamp(df)
    expected_delta = _frequency_delta(expected_frequency)
    duplicate_count = int(timestamp_utc.duplicated().sum())
    unique_timestamps = timestamp_utc.drop_duplicates().sort_values().reset_index(drop=True)
    missing_count, irregular_count, largest_gap_hours, examples = _compute_gaps(
        unique_timestamps,
        expected_delta=expected_delta,
        max_examples=max_gap_examples,
    )
    expected_rows = len(unique_timestamps) + missing_count

    report = DataQualityReport(
        rows=int(len(df)),
        unique_timestamps=int(len(unique_timestamps)),
        expected_rows=expected_rows,
        completeness_ratio=len(unique_timestamps) / expected_rows,
        start_timestamp=unique_timestamps.iloc[0],
        end_timestamp=unique_timestamps.iloc[-1],
        duplicate_timestamps=duplicate_count,
        missing_hourly_intervals=missing_count,
        irregular_intervals=irregular_count,
        largest_gap_hours=largest_gap_hours,
        null_counts={
            str(column): int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        },
        gap_examples=examples,
    )
    logger.info(
        "audit completed: rows=%d duplicates=%d missing=%d irregular=%d completeness=%.6f",
        report.rows,
        report.duplicate_timestamps,
        report.missing_hourly_intervals,
        report.irregular_intervals,
        report.completeness_ratio,
    )
    return report


def require_acceptable_quality(report: DataQualityReport) -> None:
    """Reject a report with data conditions unsafe for model training."""
    if report.duplicate_timestamps:
        raise ValueError(
            f"duplicate timestamps detected: {report.duplicate_timestamps}"
        )
