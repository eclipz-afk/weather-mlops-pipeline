import logging
import pandas as pd

logger = logging.getLogger("weather_forecasting.cleaning")

EXPECTED_INTERVAL = pd.Timedelta(hours=1)
EXPECTED_TZ = "UTC"
MAX_GAPS_LOGGED = 5

# Contract checks ---------------------------------------------------------------------------

def _check_timestamp_contract(timestamp: pd.Series) -> None:
    """Enforce the structural contract on the timestamp column.

    Raises
    ------
    ValueError
        If the timestamp is tz-naive, not in UTC, not sorted, or
        contains duplicates.
    """
    
    if timestamp.dt.tz is None:
        raise ValueError(
            "timestamp must be timezone-aware. Got tz-naive column"
        )
    if str(timestamp.dt.tz) != EXPECTED_TZ:
        raise ValueError(
            f"timestamp must be in {EXPECTED_TZ}. Got {timestamp.dt.tz}"
        )
    if not timestamp.is_monotonic_increasing:
        raise ValueError("timestamp must be stored in ascending order")
    if timestamp.duplicated().any():
        n = int(timestamp.duplicated().sum())
        raise ValueError(f"timestamp must be unique for fixed point. Found {n} duplicate(s)")


# Diagnostics logs --------------------------------------------------------------------------
def _detect_irregular_intervals(timestamp: pd.Series) -> pd.Series:
    """Return the diffs between consecutive timestamps that are not 1h.

    The first row has no predecessor and is skipped.
    """
    if len(timestamp) < 2:
        return pd.Series([], dtype="timedelta64[ns]")
    diffs = timestamp.diff().iloc[1:]
    return diffs[diffs != EXPECTED_INTERVAL]

def _log_missing_values(df: pd.DataFrame) -> None:
    """Log total missing values and per-column breakdown (only non-zero)."""
    per_column = df.isna().sum()
    total = int(per_column.sum())
    
    if total == 0:
        logger.info("total missing values: 0")
        return
    
    logger.warning(f"total missing values: {total}")
    
    non_zero = per_column[per_column > 0]
    if not non_zero.empty:
        logger.warning("missing values by column:")
        for column, count in non_zero.items():
            logger.warning("    %s: %d", column, int(count))


def _log_irregular_intervals(timestamp: pd.Series) -> None:
    """Log the number and the first N positions of irregular intervals"""
    diffs = _detect_irregular_intervals(timestamp)
    n = len(diffs)
    if n == 0:
        logger.info("irregular hourly intervals: 0")
        return
    
    logger.warning(f"irregular hourly intervals detected: {n}")
    
    sample = diffs.head(MAX_GAPS_LOGGED)
    for idx, delta in sample.items():
        prev_ts = timestamp.iloc[idx - 1]
        next_ts = timestamp.iloc[idx]
        logger.warning(f"{prev_ts} -> {next_ts} (delta={delta})")
    
    if n > MAX_GAPS_LOGGED:
        logger.warning(f"    ... and {n - MAX_GAPS_LOGGED} more")
        
# Public API ------------------------------------------------------------
def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a validated weather DataFrame for feature engineering.

    Parameters
    ----------
    df:
        DataFrame that has already passed ``validate_weather()``.

    Returns
    -------
    pandas.DataFrame
        A deep copy of the input, with the same columns and values,
        preserving the validated timestamp order and using a clean RangeIndex.

    Raises
    ------
    ValueError
        If ``timestamp`` is tz-naive, not in UTC, not sorted, or
        contains duplicates.
    """
    logger.info(f"cleaning started: shape = {df.shape}")
    
    _check_timestamp_contract(df["timestamp"])
    
    cleaned = df.copy(deep=True)
    
    _log_missing_values(cleaned)
    _log_irregular_intervals(cleaned["timestamp"])
    
    cleaned = cleaned.reset_index(drop=True)
    
    logger.info(f"cleaning completed: shape = {cleaned.shape}")
    return cleaned
