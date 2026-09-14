import logging
import pandas as pd
from weather_mlops.config import LOCATION

logger = logging.getLogger("weather_mlops.features")

LAG_HOURS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)

TARGET_HORIZON: pd.Timedelta = pd.Timedelta(hours=1)

LOCAL_TZ: str = LOCATION["timezone_local"]

SOURCE_COLUMN: str = "temperature_2m"

TARGET_COLUMN: str = "temperature_target_1h"

# Time features --------------------------------------------------
def _build_time_features(df: pd.DataFrame) -> None:
    """Add hour / day_of_week / day_of_year / month (in-place).

    Time-of-day features are computed in the LOCAL timezone of the
    target location, because temperature follows the local solar
    cycle, not UTC. Calendar features (day_of_week, day_of_year,
    month) are unaffected by timezone for a single location but are
    taken from the same local timestamp for consistency.
    """
    local_ts = df["timestamp"].dt.tz_convert(LOCAL_TZ)
    
    df["hour"] = local_ts.dt.hour.astype("int64")
    df["day_of_week"] = local_ts.dt.dayofweek.astype("int64")
    df["day_of_year"] = local_ts.dt.dayofyear.astype("int64")
    df["month"] = local_ts.dt.month.astype("int64")
    

# Lag features -------------------------------------------------------------
def _build_lag_features(df: pd.DataFrame) -> None:
    """Add temperature_lag_<N>h columns (in-place).

    Each lag is built by EXACT timestamp lookup: lag_Nh at time t is
    the value of SOURCE_COLUMN at timestamp t - N hours, or NaN if
    that timestamp does not exist in the data.
    """
    
    lookup = df.set_index("timestamp")[SOURCE_COLUMN]
    
    for hours in LAG_HOURS:
        shifted = df["timestamp"] - pd.Timedelta(hours=hours)
        df[f"temperature_lag_{hours}h"] = lookup.reindex(shifted).to_numpy()
        

# Target -------------------------------------------------------------------
def _build_target(df: pd.DataFrame) -> None:
    """Add the target column (in-place).

    target(t) = SOURCE_COLUMN at timestamp t + TARGET_HORIZON, or NaN
    if that timestamp does not exist in the data.
    """
    lookup = df.set_index("timestamp")[SOURCE_COLUMN]
    target_ts = df["timestamp"] + TARGET_HORIZON
    df[TARGET_COLUMN] = lookup.reindex(target_ts).to_numpy()
    

# Public API ----------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features and target from a cleaned weather DataFrame.

    Parameters
    ----------
    df:
        DataFrame that has already passed ``clean_weather()``:
        tz-aware UTC timestamps, sorted, unique.

    Returns
    -------
    pandas.DataFrame
        A deep copy of the input with added columns:

        * time features: hour, day_of_week, day_of_year, month;
        * lag features:  temperature_lag_1h .. temperature_lag_24h;
        * target:        temperature_target_1h.

        Row order and row count are preserved. A clean RangeIndex is
        returned. NaN values in lag/target are preserved; the
        function never drops or imputes rows.
    """
    
    logger.info(f"features started: shape = {df.shape}")
    
    result = df.copy(deep=True)
    
    _build_time_features(result)
    _build_lag_features(result)
    _build_target(result)
    
    n_target_missing = int(result[TARGET_COLUMN].isna().sum())
    logger.info(f"created features: {result.shape[1]} columns")
    logger.info(f"target missing values: {n_target_missing}")
    
    result = result.reset_index(drop=True)
    
    logger.info(f"features completed: shape = {result.shape}")
    return result    