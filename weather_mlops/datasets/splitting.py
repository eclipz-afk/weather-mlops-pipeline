import logging

from typing import NamedTuple
import pandas as pd

from weather_mlops.datasets.features import TARGET_HORIZON, TARGET_COLUMN

logger = logging.getLogger("weather_mlops.splitting")

NON_FEATURE_COLUMNS: tuple[str, ...] = ("timestamp", TARGET_COLUMN)

# Result container -----------------------------------------------------

class SplitResult(NamedTuple):
    """Named tuple with six aligned X/y pairs.

    Supports both attribute access and positional unpacking::

        result = split_data(df)
        result.X_train

        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    """
    
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


# Helpers ---------------------------------------------------------------

def _validate_ratios(train_ratio: float, val_ratio: float) -> float:
    """Validate train/val ratios and return the computed test ratio.

    Raises
    ------
    ValueError
        If any ratio is not in (0, 1) or if train + val >= 1.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            f"train_ratio must be in (0, 1), but got {train_ratio}"
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(
            f"val_ratio must be in (0, 1), but got {train_ratio}"
        )
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1; got "
            f"{train_ratio} + {val_ratio} = {train_ratio + val_ratio}"
        )
    return test_ratio

def _apply_embargo(train_df: pd.DataFrame, train_end_boundary: pd.Timestamp, horizon: pd.Timedelta) -> pd.DataFrame:
    """Drop train rows whose target falls strictly after the boundary.

    A train row at time ``t`` has target at ``t + horizon``. If that
    target timestamp is beyond ``train_end_boundary``, the row is
    removed so that the temporal boundary between train and
    validation remains clean.
    """
    cutoff = train_end_boundary - horizon
    return train_df[train_df["timestamp"] <= cutoff]

def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop non-feature columns and split into X and y"""
    y = df[TARGET_COLUMN]
    X = df.drop(columns=list(NON_FEATURE_COLUMNS))
    return X, y

def _drop_unusable_rows(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Remove rows where any feature is NaN or the target is NaN.

    X and y are assumed to be aligned on their index.
    """
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]

def _ensure_non_empty(name: str, X: pd.DataFrame, y: pd.Series) -> None:
    """Raise if either X or y of a given split is empty"""
    if len(X) == 0 or len(y) == 0:
        raise ValueError(
            f"{name} split is empty after dropping unusable rows"
            f"(X rows={len(X)}, y rows={len(y)})"
        )

def _check_chronological_order(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Verify that train < val < test along the timestamp axis"""
    if len(train_df) and len(val_df):
        if train_df["timestamp"].max() >= val_df["timestamp"].min():
            raise ValueError(
                "train max timestamp must be < val min timestamp"
            )
    if len(val_df) and len(test_df):
        if val_df["timestamp"].max() >= test_df["timestamp"].min():
            raise ValueError(
                "val max timestamp must be < test min timestamp"
            )
    
# Public API ----------------------------------------------------------------------------
def split_data(df: pd.DataFrame, train_ratio: float=0.70, val_ratio: float=0.15) -> SplitResult:
    """Chronologically split a feature-engineered DataFrame.

    Parameters
    ----------
    df:
        Output of ``features.build_features()``.
    train_ratio:
        Fraction of rows for the training split. Must be in (0, 1).
    val_ratio:
        Fraction of rows for the validation split. Must be in (0, 1).
        ``train_ratio + val_ratio`` must be < 1.

    Returns
    -------
    SplitResult
        Named tuple with X_train / y_train / X_val / y_val /
        X_test / y_test.

    Raises
    ------
    ValueError
        If ratios are invalid, or any split is empty after dropping
        unusable rows.
    """
    
    logger.info(f"splitting started: shape={df.shape}")
    
    test_ratio = _validate_ratios(train_ratio, val_ratio)
    logger.info(
        f"ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}"
    )
    
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    _check_chronological_order(train_df, val_df, test_df)
    
    if len(train_df):
        train_end_boundary = train_df["timestamp"].iloc[-1]
        n_before = len(train_df)
        train_df = _apply_embargo(
            train_df, train_end_boundary, TARGET_HORIZON
        )
        n_embargoed = n_before - len(train_df)
        if n_embargoed:
            logger.info(f"embargo removed {n_embargoed} row(s) from train")
    
    X_train, y_train = _split_xy(train_df)
    X_val, y_val = _split_xy(val_df)
    X_test, y_test = _split_xy(test_df)
    
    n_train_before, n_val_before, n_test_before = (
        len(X_train), len(X_val), len(X_test)
    )
    X_train, y_train = _drop_unusable_rows(X_train, y_train)
    X_val, y_val = _drop_unusable_rows(X_val, y_val)
    X_test, y_test = _drop_unusable_rows(X_test, y_test)
    
    removed_train = n_train_before - len(X_train)
    removed_val = n_val_before - len(X_val)
    removed_test = n_test_before - len(X_test)
    if removed_train or removed_val or removed_test:
        logger.info(
            f"removed rows with missing values:"
            f"train {removed_train}, val {removed_val}, test {removed_test}"
        )
    _ensure_non_empty("train", X_train, y_train)
    _ensure_non_empty("val", X_val, y_val)
    _ensure_non_empty("test", X_test, y_test)
    
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    logger.info(
        f"splitting completed: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    
    return SplitResult(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )