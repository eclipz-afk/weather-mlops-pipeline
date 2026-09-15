import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger("weather_forecasting.model.train")

# Public API ---------------------------------------------------
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 143,
    n_jobs: int = 1,
) -> RandomForestRegressor:
    """Fit a RandomForestRegressor on the training split.

    Parameters
    ----------
    X_train:
        Training features, no NaN, at least one row and one column.
    y_train:
        Training target, no NaN, same length as ``X_train``.
    n_estimators, max_depth, min_samples_split, min_samples_leaf:
        RandomForestRegressor hyperparameters.
    random_state:
        Seed for reproducibility.
    n_jobs:
        Number of parallel jobs (-1 = all cores).

    Returns
    -------
    RandomForestRegressor
        Fitted model.

    Raises
    ------
    ValueError
        If X_train/y_train are empty, have different lengths, contain
        NaN, or X_train has no columns.
    """
    
    _check_training_input(X_train, y_train)
    
    logger.info(
        f"training started: X_train shape={X_train.shape}, n_estimators={n_estimators}"
    )
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    
    model.fit(X_train, y_train)
    
    logger.info(
        f"training completed: n_estimators={n_estimators} max_depth={max_depth}"
    )
    return model

def save_model(model, path: str | Path) -> Path:
    """Persist a fitted model with joblib.

    Parameters
    ----------
    model:
        Fitted sklearn-compatible model.
    path:
        Destination file path. Parent directories are created if
        missing.

    Returns
    -------
    pathlib.Path
        The path the model was saved to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"model saved to: {path}")
    return path

# Input contract --------------------------------------------------

def _check_training_input(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Enforce the training-input contract"""
    if X_train.empty:
        raise ValueError("X_train is empty")
    if y_train.empty:
        raise ValueError("y_train is empty")
    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must have the same length: "
            f"got {len(X_train)} and {len(y_train)}"
        )
    if X_train.shape[1] == 0:
        raise ValueError("X_train has no feature columns")
    if X_train.isna().any().any():
        raise ValueError("X_train contains NaN values")
    if y_train.isna().any():
        raise ValueError("y_train contains NaN values")
