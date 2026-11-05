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

    """
    
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
