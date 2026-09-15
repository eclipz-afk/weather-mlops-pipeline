import numpy as np
import pandas as pd

PERSISTANCE_FEATURE = "temperature_2m"
PREDICTION_COLUMN = "prediction"

def persistence_baseline(
    X: pd.DataFrame,
    *,
    source_column: str = PERSISTANCE_FEATURE,
) -> pd.Series:
    """Predict the next-hour temperature as the current temperature.

    Parameters
    ----------
    X:
        Feature matrix containing the current weather observation.
    source_column:
        Column holding the temperature available at prediction time.

    Returns
    -------
    pandas.Series
        Prediction series aligned with ``X.index`` and named ``"prediction"``.

    Raises
    ------
    ValueError
        If X is empty, source_column is absent, contains missing values,
        or contains non-finite values.
    """
    
    if X.empty:
        raise ValueError(
            "X is empty"
        )
    if source_column not in X.columns:
        raise ValueError(
            f"X must contain baseline source column {source_column}"
        )
    
    values = pd.to_numeric(X[source_column], errors="raise")
    
    if values.isna().any():
        raise ValueError(
            f"{source_column!r} contains not-finite values"
        )
    
    return pd.Series(
        values.to_numpy(dtype=float),
        index=X.index,
        name=PREDICTION_COLUMN,
    )
