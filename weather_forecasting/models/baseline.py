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

    """
    
    return X[source_column].astype(float).rename(PREDICTION_COLUMN)
