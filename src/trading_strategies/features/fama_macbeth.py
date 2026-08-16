from collections.abc import Sequence

import numpy as np
import pandas as pd


def _fit_one_day(
    group: pd.DataFrame, feature_columns: Sequence[str], target_column: str
) -> list[float]:
    valid = group.dropna(subset=[*feature_columns, target_column])
    n_params = len(feature_columns) + 1  # + intercept

    if len(valid) <= n_params:
        return [float("nan")] * n_params

    x = np.column_stack([np.ones(len(valid)), valid[list(feature_columns)].to_numpy()])
    y = valid[target_column].to_numpy()
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return list(coefficients)


def fama_macbeth_daily(
    panel: pd.DataFrame, feature_columns: Sequence[str], target_column: str
) -> pd.DataFrame:
    """One cross-sectional OLS per date: target ~ intercept + feature_columns.

    Each date's fit uses only that date's tickers with complete data
    (features and target both present) — a thin, per-day cross-section, not
    pooled across time.
    """
    rows = []
    for date, group in panel.groupby("date"):
        coefficients = _fit_one_day(group, feature_columns, target_column)
        row = dict(zip(["intercept", *feature_columns], coefficients, strict=True))
        rows.append({"date": date, **row})
    return pd.DataFrame(rows)
