from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


def cross_sectional_zscore(panel: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Standardize each column to zero mean, unit std within each date.

    Population std (ddof=0): the cross-section on a given date is the whole
    universe for that day, not a sample of a larger one.
    """
    result = panel.copy()

    def zscore(group: pd.DataFrame) -> pd.DataFrame:
        return (group - group.mean()) / group.std(ddof=0)

    result[columns] = panel.groupby("date")[list(columns)].transform(zscore)
    return result


def fit_cross_sectional_ols(
    features: npt.NDArray[np.float64], target: npt.NDArray[np.float64], n_params: int
) -> npt.NDArray[np.float64]:
    """intercept + OLS slope per feature column, or all-NaN if underdetermined.

    ``features`` is observations x factors (no intercept column — added
    here); ``n_params`` is factors + 1, passed in rather than derived so the
    caller's NaN-row-count decision and this function's shape agree by
    construction. Pure numpy: shared by the batch panel fit below and any
    live, bar-by-bar caller that never has a DataFrame to hand.
    """
    if len(features) <= n_params:
        return np.full(n_params, np.nan)
    x = np.column_stack([np.ones(len(features)), features])
    coefficients, _, _, _ = np.linalg.lstsq(x, target, rcond=None)
    result: npt.NDArray[np.float64] = coefficients
    return result


def fama_macbeth_daily(
    panel: pd.DataFrame, feature_columns: Sequence[str], target_column: str
) -> pd.DataFrame:
    """One cross-sectional OLS per date: target ~ intercept + feature_columns.

    Each date's fit uses only that date's tickers with complete data
    (features and target both present) — a thin, per-day cross-section, not
    pooled across time. NaN when a day has fewer observations than
    parameters, rather than a degenerate numpy.lstsq min-norm solution.
    """
    index = ["intercept", *feature_columns]
    n_params = len(index)

    rows = []
    for date, group in panel.groupby("date"):
        valid = group.dropna(subset=[*feature_columns, target_column])
        coefficients = fit_cross_sectional_ols(
            valid[list(feature_columns)].to_numpy(), valid[target_column].to_numpy(), n_params
        )
        rows.append({"date": date, **dict(zip(index, coefficients, strict=True))})
    return pd.DataFrame(rows)


def expanding_mu_f(lambda_daily: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Expanding mean of the daily Fama-MacBeth coefficients, shifted by one day.

    lambda_t is fit against the return realized between t and t+1, so it
    isn't actually known until t+1 — mu_f for date t must only use lambda
    for dates strictly before t, hence the shift(1) after expanding().mean().
    """
    result = lambda_daily.sort_values("date").copy()
    result[list(feature_columns)] = lambda_daily[list(feature_columns)].expanding().mean().shift(1)
    return result


def expanding_covariance(
    wide_returns: pd.DataFrame, min_periods: int
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Sample covariance of asset returns, keyed by the date it's usable at.

    Sigma "usable at" date t is computed from returns strictly before t —
    the same next-day shift mu_f uses, so a decision made at t never sees
    t's own return. Dates without min_periods prior returns are absent from
    the result rather than present with a degenerate covariance.
    """
    dates = wide_returns.index
    result: dict[pd.Timestamp, pd.DataFrame] = {}
    for i in range(1, len(dates)):
        history = wide_returns.iloc[:i]
        if history.dropna().shape[0] < min_periods:
            continue
        result[dates[i]] = history.cov()
    return result
