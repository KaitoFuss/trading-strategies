import pandas as pd


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
