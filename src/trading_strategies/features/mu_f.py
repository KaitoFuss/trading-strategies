from collections.abc import Sequence

import pandas as pd


def expanding_mu_f(lambda_daily: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Expanding mean of the daily Fama-MacBeth coefficients, shifted by one day.

    lambda_t is fit against the return realized between t and t+1, so it
    isn't actually known until t+1 — mu_f for date t must only use lambda
    for dates strictly before t, hence the shift(1) after expanding().mean().
    """
    result = lambda_daily.sort_values("date").copy()
    result[list(feature_columns)] = lambda_daily[list(feature_columns)].expanding().mean().shift(1)
    return result
