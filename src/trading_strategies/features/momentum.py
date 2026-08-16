import math

import pandas as pd

from trading_strategies.features.returns import daily_returns, ewm_vol

VOL_SCALED_RETURN_DELTAS = (1, 21, 63, 126, 252)


def vol_scaled_return(prices: pd.Series, sigma: pd.Series, delta: int) -> pd.Series:
    """Vol-scaled return over a delta-day window: r_{t-delta:t} / (sigma_t * sqrt(delta)).

    ``sigma`` is the daily-return vol (e.g. ``ewm_vol``), passed in rather than
    recomputed so the same estimate can be shared across every delta.
    """
    window_return = prices / prices.shift(delta) - 1
    return window_return / (sigma * math.sqrt(delta))


def vol_scaled_return_features(
    prices: pd.Series, deltas: tuple[int, ...] = VOL_SCALED_RETURN_DELTAS, vol_span: int = 60
) -> pd.DataFrame:
    """The 5 vol-scaled-return features, one column per ``delta``, sharing a
    single vol estimate across all of them."""
    sigma = ewm_vol(daily_returns(prices), span=vol_span)
    return pd.DataFrame(
        {f"vol_scaled_return_{delta}": vol_scaled_return(prices, sigma, delta) for delta in deltas}
    )
