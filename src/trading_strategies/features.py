"""Momentum and MACD features from Poh, Lim, Zohren & Roberts (2022) / network-momentum.

Every function takes and returns a single ticker's ``close`` price series,
indexed by date and sorted ascending. Cross-sectional assembly across tickers
is left to the caller.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

MOMENTUM_LOOKBACKS: tuple[int, ...] = (1, 21, 63, 126, 252)
MACD_PAIRS: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96))
VOL_SPAN = 60
WINSOR_SPAN = 252
WINSOR_Z = 5.0


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns."""
    return close.pct_change()


def ewm_vol(returns: pd.Series, span: int = VOL_SPAN) -> pd.Series:
    """EWMA daily volatility of returns."""
    return returns.ewm(span=span, min_periods=span).std()


def vol_scaled_momentum(close: pd.Series, lookback: int, vol: pd.Series) -> pd.Series:
    """Cumulative return over ``lookback`` days, scaled by vol and sqrt(lookback)."""
    raw_return = close.pct_change(lookback)
    return cast(pd.Series, raw_return / (vol * np.sqrt(lookback)))


def macd_indicator(close: pd.Series, short: int, long: int) -> pd.Series:
    """Normalized MACD signal (Baz et al. 2015), squashed to roughly [-1, 1]."""
    macd = (
        close.ewm(span=short, min_periods=short).mean()
        - close.ewm(span=long, min_periods=long).mean()
    )
    q = macd / close.rolling(63, min_periods=63).std()
    normalized = q / q.rolling(252, min_periods=252).std()
    return cast(pd.Series, normalized * np.exp(-(normalized**2) / 4) / 0.89)


def winsorize(series: pd.Series, span: int = WINSOR_SPAN, z: float = WINSOR_Z) -> pd.Series:
    """Clip to +/- z EWM standard deviations around the EWM mean."""
    mean = series.ewm(span=span, min_periods=span).mean()
    std = series.ewm(span=span, min_periods=span).std()
    return series.clip(lower=mean - z * std, upper=mean + z * std)


def compute_all_features(close: pd.Series) -> pd.DataFrame:
    """The 8 momentum/MACD features for one ticker, winsorized, as named columns."""
    returns = daily_returns(close)
    vol = ewm_vol(returns)

    features: dict[str, pd.Series] = {
        f"mom_{lookback}": winsorize(vol_scaled_momentum(close, lookback, vol))
        for lookback in MOMENTUM_LOOKBACKS
    }
    features.update(
        {
            f"macd_{short}_{long}": winsorize(macd_indicator(close, short, long))
            for short, long in MACD_PAIRS
        }
    )
    return pd.DataFrame(features, index=close.index)
