import math
from pathlib import Path

import pandas as pd

VOL_SCALED_RETURN_DELTAS = (1, 21, 63, 126, 252)
MACD_PAIRS = ((8, 24), (16, 48), (32, 96))

FEATURE_COLUMNS = [
    "vol_scaled_return_1",
    "vol_scaled_return_21",
    "vol_scaled_return_63",
    "vol_scaled_return_126",
    "vol_scaled_return_252",
    "macd_8_24",
    "macd_16_48",
    "macd_32_96",
]


def vol_scaled_return(prices: pd.Series, sigma: pd.Series, delta: int) -> pd.Series:
    """Vol-scaled return over a delta-day window: r_{t-delta:t} / (sigma_t * sqrt(delta)).

    ``sigma`` is the daily-return vol, passed in rather than recomputed so
    the same estimate can be shared across every delta.
    """
    window_return = prices / prices.shift(delta) - 1
    return window_return / (sigma * math.sqrt(delta))


def vol_scaled_return_features(
    prices: pd.Series, deltas: tuple[int, ...] = VOL_SCALED_RETURN_DELTAS, vol_span: int = 60
) -> pd.DataFrame:
    """The 5 vol-scaled-return features, one column per ``delta``, sharing a
    single vol estimate across all of them."""
    sigma = prices.pct_change().ewm(span=vol_span).std()
    return pd.DataFrame(
        {f"vol_scaled_return_{delta}": vol_scaled_return(prices, sigma, delta) for delta in deltas}
    )


def ema(prices: pd.Series, j: int) -> pd.Series:
    """Price EWMA with alpha = 1/j, per the network-momentum spec."""
    return prices.ewm(alpha=1 / j).mean()


def macd(prices: pd.Series, short: int, long: int) -> pd.Series:
    """MACD(t, S, L) = m(t, S) - m(t, L)."""
    return ema(prices, j=short) - ema(prices, j=long)


def winsorize(series: pd.Series, limit: float) -> pd.Series:
    """Clip to +-limit. NaN passes through untouched (pandas clip default)."""
    return series.clip(-limit, limit)


def normalized_macd(prices: pd.Series, short: int, long: int) -> pd.Series:
    """Two-step normalized, winsorized MACD feature.

    q(t,S,L) = MACD(t,S,L) / std(p, 63d); y = q / std(q, 252d); y winsorized
    to +-5. Both normalizing stds are rolling windows over price/q themselves.
    """
    raw = macd(prices, short=short, long=long)
    q = raw / prices.rolling(63).std()
    normalized = q / q.rolling(252).std()
    return winsorize(normalized, limit=5.0)


def macd_features(
    prices: pd.Series, pairs: tuple[tuple[int, int], ...] = MACD_PAIRS
) -> pd.DataFrame:
    """The 3 normalized-MACD features, one column per (S, L) pair."""
    return pd.DataFrame(
        {
            f"macd_{short}_{long}": normalized_macd(prices, short=short, long=long)
            for short, long in pairs
        }
    )


def build_feature_panel(raw_parquet: Path) -> pd.DataFrame:
    """The 8-feature V_t panel for every ticker in ``raw_parquet``.

    One row per (date, ticker), columns = FEATURE_COLUMNS. Rows before a
    ticker's longest warm-up (252 + 60 days for the slowest vol-scaled
    return, on top of the MACD windows) are NaN, not dropped here — the
    caller decides how to handle warm-up (drop, or let the regression step
    ignore incomplete rows).
    """
    raw = pd.read_parquet(raw_parquet)
    panels = []
    for ticker, group in raw.sort_values("date").groupby("ticker"):
        group = group.set_index("date")
        prices = group["close"]
        features = pd.concat([vol_scaled_return_features(prices), macd_features(prices)], axis=1)
        features["ticker"] = ticker
        panels.append(features)
    panel = pd.concat(panels).reset_index(names="date")
    return panel[["date", "ticker", *FEATURE_COLUMNS]]
