import pandas as pd

MACD_PAIRS = ((8, 24), (16, 48), (32, 96))


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
