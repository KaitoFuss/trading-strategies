import pandas as pd


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple period-over-period return. First entry is NaN (no prior price)."""
    return prices.pct_change()


def ewm_vol(returns: pd.Series, span: int = 60) -> pd.Series:
    """Exponentially-weighted standard deviation of returns, span 60 per spec."""
    return returns.ewm(span=span).std()
