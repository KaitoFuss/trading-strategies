import pandas as pd
import pytest

from trading_strategies.features.returns import daily_returns, ewm_vol


def test_daily_returns_computes_simple_return() -> None:
    prices = pd.Series([100.0, 110.0, 121.0])

    returns = daily_returns(prices)

    assert returns.iloc[0] != returns.iloc[0]  # NaN: no prior price to diff against
    assert returns.iloc[1] == pytest.approx(0.10)
    assert returns.iloc[2] == pytest.approx(0.10)


def test_ewm_vol_uses_span_60() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, 0.03, -0.01, 0.02, 0.005, -0.015])

    vol = ewm_vol(returns)

    expected = returns.ewm(span=60).std()
    pd.testing.assert_series_equal(vol, expected)
