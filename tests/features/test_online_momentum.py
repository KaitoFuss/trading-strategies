import math

import pandas as pd
import pytest

from trading_strategies.features.momentum_features import macd_features, vol_scaled_return_features
from trading_strategies.features.online_momentum import (
    OnlineTickerFeatures,
    RecursiveEma,
    RecursiveEwVariance,
)


def test_recursive_ema_matches_pandas_adjust_false_exactly() -> None:
    # adjust=False is the exact recursive definition RecursiveEma implements,
    # so this must match bit-for-bit, not just after burn-in.
    prices = [100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 107.0, 110.0]
    ema = RecursiveEma(j=8)

    values = [ema.update(p) for p in prices]

    expected = pd.Series(prices).ewm(alpha=1 / 8, adjust=False).mean()
    assert values == pytest.approx(expected.tolist())


def test_recursive_ew_variance_is_none_on_first_point() -> None:
    var = RecursiveEwVariance(span=60)

    result = var.update(0.01)

    assert result is None


def test_recursive_ew_variance_converges_to_pandas_after_burn_in() -> None:
    # Not exact (different bias-correction/initialization than pandas'
    # adjust=True), but must converge closely well past the span's memory.
    returns = [0.01 * ((i % 7) - 3) for i in range(500)]
    var = RecursiveEwVariance(span=60)

    for r in returns:
        var.update(r)

    native_std = var.std
    pandas_std = pd.Series(returns).ewm(span=60).std().iloc[-1]

    assert native_std == pytest.approx(pandas_std, rel=0.05)


def test_online_ticker_features_matches_batch_pandas_after_warm_up() -> None:
    # Master equivalence check: the whole native rewrite against the
    # already-tested pandas reference, well past every window's warm-up.
    prices = [100.0 + 10.0 * math.sin(i / 15) + i * 0.01 for i in range(800)]

    online = OnlineTickerFeatures()
    last_result = None
    for price in prices:
        last_result = online.update(price)
    assert last_result is not None

    prices_series = pd.Series(prices)
    expected = pd.concat(
        [vol_scaled_return_features(prices_series), macd_features(prices_series)], axis=1
    ).iloc[-1]

    assert last_result == pytest.approx(expected.tolist(), rel=0.05)


def test_online_ticker_features_is_none_during_warm_up() -> None:
    online = OnlineTickerFeatures()

    result = online.update(100.0)

    assert result is None
