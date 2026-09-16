import math
from typing import cast

import numpy as np
import pandas as pd
import pytest

from trading_strategies.utils.features import (
    WINSOR_HALFLIFE,
    WINSOR_Z,
    StreamingMacd,
    response_function,
)


def test_response_function_matches_baz_formula() -> None:
    y = 1.5
    expected = y * math.exp(-(y**2) / 4) / 0.89
    assert response_function(y) == pytest.approx(expected)


def test_response_function_bounded_near_one() -> None:
    for y in (-10.0, -1.0, 0.0, 1.0, 10.0):
        assert abs(response_function(y)) <= 1.1


def test_winsorizer_passes_value_through_during_warmup() -> None:
    from trading_strategies.utils.features import _Winsorizer

    winsorizer = _Winsorizer()
    # WINSOR_HALFLIFE=31 min_periods -- well before that, clip is a no-op,
    # matching Series.clip(lower=NaN, upper=NaN) leaving the value unchanged.
    assert winsorizer.apply(1000.0) == 1000.0


def test_winsorizer_matches_pandas_clip_bit_for_bit() -> None:
    from trading_strategies.utils.features import _Winsorizer

    rng = np.random.default_rng(5)
    values = rng.normal(0, 1, 100)
    values[-1] = 1000.0  # extreme outlier at the end

    series = pd.Series(values)
    mean = series.ewm(halflife=WINSOR_HALFLIFE, min_periods=WINSOR_HALFLIFE).mean()
    std = series.ewm(halflife=WINSOR_HALFLIFE, min_periods=WINSOR_HALFLIFE).std()
    expected = series.clip(lower=mean - WINSOR_Z * std, upper=mean + WINSOR_Z * std)

    winsorizer = _Winsorizer()
    actual = [winsorizer.apply(v) for v in values]

    for a, e in zip(actual, expected, strict=True):
        assert a == pytest.approx(e)


def _batch_macd(close: pd.Series, short: int, long: int, apply_phi: bool) -> pd.Series:
    macd = close.ewm(span=short, min_periods=short).mean() - close.ewm(
        span=long, min_periods=long
    ).mean()
    q = macd / close.rolling(63, min_periods=63).std()
    normalized = q / q.rolling(252, min_periods=252).std()
    if apply_phi:
        return cast(pd.Series, normalized * np.exp(-(normalized**2) / 4) / 0.89)
    return normalized


def test_streaming_macd_matches_batch_formula_unwinsorized() -> None:
    rng = np.random.default_rng(6)
    n = 400
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    expected = _batch_macd(close, short=8, long=24, apply_phi=True)

    macd = StreamingMacd(short=8, long=24, apply_phi=True, winsorize=False)
    actual = [macd.update(c) for c in close]

    for a, e in zip(actual, expected, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e, abs=1e-6)


def test_streaming_macd_positive_for_sustained_uptrend() -> None:
    rng = np.random.default_rng(7)
    n = 400
    trend = np.linspace(100, 200, n)
    noise = rng.normal(0, 0.5, n)
    macd = StreamingMacd(short=8, long=24)
    last = None
    for v in trend + noise:
        last = macd.update(v)
    assert last is not None
    assert last > 0


def test_streaming_macd_wires_winsorizer_based_on_flag() -> None:
    from trading_strategies.utils.features import _Winsorizer

    unwinsorized = StreamingMacd(short=8, long=24, winsorize=False)
    winsorized = StreamingMacd(short=8, long=24, winsorize=True)

    assert unwinsorized._winsorizer is None
    assert isinstance(winsorized._winsorizer, _Winsorizer)
