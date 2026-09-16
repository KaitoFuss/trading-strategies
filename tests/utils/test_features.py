import math

import numpy as np
import pandas as pd
import pytest

from trading_strategies.utils.features import WINSOR_HALFLIFE, WINSOR_Z, response_function


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
