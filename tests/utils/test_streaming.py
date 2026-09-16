import numpy as np
import pandas as pd
import pytest

from trading_strategies.utils.streaming import EwmMean


def test_ewm_mean_matches_pandas_span() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 40)
    span, min_periods = 5, 5
    expected = pd.Series(values).ewm(span=span, min_periods=min_periods).mean()

    ewm = EwmMean(span=span, min_periods=min_periods)
    actual = [ewm.update(v) for v in values]

    for a, e in zip(actual, expected, strict=True):
        if np.isnan(e):
            assert a is None
        else:
            assert a == pytest.approx(e)


def test_ewm_mean_matches_pandas_halflife() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, 40)
    halflife, min_periods = 10.0, 10
    expected = pd.Series(values).ewm(halflife=halflife, min_periods=min_periods).mean()

    ewm = EwmMean(halflife=halflife, min_periods=min_periods)
    actual = [ewm.update(v) for v in values]

    for a, e in zip(actual, expected, strict=True):
        if np.isnan(e):
            assert a is None
        else:
            assert a == pytest.approx(e)


def test_ewm_mean_rejects_both_span_and_halflife() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        EwmMean(span=5, halflife=5.0, min_periods=5)


def test_ewm_mean_rejects_neither_span_nor_halflife() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        EwmMean(min_periods=5)
