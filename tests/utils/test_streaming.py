import numpy as np
import pandas as pd
import pytest

from trading_strategies.utils.streaming import EwmMean, EwmMoments, RollingLag, RollingStd


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


def test_ewm_moments_matches_pandas_mean_and_std_span() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(0, 1, 60)
    span, min_periods = 8, 8
    expected_mean = pd.Series(values).ewm(span=span, min_periods=min_periods).mean()
    expected_std = pd.Series(values).ewm(span=span, min_periods=min_periods).std()

    ewm = EwmMoments(span=span, min_periods=min_periods)
    means, stds = [], []
    for v in values:
        ewm.update(v)
        means.append(ewm.mean)
        stds.append(ewm.std)

    for a, e in zip(means, expected_mean, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e)
    for a, e in zip(stds, expected_std, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e)


def test_ewm_moments_matches_pandas_std_halflife() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(0, 1, 60)
    halflife, min_periods = 31, 31
    expected_std = pd.Series(values).ewm(halflife=halflife, min_periods=min_periods).std()

    ewm = EwmMoments(halflife=halflife, min_periods=min_periods)
    stds = []
    for v in values:
        ewm.update(v)
        stds.append(ewm.std)

    for a, e in zip(stds, expected_std, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e)


def test_ewm_moments_zero_for_constant_input() -> None:
    ewm = EwmMoments(span=10, min_periods=10)
    for _ in range(20):
        ewm.update(1.0)
    assert ewm.std == pytest.approx(0.0, abs=1e-12)


def test_rolling_std_matches_pandas_rolling_std() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(0, 1, 30)
    window = 6
    expected = pd.Series(values).rolling(window, min_periods=window).std()

    roller = RollingStd(window)
    actual = [roller.update(v) for v in values]

    for a, e in zip(actual, expected, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e)


def test_rolling_lag_returns_value_from_lookback_steps_ago() -> None:
    lag = RollingLag(lookback=3)
    results = [lag.update(v) for v in [10.0, 20.0, 30.0, 40.0, 50.0]]
    assert results == [None, None, None, 10.0, 20.0]
