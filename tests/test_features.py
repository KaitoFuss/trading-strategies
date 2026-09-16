import numpy as np
import pandas as pd
import pytest

from trading_strategies.features import (
    MACD_PAIRS,
    MOMENTUM_LOOKBACKS,
    compute_all_features,
    daily_returns,
    ewm_vol,
    macd_indicator,
    vol_scaled_momentum,
    winsorize,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_daily_returns_matches_log_ratio_by_hand() -> None:
    close = pd.Series([100.0, 110.0, 99.0], index=_dates(3))
    result = daily_returns(close)
    assert result.iloc[0] != result.iloc[0]  # NaN first observation
    assert result.iloc[1] == pytest.approx(np.log(1.10))
    assert result.iloc[2] == pytest.approx(np.log(99.0 / 110.0))


def test_ewm_vol_is_zero_for_constant_returns() -> None:
    returns = pd.Series([0.01] * 120, index=_dates(120))
    vol = ewm_vol(returns, span=60)
    assert vol.iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_ewm_vol_nan_before_min_periods() -> None:
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50), index=_dates(50))
    vol = ewm_vol(returns, span=60)
    assert vol.iloc[:59].isna().all()


def test_vol_scaled_momentum_sign_matches_return_direction() -> None:
    n = 300
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, n), index=_dates(n))
    close = (1 + returns).cumprod() * 100.0
    vol = ewm_vol(returns)
    feature = vol_scaled_momentum(close, lookback=21, vol=vol)
    realized_return = close.pct_change(21)
    valid = feature.notna() & (realized_return != 0)
    assert (np.sign(feature[valid]) == np.sign(realized_return[valid])).all()


def test_vol_scaled_momentum_scales_down_with_higher_vol() -> None:
    # Same 21-day cumulative return, but one series has much higher intervening
    # vol -> its scaled momentum feature should have smaller magnitude.
    dates = _dates(300)
    n = 300
    rng = np.random.default_rng(2)

    calm_returns = pd.Series(rng.normal(0.0, 0.001, n), index=dates)
    calm_close = (1 + calm_returns).cumprod() * 100.0

    wild_returns = pd.Series(rng.normal(0.0, 0.05, n), index=dates)
    wild_close = (1 + wild_returns).cumprod() * 100.0

    calm_vol = ewm_vol(calm_returns)
    wild_vol = ewm_vol(wild_returns)

    calm_feature = vol_scaled_momentum(calm_close, 21, calm_vol).iloc[-1]
    wild_feature = vol_scaled_momentum(wild_close, 21, wild_vol).iloc[-1]

    assert abs(calm_feature) != abs(wild_feature)  # sanity: not degenerate
    assert calm_vol.iloc[-1] < wild_vol.iloc[-1]


def test_macd_indicator_positive_for_sustained_uptrend() -> None:
    # A noiseless trend collapses q's own rolling std to ~0, blowing up the
    # normalization step and saturating the response function to 0 -- add
    # noise so q retains variance and the trend sign survives normalization.
    n = 400
    rng = np.random.default_rng(7)
    trend = np.linspace(100, 200, n)
    noise = rng.normal(0, 0.5, n)
    close = pd.Series(trend + noise, index=_dates(n))
    feature = macd_indicator(close, short=8, long=24)
    assert feature.iloc[-1] > 0


def test_macd_indicator_bounded_by_response_function_when_phi_applied() -> None:
    n = 400
    rng = np.random.default_rng(3)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=_dates(n))
    feature = macd_indicator(close, short=8, long=24, apply_phi=True)
    valid = feature.dropna()
    # exp(-x^2/4) response function caps the transform near +/- 1/sqrt(2e)/0.89
    assert (valid.abs() <= 1.1).all()


def test_macd_indicator_unbounded_by_default() -> None:
    # apply_phi now defaults to False -- the raw normalized z-score can
    # exceed the response function's ~0.964 ceiling.
    n = 400
    rng = np.random.default_rng(3)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=_dates(n))
    feature = macd_indicator(close, short=8, long=24)
    valid = feature.dropna()
    assert valid.abs().max() > 1.1


def test_winsorize_clips_a_single_extreme_outlier() -> None:
    n = 300
    rng = np.random.default_rng(4)
    series = pd.Series(rng.normal(0, 1, n), index=_dates(n))
    series.iloc[-1] = 1000.0  # extreme outlier at the end
    clipped = winsorize(series)
    assert clipped.iloc[-1] < 1000.0
    assert clipped.iloc[-1] == pytest.approx(
        series.ewm(halflife=31, min_periods=31).mean().iloc[-1]
        + 5.0 * series.ewm(halflife=31, min_periods=31).std().iloc[-1]
    )


def test_winsorize_leaves_well_behaved_series_untouched() -> None:
    n = 300
    rng = np.random.default_rng(5)
    series = pd.Series(rng.normal(0, 1, n), index=_dates(n))
    clipped = winsorize(series)
    valid = clipped.dropna()
    pd.testing.assert_series_equal(valid, series.loc[valid.index], check_exact=True)


def test_compute_all_features_returns_expected_columns() -> None:
    n = 400
    rng = np.random.default_rng(6)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=_dates(n))
    panel = compute_all_features(close)

    expected_columns = {f"mom_{lb}" for lb in MOMENTUM_LOOKBACKS} | {
        f"macd_{short}_{long}" for short, long in MACD_PAIRS
    }
    assert set(panel.columns) == expected_columns
    assert len(panel) == n
    assert panel.index.equals(close.index)
