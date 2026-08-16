import pandas as pd
import pytest

from trading_strategies.features.momentum import (
    VOL_SCALED_RETURN_DELTAS,
    vol_scaled_return,
    vol_scaled_return_features,
)


def test_vol_scaled_return_divides_window_return_by_vol_times_sqrt_delta() -> None:
    prices = pd.Series([100.0, 105.0, 100.0])
    sigma = pd.Series([float("nan"), 0.02, 0.02])

    feature = vol_scaled_return(prices, sigma, delta=1)

    assert feature.iloc[1] == pytest.approx(0.05 / (0.02 * 1**0.5))
    assert feature.iloc[2] == pytest.approx((100.0 / 105.0 - 1) / (0.02 * 1**0.5))


def test_vol_scaled_return_scales_by_sqrt_delta_for_multi_day_windows() -> None:
    prices = pd.Series([100.0, 101.0, 102.0, 110.0])
    sigma = pd.Series([float("nan"), float("nan"), float("nan"), 0.03])

    feature = vol_scaled_return(prices, sigma, delta=3)

    window_return = 110.0 / 100.0 - 1
    assert feature.iloc[3] == pytest.approx(window_return / (0.03 * 3**0.5))


def test_vol_scaled_return_is_nan_before_delta_days_of_history() -> None:
    prices = pd.Series([100.0, 101.0, 102.0])
    sigma = pd.Series([0.02, 0.02, 0.02])

    feature = vol_scaled_return(prices, sigma, delta=3)

    assert feature.isna().all()


def test_vol_scaled_return_features_has_one_column_per_spec_delta() -> None:
    prices = pd.Series(range(300, 600), dtype=float)  # long enough to fill every delta

    features = vol_scaled_return_features(prices)

    assert list(features.columns) == [f"vol_scaled_return_{d}" for d in VOL_SCALED_RETURN_DELTAS]
    assert features["vol_scaled_return_1"].iloc[-1] == pytest.approx(
        vol_scaled_return(
            prices,
            prices.pct_change().ewm(span=60).std(),
            delta=1,
        ).iloc[-1]
    )
