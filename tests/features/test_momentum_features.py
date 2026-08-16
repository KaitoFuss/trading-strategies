from pathlib import Path

import pandas as pd
import pytest

from trading_strategies.features.momentum_features import (
    FEATURE_COLUMNS,
    MACD_PAIRS,
    VOL_SCALED_RETURN_DELTAS,
    build_feature_panel,
    ema,
    macd,
    macd_features,
    normalized_macd,
    vol_scaled_return,
    vol_scaled_return_features,
    winsorize,
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


def test_ema_uses_alpha_one_over_j() -> None:
    prices = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0])

    result = ema(prices, j=8)

    expected = prices.ewm(alpha=1 / 8).mean()
    pd.testing.assert_series_equal(result, expected)


def test_macd_is_difference_of_two_emas() -> None:
    prices = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 107.0, 110.0])

    result = macd(prices, short=8, long=24)

    expected = ema(prices, j=8) - ema(prices, j=24)
    pd.testing.assert_series_equal(result, expected)


def test_normalized_macd_matches_two_step_normalization_formula() -> None:
    # Deterministic but non-trivial: enough history for both rolling windows.
    prices = pd.Series(100.0 + 10.0 * pd.Series(range(400)).apply(lambda i: (i % 17) - 8))

    result = normalized_macd(prices, short=8, long=24)

    raw = macd(prices, short=8, long=24)
    q = raw / prices.rolling(63).std()
    expected = (q / q.rolling(252).std()).clip(-5, 5)
    pd.testing.assert_series_equal(result, expected)


def test_winsorize_clips_to_plus_minus_limit() -> None:
    series = pd.Series([10.0, -10.0, 2.0, float("nan")])

    result = winsorize(series, limit=5.0)

    assert result.tolist()[:3] == [5.0, -5.0, 2.0]
    assert result.iloc[3] != result.iloc[3]  # NaN passes through


def test_macd_features_has_one_column_per_spec_pair() -> None:
    prices = pd.Series(100.0 + 10.0 * pd.Series(range(400)).apply(lambda i: (i % 17) - 8))

    features = macd_features(prices)

    assert list(features.columns) == [f"macd_{short}_{long}" for short, long in MACD_PAIRS]
    pd.testing.assert_series_equal(
        features["macd_8_24"], normalized_macd(prices, short=8, long=24), check_names=False
    )


def _write_raw_parquet(path: Path, prices_by_ticker: dict[str, list[float]]) -> None:
    rows = []
    for ticker, prices in prices_by_ticker.items():
        dates = pd.bdate_range("2020-01-01", periods=len(prices))
        for date, close in zip(dates, prices, strict=True):
            rows.append({"date": date, "ticker": ticker, "close": close})
    pd.DataFrame(rows).to_parquet(path)


def test_build_feature_panel_has_expected_columns(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _write_raw_parquet(raw_path, {"AAA": [100.0, 101.0, 102.0], "BBB": [50.0, 51.0, 49.0]})

    panel = build_feature_panel(raw_path)

    assert list(panel.columns) == ["date", "ticker", *FEATURE_COLUMNS]
    assert set(panel["ticker"]) == {"AAA", "BBB"}
    assert len(panel) == 6  # 3 dates x 2 tickers


def test_build_feature_panel_matches_per_ticker_computation(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    prices = [100.0 + (i % 17) - 8 for i in range(400)]
    _write_raw_parquet(raw_path, {"AAA": prices})

    panel = build_feature_panel(raw_path)

    prices_series = pd.Series(prices)
    expected = pd.concat(
        [vol_scaled_return_features(prices_series), macd_features(prices_series)], axis=1
    )
    aaa = panel[panel["ticker"] == "AAA"].reset_index(drop=True)
    for column in FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            aaa[column], expected[column], check_names=False, check_index=False
        )
