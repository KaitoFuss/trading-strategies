import pandas as pd

from trading_strategies.features.macd import (
    MACD_PAIRS,
    ema,
    macd,
    macd_features,
    normalized_macd,
    winsorize,
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
