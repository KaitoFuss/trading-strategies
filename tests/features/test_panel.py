from pathlib import Path

import pandas as pd

from trading_strategies.features.macd import macd_features
from trading_strategies.features.momentum import vol_scaled_return_features
from trading_strategies.features.panel import FEATURE_COLUMNS, build_feature_panel


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
