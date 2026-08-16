import pandas as pd
import pytest

from trading_strategies.features.standardize import cross_sectional_zscore


def test_cross_sectional_zscore_standardizes_within_each_date() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "ticker": ["A", "B", "C", "A", "B"],
            "feature_1": [1.0, 2.0, 3.0, 10.0, 20.0],
        }
    )

    result = cross_sectional_zscore(panel, columns=["feature_1"])

    day1 = result[result["date"] == "2020-01-01"]["feature_1"]
    assert day1.mean() == pytest.approx(0.0, abs=1e-9)
    assert day1.std(ddof=0) == pytest.approx(1.0)

    day2 = result[result["date"] == "2020-01-02"]["feature_1"]
    assert day2.mean() == pytest.approx(0.0, abs=1e-9)


def test_cross_sectional_zscore_excludes_nan_tickers_from_the_days_stats() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "ticker": ["A", "B", "C"],
            "feature_1": [1.0, 2.0, float("nan")],
        }
    )

    result = cross_sectional_zscore(panel, columns=["feature_1"])

    # A and B alone: mean 1.5, std 0.5 -> z-scores -1.0 and 1.0
    assert result.loc[result["ticker"] == "A", "feature_1"].iloc[0] == pytest.approx(-1.0)
    assert result.loc[result["ticker"] == "B", "feature_1"].iloc[0] == pytest.approx(1.0)
    nan_value = result.loc[result["ticker"] == "C", "feature_1"].iloc[0]
    assert nan_value != nan_value
