import pandas as pd

from trading_strategies.features.covariance import expanding_covariance


def test_expanding_covariance_uses_only_returns_strictly_before_the_date() -> None:
    # 4 dates, 2 tickers. Sigma "usable at" date_3 must use only date_1, date_2
    # returns (date_0 has no return, it's the price series' first observation).
    wide_returns = pd.DataFrame(
        {"A": [float("nan"), 0.01, -0.02, 0.03], "B": [float("nan"), 0.02, 0.01, -0.01]},
        index=pd.bdate_range("2020-01-01", periods=4),
    )

    result = expanding_covariance(wide_returns, min_periods=2)

    usable_at_date3 = result[wide_returns.index[3]]
    expected = wide_returns.iloc[1:3].cov()
    pd.testing.assert_frame_equal(usable_at_date3, expected)


def test_expanding_covariance_is_missing_before_min_periods_is_reached() -> None:
    wide_returns = pd.DataFrame(
        {"A": [float("nan"), 0.01, -0.02], "B": [float("nan"), 0.02, 0.01]},
        index=pd.bdate_range("2020-01-01", periods=3),
    )

    result = expanding_covariance(wide_returns, min_periods=2)

    # Only 1 return exists before date_2 (date_1's), short of min_periods=2.
    assert wide_returns.index[2] not in result
