import numpy as np
import pandas as pd
import pytest

from trading_strategies.features.factor_model import (
    cross_sectional_zscore,
    expanding_covariance,
    expanding_mu_f,
    fama_macbeth_daily,
    fit_cross_sectional_ols,
)


def test_fit_cross_sectional_ols_recovers_exact_linear_relationship() -> None:
    features = np.array([[1.0, 5.0], [2.0, 1.0], [3.0, 4.0], [4.0, 2.0], [5.0, 3.0]])
    target = 2 + 3 * features[:, 0] - 1 * features[:, 1]

    coefficients = fit_cross_sectional_ols(features, target, n_params=3)

    assert coefficients == pytest.approx([2.0, 3.0, -1.0])


def test_fit_cross_sectional_ols_is_nan_when_underdetermined() -> None:
    features = np.array([[1.0, 5.0], [2.0, 1.0]])
    target = np.array([10.0, 20.0])

    coefficients = fit_cross_sectional_ols(features, target, n_params=3)

    assert np.isnan(coefficients).all()


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


def test_fama_macbeth_daily_recovers_exact_linear_relationship() -> None:
    # y = 2 + 3*f1 - 1*f2, exactly, no noise: OLS must recover these exactly.
    tickers = ["A", "B", "C", "D", "E"]
    f1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    f2 = [5.0, 1.0, 4.0, 2.0, 3.0]
    y = [2 + 3 * a - 1 * b for a, b in zip(f1, f2, strict=True)]

    panel = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 5,
            "ticker": tickers,
            "f1": f1,
            "f2": f2,
            "target": y,
        }
    )

    result = fama_macbeth_daily(panel, feature_columns=["f1", "f2"], target_column="target")

    row = result.loc[result["date"] == "2020-01-01"].iloc[0]
    assert row["intercept"] == pytest.approx(2.0)
    assert row["f1"] == pytest.approx(3.0)
    assert row["f2"] == pytest.approx(-1.0)


def test_fama_macbeth_daily_is_nan_when_fewer_observations_than_parameters() -> None:
    # 2 features + intercept = 3 params, only 2 tickers with data: underdetermined.
    panel = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "ticker": ["A", "B"],
            "f1": [1.0, 2.0],
            "f2": [5.0, 1.0],
            "target": [10.0, 20.0],
        }
    )

    result = fama_macbeth_daily(panel, feature_columns=["f1", "f2"], target_column="target")

    row = result.loc[result["date"] == "2020-01-01"].iloc[0]
    assert row["intercept"] != row["intercept"]
    assert row["f1"] != row["f1"]
    assert row["f2"] != row["f2"]


def test_expanding_mu_f_is_nan_on_the_first_date() -> None:
    lambda_daily = pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02"], "intercept": [1.0, 3.0], "f1": [10.0, 20.0]}
    )

    result = expanding_mu_f(lambda_daily, feature_columns=["intercept", "f1"])

    first = result.iloc[0]
    assert first["intercept"] != first["intercept"]
    assert first["f1"] != first["f1"]


def test_expanding_mu_f_excludes_the_current_dates_own_lambda() -> None:
    lambda_daily = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "intercept": [1.0, 3.0, 5.0],
            "f1": [10.0, 20.0, 30.0],
        }
    )

    result = expanding_mu_f(lambda_daily, feature_columns=["intercept", "f1"])

    # Row for 2020-01-02 must use only 2020-01-01's lambda, not its own.
    day2 = result[result["date"] == "2020-01-02"].iloc[0]
    assert day2["intercept"] == pytest.approx(1.0)
    assert day2["f1"] == pytest.approx(10.0)

    # Row for 2020-01-03 must average 01-01 and 01-02, not include 01-03.
    day3 = result[result["date"] == "2020-01-03"].iloc[0]
    assert day3["intercept"] == pytest.approx(2.0)
    assert day3["f1"] == pytest.approx(15.0)


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
