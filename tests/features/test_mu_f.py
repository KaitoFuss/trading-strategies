import pandas as pd
import pytest

from trading_strategies.features.mu_f import expanding_mu_f


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
