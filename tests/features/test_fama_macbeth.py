import pandas as pd
import pytest

from trading_strategies.features.fama_macbeth import fama_macbeth_daily


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
