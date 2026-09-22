from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_strategies.risk.covariance import EwmCovariance


def _frame(n: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, (n, 1))
    data = base + rng.normal(0, 0.01, (n, 3))
    return pd.DataFrame(data, columns=["A", "B", "C"])


def _feed(cov: EwmCovariance, frame: pd.DataFrame) -> None:
    for _, row in frame.iterrows():
        cov.update({str(k): float(v) for k, v in row.items()})


def test_matches_pandas_ewm_cov_on_complete_data() -> None:
    frame = _frame()
    cov = EwmCovariance(["A", "B", "C"], span=20, min_periods=2)
    _feed(cov, frame)

    expected = frame.ewm(span=20).cov().xs(frame.index[-1], level=0)
    np.testing.assert_allclose(cov.covariance(), expected.to_numpy(), rtol=1e-10)


def test_late_listed_column_matches_pandas_since_listing() -> None:
    frame = _frame()
    frame.loc[:39, "C"] = np.nan
    cov = EwmCovariance(["A", "B", "C"], span=20, min_periods=2)
    _feed(cov, frame)

    full = frame[["A", "B"]].ewm(span=20).cov().xs(frame.index[-1], level=0)
    since = frame.iloc[40:].ewm(span=20).cov().xs(frame.index[-1], level=0)
    result = cov.covariance()
    assert result[0, 1] == pytest.approx(full.loc["A", "B"], rel=1e-10)
    assert result[0, 2] == pytest.approx(since.loc["A", "C"], rel=1e-10)
    assert result[2, 2] == pytest.approx(since.loc["C", "C"], rel=1e-10)


def test_nan_until_min_periods_joint_observations() -> None:
    cov = EwmCovariance(["A", "B"], span=10, min_periods=5)
    for i in range(4):
        cov.update({"A": 0.01 * i, "B": -0.02 * i})
    assert np.isnan(cov.covariance()).all()

    cov.update({"A": 0.05, "B": 0.01})
    assert np.isfinite(cov.covariance()).all()


def test_unknown_and_missing_tickers_are_ignored() -> None:
    cov = EwmCovariance(["A", "B"], span=10, min_periods=2)
    cov.update({"A": 0.01, "ZZZ": 1.0})
    cov.update({"A": 0.02, "B": float("nan")})
    cov.update({"A": 0.03})

    result = cov.covariance()
    assert np.isfinite(result[0, 0])
    assert np.isnan(result[1, 1])
    assert np.isnan(result[0, 1])


def test_min_periods_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_periods"):
        EwmCovariance(["A"], span=10, min_periods=1)
