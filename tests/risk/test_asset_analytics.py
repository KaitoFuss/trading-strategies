from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from trading_strategies.risk.asset_analytics import (
    ANNUAL,
    BIAS,
    R_SQUARED,
    SPECIFIC_SHARE,
    AssetRiskAccumulator,
    ModelFitAccumulator,
)
from trading_strategies.risk.model import RiskModelSnapshot

_B = np.array([[1.0], [0.5]])
_F = np.array([[1e-4]])
_D = np.array([1e-4, 4e-4])


def _snapshot() -> RiskModelSnapshot:
    zeros = np.zeros((2, 1))
    return RiskModelSnapshot(
        ("X", "Y"), ("Equity",), _B, _F, _D, np.zeros(2), zeros, zeros, None, True
    )


def _bars(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, n)
    returns = _B[:, 0][None, :] * f[:, None] + rng.normal(0, 0.02, (n, 2))
    return f, returns


def _expected(r: np.ndarray, f: np.ndarray, i: int) -> tuple[float, float]:
    total = float((_B @ _F * _B).sum(axis=1)[i] + _D[i])
    z = r / math.sqrt(total)
    eps = r - _B[i, 0] * f
    return float(np.std(z, ddof=1)), 1 - float(np.var(eps, ddof=1) / np.var(r, ddof=1))


def test_fit_matches_a_direct_computation() -> None:
    f, returns = _bars()
    acc = ModelFitAccumulator()
    for t in range(len(f)):
        acc.observe(_snapshot(), {"X": returns[t, 0], "Y": returns[t, 1]}, np.array([f[t]]))

    result = acc.result(["X", "Y"])

    bias, r_squared = _expected(returns[:, 1], f, 1)
    assert result.loc["Y", BIAS] == pytest.approx(bias)
    assert result.loc["Y", R_SQUARED] == pytest.approx(r_squared)
    assert result.loc["Y", SPECIFIC_SHARE] == pytest.approx(4e-4 / (0.25e-4 + 4e-4))


def test_missing_or_nan_returns_are_left_out() -> None:
    f, returns = _bars()
    acc = ModelFitAccumulator()
    kept: list[int] = []
    for t in range(len(f)):
        bar = {"X": returns[t, 0]}
        if t % 2 == 0:
            bar["Y"] = returns[t, 1]
            kept.append(t)
        elif t % 3 == 0:
            bar["Y"] = math.nan
        acc.observe(_snapshot(), bar, np.array([f[t]]))

    result = acc.result(["Y"])

    bias, r_squared = _expected(returns[kept, 1], f[kept], 1)
    assert result.loc["Y", BIAS] == pytest.approx(bias)
    assert result.loc["Y", R_SQUARED] == pytest.approx(r_squared)


def test_constant_return_gives_nan() -> None:
    f, returns = _bars()
    acc = ModelFitAccumulator()
    for t in range(len(f)):
        acc.observe(_snapshot(), {"X": 0.0, "Y": returns[t, 1]}, np.array([f[t]]))

    result = acc.result(["X"])

    assert result.loc["X", [BIAS, R_SQUARED]].isna().all()
    assert result.loc["X", SPECIFIC_SHARE] == pytest.approx(1e-4 / (1e-4 + 1e-4))


def test_result_follows_the_requested_assets() -> None:
    f, returns = _bars()
    acc = ModelFitAccumulator()
    for t in range(len(f)):
        acc.observe(_snapshot(), {"X": returns[t, 0], "Y": returns[t, 1]}, np.array([f[t]]))

    result = acc.result(["Y", "X", "Z"])

    assert list(result.index) == ["Y", "X", "Z"]
    assert list(result.columns) == [BIAS, R_SQUARED, SPECIFIC_SHARE]
    assert result.loc["Z"].isna().all()


_START = datetime(2024, 1, 1)


def _risk_snapshot(factors: tuple[str, ...] = ("Equity",), n: int = 3) -> RiskModelSnapshot:
    k = len(factors)
    rng = np.random.default_rng(k)
    B = rng.normal(0.8, 0.3, (n, k))
    F = np.diag(np.full(k, 1e-4))
    D = np.linspace(1e-4, 3e-4, n)
    tickers = ("X", "Y", "Z")[:n]
    zeros = np.zeros((n, k))
    return RiskModelSnapshot(tickers, factors, B, F, D, np.zeros(n), zeros, zeros, None, True)


def _feed(
    acc: AssetRiskAccumulator, snapshot: RiskModelSnapshot, n: int, seed: int = 0, start: int = 0
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    books = []
    for t in range(n):
        w = rng.normal(0, 0.3, len(snapshot.tickers))
        r = rng.normal(0, 0.01, len(snapshot.tickers))
        acc.observe(_START + timedelta(days=start + t), snapshot, w, r)
        books.append(w)
    return books


def test_predicted_parts_sum_to_mean_annualized_vol() -> None:
    snapshot = _risk_snapshot()
    acc = AssetRiskAccumulator()
    books = _feed(acc, snapshot, 50)
    zero = np.zeros(3)
    acc.observe(_START + timedelta(days=50), snapshot, zero, np.full(3, 0.01))
    sigmas = [math.sqrt(w @ snapshot.covariance() @ w) for w in [*books, zero]]

    risk = acc.result(["X", "Y", "Z"])

    total = risk.predicted_factor.sum() + risk.predicted_specific.sum()
    assert total == pytest.approx(np.mean(sigmas) * ANNUAL)


def test_realized_contributions_sum_to_realized_vol() -> None:
    acc = AssetRiskAccumulator()
    _feed(acc, _risk_snapshot(), 100)

    risk = acc.result(["X", "Y", "Z"])

    holdings = risk.holding_returns.sum(axis=1)
    assert risk.realized().sum() == pytest.approx(holdings.std() * ANNUAL)


def test_exposure_contributions_sum_to_mean_exposure() -> None:
    snapshot = _risk_snapshot(("Equity", "Rates"))
    acc = AssetRiskAccumulator()
    books = _feed(acc, snapshot, 30)

    risk = acc.result(["X", "Y", "Z"])

    mean_exposure = np.mean([snapshot.B.T @ w for w in books], axis=0)
    np.testing.assert_allclose(risk.exposure_contributions.sum().to_numpy(), mean_exposure)


def test_factor_that_joins_later_counts_as_zero_before() -> None:
    early, late = _risk_snapshot(("Equity",)), _risk_snapshot(("Equity", "Momentum"))
    acc = AssetRiskAccumulator()
    _feed(acc, early, 10)
    late_books = _feed(acc, late, 10, seed=1, start=10)

    risk = acc.result(["X", "Y", "Z"])

    assert list(risk.exposure_contributions.columns) == ["Equity", "Momentum"]
    expected = np.sum([w * late.B[:, 1] for w in late_books], axis=0) / 20
    np.testing.assert_allclose(risk.exposure_contributions["Momentum"].to_numpy(), expected)


def test_hedge_can_have_negative_realized_contribution() -> None:
    snapshot = _risk_snapshot(n=2)
    acc = AssetRiskAccumulator()
    rng = np.random.default_rng(3)
    for t in range(200):
        common = rng.normal(0, 0.01)
        r = np.array([common + rng.normal(0, 0.002), common + rng.normal(0, 0.002)])
        acc.observe(_START + timedelta(days=t), snapshot, np.array([1.0, -0.3]), r)

    risk = acc.result(["X", "Y"])

    assert risk.realized()["Y"] < 0
    assert risk.realized()["X"] > 0


def test_result_follows_the_requested_assets_with_zeros() -> None:
    acc = AssetRiskAccumulator()
    _feed(acc, _risk_snapshot(), 10)

    risk = acc.result(["Z", "W"])

    assert list(risk.predicted_factor.index) == ["Z", "W"]
    assert list(risk.holding_returns.columns) == ["Z", "W"]
    assert risk.predicted_factor["W"] == 0.0
    assert risk.realized()["W"] == 0.0


def test_empty_accumulator_gives_empty_risk() -> None:
    risk = AssetRiskAccumulator().result(["X"])

    assert risk.is_empty()
    assert risk.realized().empty or (risk.realized() == 0).all()
