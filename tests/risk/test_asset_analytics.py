from __future__ import annotations

import math

import numpy as np
import pytest

from trading_strategies.risk.asset_analytics import (
    BIAS,
    R_SQUARED,
    SPECIFIC_SHARE,
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
