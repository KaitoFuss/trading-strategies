from collections.abc import Callable

import numpy as np
import pytest

from trading_strategies.factor_risk import FactorRiskModel, _EwCovariance


def test_ew_covariance_is_none_before_a_second_observation() -> None:
    cov = _EwCovariance(halflife=10.0)
    cov.update(np.array([1.0, 2.0]))
    assert cov.value is None


def test_ew_covariance_follows_the_incremental_recurrence() -> None:
    # halflife=1 => alpha=0.5. Feed x1=[2,0], x2=[0,0]. By the West/Finch form:
    #   diff = x2 - mean = [-2, 0]; mean -> [1, 0]
    #   cov  = (1-a)*(0 + a*outer(diff,diff)) = 0.5*(0.5*[[4,0],[0,0]]) = [[1,0],[0,0]]
    cov = _EwCovariance(halflife=1.0)
    cov.update(np.array([2.0, 0.0]))
    cov.update(np.array([0.0, 0.0]))
    assert cov.value == pytest.approx(np.array([[1.0, 0.0], [0.0, 0.0]]))


def _feed(
    model: FactorRiskModel,
    loadings: dict[str, list[float]],
    factors: Callable[[int], np.ndarray],
    returns_fn: Callable[[int, np.ndarray], dict[str, float]],
    n: int,
) -> None:
    """Feed ``n`` bars: constant ``loadings`` (also used as prev), factor
    returns ``factors(t)``, asset returns ``returns_fn(t, factor_vector)``."""
    prev = None
    for t in range(n):
        f = np.asarray(factors(t), dtype=float)
        returns = returns_fn(t, f)
        model.update(loadings=loadings, prev_loadings=prev, factor_returns=f, returns=returns)
        prev = loadings


def test_covariance_is_positive_definite_when_assets_exceed_factors() -> None:
    # p=4 assets, k=2 factors: B Sigma_f B' has rank <= 2 and is singular on
    # its own. The strictly positive idiosyncratic diagonal D must lift every
    # eigenvalue above zero -- this is what makes shrinkage unnecessary.
    rng = np.random.default_rng(0)
    loadings = {"A": [1.0, 0.0], "B": [0.0, 1.0], "C": [1.0, 1.0], "D": [1.0, -1.0]}
    model = FactorRiskModel(halflife=50.0)
    _feed(
        model,
        loadings,
        factors=lambda t: rng.standard_normal(2) * 0.01,
        returns_fn=lambda t, f: {
            name: float(np.dot(b, f)) + rng.standard_normal() * 0.005
            for name, b in loadings.items()
        },
        n=300,
    )

    cov = model.covariance(["A", "B", "C", "D"])
    systematic = (
        np.array([loadings[t] for t in "ABCD"])
        @ model.factor_covariance()
        @ np.array([loadings[t] for t in "ABCD"]).T
    )

    assert np.linalg.eigvalsh(systematic).min() < 1e-9  # rank-deficient without D
    assert np.linalg.eigvalsh(cov).min() > 1e-9  # positive-definite with D


def test_covariance_assembles_B_sigma_f_Bt_plus_D_in_ticker_order() -> None:
    rng = np.random.default_rng(1)
    loadings = {"A": [2.0, -1.0], "B": [0.5, 3.0], "C": [1.0, 0.0]}
    model = FactorRiskModel(halflife=50.0)
    _feed(
        model,
        loadings,
        factors=lambda t: rng.standard_normal(2) * 0.01,
        returns_fn=lambda t, f: {
            name: float(np.dot(b, f)) + rng.standard_normal() * 0.005
            for name, b in loadings.items()
        },
        n=200,
    )

    tickers = ["C", "A", "B"]  # deliberately not the insertion order
    b = np.array([loadings[t] for t in tickers])
    d = np.diag([model.idiosyncratic_variance(t) for t in tickers])
    expected = b @ model.factor_covariance() @ b.T + d

    assert model.covariance(tickers) == pytest.approx(expected)


def test_residual_uses_prev_loadings_not_current_loadings() -> None:
    # Returns are exactly explained by the *previous* bar's loadings against a
    # per-bar-varying f, so the true residual is identically zero (and D -> 0).
    # If residuals were formed from the current (different) loadings, they would
    # vary bar to bar and D would inflate -- f must vary for the test to see it.
    model = FactorRiskModel(halflife=10.0)
    prev = {"A": [1.0, 1.0]}
    for t in range(6):
        f = np.array([0.01 * (t + 1), 0.0])  # varies each bar
        realized = float(np.dot(prev["A"], f))  # residual == 0 by construction
        model.update(
            loadings={"A": [9.0, 9.0]},  # differs from prev, so a bug would show
            prev_loadings=prev,
            factor_returns=f,
            returns={"A": realized},
        )

    assert model.idiosyncratic_variance("A") == pytest.approx(0.0, abs=1e-15)
