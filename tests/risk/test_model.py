from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tests.risk.helpers import market_factor_config
from trading_strategies.risk.model import (
    IDIOSYNCRATIC,
    MOMENTUM,
    FactorRiskModel,
    RiskModelSnapshot,
)
from trading_strategies.utils.streaming import FloatArray

REPO_FACTORS = Path(__file__).resolve().parents[2] / "configs" / "factors.json"


def _simulate(n_bars: int, betas: dict[str, float], seed: int = 0) -> list[dict[str, float]]:
    """Closes where each ticker's log return = beta * market + noise."""
    rng = np.random.default_rng(seed)
    closes = {"MKT": 100.0} | dict.fromkeys(betas, 100.0)
    path = [dict(closes)]
    for _ in range(n_bars - 1):
        market = rng.normal(0, 0.01)
        closes["MKT"] *= math.exp(market)
        for ticker, beta in betas.items():
            closes[ticker] *= math.exp(beta * market + rng.normal(0, 0.005))
        path.append(dict(closes))
    return path


def _run(model: FactorRiskModel, path: list[dict[str, float]]) -> RiskModelSnapshot:
    for closes in path:
        model.update(closes)
    return model.snapshot


def _weight_vector(snapshot: RiskModelSnapshot, weights: dict[str, float]) -> FloatArray:
    return np.array([weights.get(t, 0.0) for t in snapshot.tickers], dtype=np.float64)


def test_not_ready_until_correlation_warm(tmp_path: Path) -> None:
    path = _simulate(60, {"A": 0.5, "B": -1.0})
    model = FactorRiskModel(["MKT", "A", "B"], market_factor_config(tmp_path))

    _run(model, path[:50])  # 49 returns < corr_min_periods=50
    assert model.snapshot.ready is False

    model.update(path[50])
    assert model.snapshot.ready is True


def test_recovers_known_betas(tmp_path: Path) -> None:
    path = _simulate(600, {"A": 0.5, "B": -1.0})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B"], market_factor_config(tmp_path)), path)

    market = snapshot.factors.index("Market")
    betas = dict(zip(snapshot.tickers, snapshot.B[:, market], strict=True))
    assert betas["MKT"] == pytest.approx(1.0, abs=0.2)
    assert betas["A"] == pytest.approx(0.5, abs=0.2)
    assert betas["B"] == pytest.approx(-1.0, abs=0.2)


def test_covariance_is_psd_and_floor_holds(tmp_path: Path) -> None:
    path = _simulate(300, {"A": 0.5, "B": -1.0})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B"], market_factor_config(tmp_path)), path)

    assert np.linalg.eigvalsh(snapshot.covariance()).min() > 0
    # MKT *is* the Market factor: without the floor its specific risk is ~0.
    mkt = snapshot.tickers.index("MKT")
    assert snapshot.D[mkt] >= 0.10 * snapshot.sigma[mkt] ** 2 * 0.99


def test_momentum_joins_once_its_variance_is_warm(tmp_path: Path) -> None:
    path = _simulate(300, {"A": 0.5, "B": -1.0, "C": 0.2})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B", "C"], market_factor_config(tmp_path)), path)

    assert snapshot.factors == ("Market", MOMENTUM)
    assert snapshot.momentum is not None
    assert snapshot.B.shape == (4, 2)


def test_risk_decomposition_sums_to_one(tmp_path: Path) -> None:
    path = _simulate(300, {"A": 0.5, "B": -1.0, "C": 0.2})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B", "C"], market_factor_config(tmp_path)), path)
    w = _weight_vector(snapshot, {"A": 0.5, "B": 0.3, "C": -0.2})

    shares = snapshot.risk_decomposition(w)

    assert set(shares) == {*snapshot.factors, IDIOSYNCRATIC}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert snapshot.portfolio_variance(w) == pytest.approx(w @ snapshot.covariance() @ w)


def test_zero_portfolio_has_zero_risk_shares(tmp_path: Path) -> None:
    path = _simulate(120, {"A": 0.5, "B": -1.0})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B"], market_factor_config(tmp_path)), path)

    shares = snapshot.risk_decomposition(_weight_vector(snapshot, {}))

    assert all(value == 0.0 for value in shares.values())


def test_macro_factor_returns_are_residualized_portfolio_returns(tmp_path: Path) -> None:
    path = _simulate(120, {"A": 0.5, "B": -1.0})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B"], market_factor_config(tmp_path)), path)
    returns = {"MKT": 0.01, "A": 0.02, "B": -0.01}

    result = snapshot.factor_returns(returns)

    r = np.array([returns[t] for t in snapshot.tickers])
    np.testing.assert_allclose(
        result[: snapshot.factor_weights.shape[1]], snapshot.factor_weights.T @ r
    )


@pytest.mark.parametrize(("granular", "expected"), [(False, 9), (True, 16)])
def test_granular_toggle_changes_factor_count(
    tmp_path: Path, granular: bool, expected: int
) -> None:
    tickers = sorted(
        {t for entry in json.loads(REPO_FACTORS.read_text())["factors"] for t in entry["weights"]}
    )
    rng = np.random.default_rng(3)
    closes = dict.fromkeys(tickers, 100.0)
    model = FactorRiskModel(
        tickers,
        market_factor_config(tmp_path, factors_file=str(REPO_FACTORS), granular=granular),
    )
    for _ in range(150):
        closes = {t: c * math.exp(rng.normal(0, 0.01)) for t, c in closes.items()}
        model.update(closes)

    assert len(model.snapshot.factors) == expected
    assert model.snapshot.factors[-1] == MOMENTUM


def test_empty_snapshot_is_not_ready() -> None:
    snapshot = RiskModelSnapshot.empty()

    assert snapshot.ready is False
    assert snapshot.factor_returns({"A": 0.01}).size == 0
    assert snapshot.momentum_return({"A": 0.01}) is None


def _simulate_vol_jump(n_bars: int, jump_at: int, seed: int = 11) -> list[dict[str, float]]:
    """MKT and A (true beta 1.0); every return is 3x larger from ``jump_at``."""
    rng = np.random.default_rng(seed)
    closes = {"MKT": 100.0, "A": 100.0}
    path = [dict(closes)]
    for bar in range(1, n_bars):
        vol = 3.0 if bar >= jump_at else 1.0
        market = rng.normal(0, 0.01 * vol)
        closes["MKT"] *= math.exp(market)
        closes["A"] *= math.exp(market + rng.normal(0, 0.005 * vol))
        path.append(dict(closes))
    return path


def _unsplit_covariance(model: FactorRiskModel, snapshot: RiskModelSnapshot) -> FloatArray:
    """Sigma built without any factor-side rescale:
    diag(k) (b F b^T + D_raw) diag(k), with k = sigma_short / sqrt(S_ii)."""
    s = model._corr.covariance()
    s_ii = np.diag(s)
    b = snapshot.macro_betas
    w = snapshot.factor_weights
    f = w.T @ s @ w
    specific = np.maximum(0.10 * s_ii, s_ii - np.einsum("ik,kl,il->i", b, f, b))
    k = snapshot.sigma / np.sqrt(s_ii)
    return np.outer(k, k) * (b @ f @ b.T + np.diag(specific))


def test_exposures_stay_in_real_units_after_a_vol_jump(tmp_path: Path) -> None:
    jump_at = 700
    path = _simulate_vol_jump(1000, jump_at)
    config = market_factor_config(tmp_path, vol_span=60, corr_span=250, corr_min_periods=250)
    model = FactorRiskModel(["MKT", "A"], config)

    for bar, closes in enumerate(path):
        model.update(closes)
        if bar in (jump_at + 60, len(path) - 1):
            snapshot = model.snapshot
            assert snapshot.factors == ("Market",)
            beta = snapshot.B[snapshot.tickers.index("A"), 0]
            assert beta == pytest.approx(1.0, abs=0.2), f"bar {bar}"
            np.testing.assert_allclose(
                snapshot.covariance(), _unsplit_covariance(model, snapshot), rtol=1e-12
            )


def test_bad_closes_do_not_poison_the_snapshot(tmp_path: Path) -> None:
    path = _simulate(160, {"A": 0.5, "B": -1.0, "C": 0.2})
    model = FactorRiskModel(["MKT", "A", "B", "C"], market_factor_config(tmp_path))
    _run(model, path[:150])

    model.update(path[150] | {"A": math.nan, "B": 0.0, "C": -1.0})
    for closes in path[151:]:
        model.update(closes)
        snapshot = model.snapshot
        assert snapshot.ready
        for array in (snapshot.B, snapshot.F, snapshot.D):
            assert np.isfinite(array).all()


def test_snapshot_arrays_are_read_only(tmp_path: Path) -> None:
    snapshot = _run(
        FactorRiskModel(["MKT", "A", "B"], market_factor_config(tmp_path)),
        _simulate(120, {"A": 0.5, "B": -1.0}),
    )

    with pytest.raises(ValueError, match="read-only"):
        snapshot.B[0, 0] = 1.0
