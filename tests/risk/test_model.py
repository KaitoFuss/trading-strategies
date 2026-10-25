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
    cap_common,
    residualize_exposure,
    risk_adjusted_exposure,
)
from trading_strategies.utils.streaming import EwmMoments, FloatArray

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


@pytest.mark.parametrize(("granular", "expected"), [(False, 9), (True, 18)])
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
    config = market_factor_config(
        tmp_path, vol_span=60, corr_span=250, corr_min_periods=250, idiosyncratic_risk="implied"
    )
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


def test_snapshot_arrays_are_read_only(tmp_path: Path) -> None:
    snapshot = _run(
        FactorRiskModel(["MKT", "A", "B"], market_factor_config(tmp_path)),
        _simulate(120, {"A": 0.5, "B": -1.0}),
    )

    with pytest.raises(ValueError, match="read-only"):
        snapshot.B[0, 0] = 1.0


def test_cap_common_scales_only_over_explained_rows() -> None:
    b = np.array([[2.0], [0.5]])
    f = np.array([[1.0]])
    s_ii = np.array([1.0, 1.0])

    capped, common = cap_common(b, f, s_ii)

    # row 0 claimed 4.0 of variance against 1.0 available -> halved
    assert capped[0, 0] == pytest.approx(1.0)
    assert capped[1, 0] == pytest.approx(0.5)  # untouched
    np.testing.assert_allclose(common, [1.0, 0.25])


def test_cap_common_leaves_a_well_specified_model_alone() -> None:
    b = np.array([[0.6, 0.1], [0.2, 0.3]])
    f = np.array([[1.0, 0.0], [0.0, 2.0]])
    s_ii = np.array([5.0, 5.0])

    capped, common = cap_common(b, f, s_ii)

    np.testing.assert_allclose(capped, b)
    np.testing.assert_allclose(common, np.einsum("ik,kl,il->i", b, f, b))


def test_model_variance_never_exceeds_the_floored_total(tmp_path: Path) -> None:
    """Σ_ii = common + D, and D is floored at 10% of S_ii, so the cap makes
    Σ_ii <= sigma_i^2 * 1.1 for every asset. Before the cap, factor-defining
    assets came out far above that."""
    path = _simulate(400, {"A": 0.5, "B": -1.0, "C": 0.2})
    config = market_factor_config(tmp_path, idiosyncratic_risk="implied")
    snapshot = _run(FactorRiskModel(["MKT", "A", "B", "C"], config), path)

    total = np.diag(snapshot.covariance())
    assert np.all(total <= snapshot.sigma**2 * (1.0 + 0.10) + 1e-12)


def test_capped_model_stays_psd(tmp_path: Path) -> None:
    path = _simulate(400, {"A": 0.5, "B": -1.0, "C": 0.2})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B", "C"], market_factor_config(tmp_path)), path)

    assert np.linalg.eigvalsh(snapshot.covariance()).min() > 0


def test_residualize_exposure_removes_the_beta_component() -> None:
    rng = np.random.default_rng(0)
    betas = rng.normal(0, 1, (40, 2))
    exposure = 2.0 * betas[:, 0] - 0.5 * betas[:, 1] + rng.normal(0, 0.3, 40)

    residual = residualize_exposure(exposure, betas)

    assert residual is not None
    assert residual.std() == pytest.approx(1.0)
    for k in range(betas.shape[1]):
        assert abs(np.corrcoef(residual, betas[:, k])[0, 1]) < 1e-10


def test_residualize_exposure_returns_none_without_dispersion() -> None:
    betas = np.array([[1.0], [2.0], [3.0]])
    exposure = np.array([2.0, 4.0, 6.0])  # exactly 2x the beta -> nothing left

    assert residualize_exposure(exposure, betas) is None


def test_risk_adjusted_exposure_is_orthogonal_per_unit_of_risk() -> None:
    """The overlap with the macro betas is removed in risk-adjusted units
    (beta / vol), the unit the momentum z-score lives in."""
    rng = np.random.default_rng(0)
    vol = rng.uniform(0.001, 0.02, 40)
    rel = vol / np.median(vol)
    betas = rng.normal(0, 1, (40, 2)) * rel[:, None]
    z = 2.0 * betas[:, 0] / rel + rng.normal(0, 0.3, 40)

    x = risk_adjusted_exposure(z, betas, vol)

    assert x is not None
    assert (x / rel).std() == pytest.approx(1.0)
    for k in range(betas.shape[1]):
        assert abs(np.corrcoef(x / rel, betas[:, k] / rel)[0, 1]) < 1e-10


def test_risk_adjusted_exposure_ignores_the_overall_vol_level() -> None:
    rng = np.random.default_rng(1)
    vol = rng.uniform(0.001, 0.02, 30)
    betas = rng.normal(0, 1, (30, 2))
    z = rng.normal(0, 1, 30)

    quiet = risk_adjusted_exposure(z, betas, vol)
    loud = risk_adjusted_exposure(z, betas, 3.0 * vol)

    assert quiet is not None
    assert loud is not None
    np.testing.assert_allclose(quiet, loud)


def test_risk_adjusted_exposure_scales_with_relative_vol() -> None:
    """Two assets with the same risk-adjusted residual: the one with 10x the
    vol gets 10x the exposure. This is the SHY-vs-SPY unit fix."""
    vol = np.array([0.001, 0.01, 0.01, 0.01, 0.001, 0.01])
    rel = vol / np.median(vol)
    betas = np.zeros((6, 1))
    betas[:, 0] = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]) * rel
    z = np.array([1.0, 0.0, 0.0, 1.0, -1.0, -1.0])

    x = risk_adjusted_exposure(z, betas, vol)

    assert x is not None
    assert x[0] / x[3] == pytest.approx(0.1)
    assert x[4] / x[5] == pytest.approx(0.1)


def test_risk_adjusted_exposure_returns_none_without_dispersion() -> None:
    vol = np.array([0.01, 0.02, 0.04])
    rel = vol / np.median(vol)
    betas = np.array([[1.0], [2.0], [3.0]]) * rel[:, None]
    z = np.array([2.0, 4.0, 6.0])  # exactly 2x the risk-adjusted beta

    assert risk_adjusted_exposure(z, betas, vol) is None


def _simulate_mixed_vol(n_bars: int, seed: int = 0) -> list[dict[str, float]]:
    """Three low-vol names (a short-bond-like 0.05% daily noise, tiny market
    beta) and four high-vol names (1% noise)."""
    rng = np.random.default_rng(seed)
    spec = {
        "L1": (0.02, 0.0005),
        "L2": (0.02, 0.0005),
        "L3": (0.02, 0.0005),
        "H1": (1.0, 0.01),
        "H2": (0.5, 0.01),
        "H3": (-0.5, 0.01),
        "H4": (1.5, 0.01),
    }
    closes = {"MKT": 100.0} | dict.fromkeys(spec, 100.0)
    path = [dict(closes)]
    for _ in range(n_bars - 1):
        market = rng.normal(0, 0.01)
        closes["MKT"] *= math.exp(market)
        for ticker, (beta, noise) in spec.items():
            closes[ticker] *= math.exp(beta * market + rng.normal(0, noise))
        path.append(dict(closes))
    return path


def test_low_vol_assets_do_not_carry_equity_sized_momentum_risk(tmp_path: Path) -> None:
    """Before the unit fix the low-vol names sat at the cap_common limit,
    1 / 1.1 = 0.91 of their predicted variance in momentum, on most bars."""
    tickers = ["MKT", "L1", "L2", "L3", "H1", "H2", "H3", "H4"]
    model = FactorRiskModel(tickers, market_factor_config(tmp_path))
    shares = []
    for closes in _simulate_mixed_vol(400):
        model.update(closes)
        snapshot = model.snapshot
        if not snapshot.ready or MOMENTUM not in snapshot.factors:
            continue
        k = snapshot.factors.index(MOMENTUM)
        share = snapshot.B[:, k] ** 2 * snapshot.F[k, k] / np.diag(snapshot.covariance())
        shares.append(np.mean([share[snapshot.tickers.index(t)] for t in ("L1", "L2", "L3")]))

    assert len(shares) > 100
    assert float(np.median(shares)) < 0.5


_NOISY = {"A": 0.5, "B": -1.0, "C": 0.2, "D": 0.8, "E": -0.3, "F": 1.2}


def _returns(closes: dict[str, float], last: dict[str, float]) -> dict[str, float]:
    return {t: math.log(c / last[t]) for t, c in closes.items() if t in last}


def test_residual_d_is_the_ewma_of_realized_residuals(tmp_path: Path) -> None:
    """Replays the model: eps_t = r_t - B_{t-1} f_t from the previous
    snapshot, one EWMA per ticker, only on bars where the ticker has a
    return. C misses every 7th bar after bar 100."""
    config = market_factor_config(tmp_path, idiosyncratic_risk="residual")
    tickers = ["MKT", *_NOISY]
    model = FactorRiskModel(tickers, config)
    replay = {t: EwmMoments(span=config.vol_span, min_periods=config.vol_span) for t in tickers}
    last: dict[str, float] = {}
    for i, full in enumerate(_simulate(300, _NOISY)):
        closes = {t: c for t, c in full.items() if not (t == "C" and i > 100 and i % 7 == 0)}
        previous = model.snapshot
        returns = _returns(closes, last)
        if previous.ready:
            f = previous.factor_returns(returns)
            r = np.array([returns.get(t, 0.0) for t in previous.tickers])
            eps = r - previous.B @ f
            for k, t in enumerate(previous.tickers):
                if t in returns:
                    replay[t].update(float(eps[k]))
        last.update(closes)
        model.update(closes)

    snapshot = model.snapshot
    for k, t in enumerate(snapshot.tickers):
        std = replay[t].std
        assert std is not None
        floor = config.residual_floor * snapshot.sigma[k] ** 2
        assert snapshot.D[k] == pytest.approx(max(floor, std**2), rel=1e-12)


def test_residual_d_recovers_the_true_noise_variance(tmp_path: Path) -> None:
    """Each name has 0.5% daily noise. The momentum regression absorbs about
    1/N of it across 7 names (F carries that part), so the median lands near
    0.85 of the true variance; the prototype measured 0.78-0.88 over 5 seeds."""
    config = market_factor_config(tmp_path, vol_span=100, idiosyncratic_risk="residual")
    snapshot = _run(FactorRiskModel(["MKT", *_NOISY], config), _simulate(600, _NOISY))

    ratios = [snapshot.D[snapshot.tickers.index(t)] / 0.005**2 for t in _NOISY]
    assert 0.7 < float(np.median(ratios)) < 1.1


def test_late_ticker_uses_the_implied_d_until_warm(tmp_path: Path) -> None:
    path = _simulate(300, _NOISY)
    late = [
        {t: c for t, c in closes.items() if not (t == "F" and i < 150)}
        for i, closes in enumerate(path)
    ]
    tickers = ["MKT", *_NOISY]
    implied = FactorRiskModel(tickers, market_factor_config(tmp_path, idiosyncratic_risk="implied"))
    residual = FactorRiskModel(
        tickers, market_factor_config(tmp_path, idiosyncratic_risk="residual")
    )
    joined = False
    for closes in late:
        implied.update(closes)
        residual.update(closes)
        if "F" in residual.snapshot.tickers:
            k = residual.snapshot.tickers.index("F")
            assert residual.snapshot.D[k] == implied.snapshot.D[implied.snapshot.tickers.index("F")]
            joined = True
            break
    assert joined


@pytest.mark.parametrize("mode", ["implied", "residual"])
def test_covariance_is_psd_and_floor_holds(tmp_path: Path, mode: str) -> None:
    path = _simulate(300, {"A": 0.5, "B": -1.0})
    config = market_factor_config(tmp_path, idiosyncratic_risk=mode)
    snapshot = _run(FactorRiskModel(["MKT", "A", "B"], config), path)

    assert np.linalg.eigvalsh(snapshot.covariance()).min() > 0
    # MKT *is* the Market factor: without the floor its specific risk is ~0.
    mkt = snapshot.tickers.index("MKT")
    assert snapshot.D[mkt] >= 0.10 * snapshot.sigma[mkt] ** 2 * 0.99


def test_residual_covariance_is_psd_without_a_floor(tmp_path: Path) -> None:
    config = market_factor_config(tmp_path, residual_floor=0.0, idiosyncratic_risk="residual")
    snapshot = _run(FactorRiskModel(["MKT", *_NOISY], config), _simulate(300, _NOISY))

    assert np.all(snapshot.D >= 0)
    assert np.linalg.eigvalsh(snapshot.covariance()).min() > -1e-15


@pytest.mark.parametrize("mode", ["implied", "residual"])
def test_bad_closes_do_not_poison_the_snapshot(tmp_path: Path, mode: str) -> None:
    path = _simulate(160, {"A": 0.5, "B": -1.0, "C": 0.2})
    config = market_factor_config(tmp_path, idiosyncratic_risk=mode)
    model = FactorRiskModel(["MKT", "A", "B", "C"], config)
    _run(model, path[:150])

    model.update(path[150] | {"A": math.nan, "B": 0.0, "C": -1.0})
    for closes in path[151:]:
        model.update(closes)
        snapshot = model.snapshot
        assert snapshot.ready
        for array in (snapshot.B, snapshot.F, snapshot.D):
            assert np.isfinite(array).all()
