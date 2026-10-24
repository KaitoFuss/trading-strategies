from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.risk.helpers import market_factor_config
from tests.risk.test_model import _simulate
from trading_strategies.risk.diagnostics import (
    exposure_collinearity,
    loading_drift,
    residual_correlations,
)
from trading_strategies.risk.model import FactorRiskModel, RiskModelSnapshot
from trading_strategies.utils.streaming import EwmCovariance, FloatArray


def _snapshot(
    b: FloatArray, f: FloatArray, d: FloatArray, *, momentum: FloatArray | None = None
) -> RiskModelSnapshot:
    n, k = b.shape
    tickers = tuple(f"T{i}" for i in range(n))
    factors = tuple(f"F{j}" for j in range(k - (1 if momentum is not None else 0)))
    if momentum is not None:
        factors = (*factors, "Momentum")
    return RiskModelSnapshot(
        tickers=tickers,
        factors=factors,
        B=b,
        F=f,
        D=d,
        sigma=np.sqrt(np.einsum("ik,kl,il->i", b, f, b) + d),
        factor_weights=np.zeros((n, k)),
        macro_betas=b[:, : k - (1 if momentum is not None else 0)],
        momentum=momentum,
        ready=True,
    )


def test_residual_correlations_recovers_a_planted_pair() -> None:
    b = np.array([[1.0], [1.0], [1.0]])
    f = np.array([[1.0]])
    d = np.array([1.0, 1.0, 1.0])
    snapshot = _snapshot(b, f, d)
    # empirical: assets 0 and 1 share half of their residual
    empirical = b @ f @ b.T + np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])

    report = residual_correlations(snapshot, empirical, snapshot.tickers)

    assert report.correlations.loc["T0", "T1"] == pytest.approx(0.5)
    assert report.correlations.loc["T0", "T2"] == pytest.approx(0.0)
    assert np.isnan(float(report.correlations["T0"].loc["T0"]))
    assert report.worst_pairs(1).index[0] == "T0/T1"


def test_residual_correlations_masks_factor_defining_assets() -> None:
    b = np.array([[1.0], [1.0]])
    f = np.array([[1.0]])
    d = np.array([1.0, 1e-9])
    snapshot = _snapshot(b, f, d)
    # asset 1 is fully explained: residual share ~0 -> masked, not reported
    empirical = b @ f @ b.T + np.diag([1.0, 1e-9])

    report = residual_correlations(snapshot, empirical, snapshot.tickers)

    assert report.masked == ("T1",)
    assert list(report.correlations.index) == ["T0"]
    assert report.share["T1"] < 0.05


def test_exposure_collinearity_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    betas = rng.normal(0, 1, (30, 2))
    momentum = rng.normal(0, 1, 30)
    b = np.column_stack([betas, momentum])
    snapshot = _snapshot(b, np.eye(3), np.ones(30), momentum=momentum)

    result = exposure_collinearity(snapshot)

    assert sorted(result.index) == ["F0", "F1"]
    assert result["F0"] == pytest.approx(np.corrcoef(momentum, betas[:, 0])[0, 1])


def test_loading_drift_reports_change_over_level() -> None:
    tickers = ("A", "B")
    factors = ("Equity", "Momentum")
    first = np.array([[1.0, 0.5], [1.0, 0.5]])
    second = np.array([[1.0, 1.0], [1.0, 1.0]])  # Equity still, Momentum moved
    history = [(tickers, factors, first), (tickers, factors, second)]

    frame = loading_drift(history)

    assert frame.loc["Equity", "median_abs_change"] == pytest.approx(0.0)
    assert frame.loc["Momentum", "median_abs_change"] == pytest.approx(0.5)
    drift_ratio = frame["drift_ratio"]
    assert float(drift_ratio.loc["Momentum"]) > float(drift_ratio.loc["Equity"])


def test_exposure_collinearity_without_momentum_is_empty() -> None:
    snapshot = _snapshot(np.array([[1.0], [0.5]]), np.eye(1), np.ones(2))

    assert exposure_collinearity(snapshot).empty


def test_residual_share_is_bounded_for_a_model_built_snapshot(tmp_path: Path) -> None:
    """The hand-built ``_snapshot`` fixtures bypass ``_finalize``'s rescaling,
    so a unit mismatch between ``share``'s numerator and denominator would
    not show up there. Drive the real model instead and check the invariant
    the brief promises: ``share`` is a fraction of variance, so it must stay
    in ``[0, 1]``."""
    tickers = ("MKT", "A", "B")
    config = market_factor_config(tmp_path)
    path = _simulate(300, {"A": 0.5, "B": -1.0})
    model = FactorRiskModel(tickers, config)
    empirical = EwmCovariance(tickers, span=config.corr_span, min_periods=config.corr_min_periods)

    last_close: dict[str, float] | None = None
    for closes in path:
        if last_close is not None:
            logs = {t: float(np.log(closes[t] / last_close[t])) for t in tickers}
            empirical.update(logs)
        model.update(closes)
        last_close = closes

    snapshot = model.snapshot
    assert snapshot.ready
    report = residual_correlations(snapshot, empirical.covariance(), tickers)

    # A tiny floating-point overshoot (MKT *is* the Market factor, so its
    # share is ~0 up to roundoff) is fine; a real unit mismatch would blow
    # past this tolerance by many orders of magnitude.
    tolerance = 1e-9
    assert ((report.share >= -tolerance) & (report.share <= 1.0 + tolerance)).all()
