"""The optimizer's problem, as a pure function of a risk-model snapshot.

``maximize alpha'w - kappa*||w - w_prev||_1`` subject to an ex-ante vol budget
and a gross-leverage cap (plus optional factor bounds). Convex SOCP, solved
with cvxpy/Clarabel. Kept free of engine types so it can be tested on
hand-built snapshots.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from backtester.core.events import Ticker
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR

from trading_strategies.risk.model import RiskModelSnapshot
from trading_strategies.utils.streaming import FloatArray

_BINDING_TOLERANCE = 1e-4


@dataclass(frozen=True)
class OptimizerParams:
    target_vol: float  # annualized
    max_gross: float
    ic: float = 0.05
    turnover_penalty_bps: float = 5.0
    factor_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class SolveResult:
    weights: FloatArray
    status: str
    binding: str
    ex_ante_vol: float  # annualized
    gross: float


def solve_weights(
    scores: Mapping[Ticker, float],
    snapshot: RiskModelSnapshot,
    w_prev: FloatArray,
    params: OptimizerParams,
) -> SolveResult:
    if not snapshot.tickers:
        return SolveResult(np.zeros(0), "empty", "none", 0.0, 0.0)

    unknown = set(params.factor_bounds) - set(snapshot.factors)
    if unknown:
        raise ValueError(f"unknown factor bound(s): {', '.join(sorted(unknown))}")

    # Grinold: alpha = IC * vol * score, so alpha is in daily return units and
    # is comparable with the turnover penalty.
    score = np.array([scores.get(t, 0.0) for t in snapshot.tickers], dtype=np.float64)
    alpha = params.ic * snapshot.sigma * score
    target_daily = params.target_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    kappa = params.turnover_penalty_bps / 1e4

    values, vectors = np.linalg.eigh(snapshot.F)
    f_half = vectors * np.sqrt(np.clip(values, 0.0, None))

    w = cp.Variable(len(snapshot.tickers))
    risk = cp.norm(  # type: ignore[attr-defined]
        cp.hstack(  # type: ignore[attr-defined]
            [f_half.T @ (snapshot.B.T @ w), cp.multiply(np.sqrt(snapshot.D), w)]  # type: ignore[attr-defined]
        )
    )
    constraints = [risk <= target_daily, cp.norm1(w) <= params.max_gross]  # type: ignore[attr-defined]
    for name, (low, high) in params.factor_bounds.items():
        exposure = snapshot.B[:, snapshot.factors.index(name)] @ w
        constraints += [exposure >= low, exposure <= high]

    objective = cp.Maximize(alpha @ w - kappa * cp.norm1(w - w_prev))  # type: ignore[attr-defined]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]

    if w.value is None:
        return SolveResult(w_prev, str(problem.status), "none", 0.0, float(np.abs(w_prev).sum()))

    weights = np.asarray(w.value, dtype=np.float64)
    vol = float(np.sqrt(snapshot.portfolio_variance(weights)) * np.sqrt(TRADING_DAYS_PER_YEAR))
    gross = float(np.abs(weights).sum())
    at_vol = vol >= params.target_vol * (1.0 - _BINDING_TOLERANCE)
    at_gross = gross >= params.max_gross * (1.0 - _BINDING_TOLERANCE)
    binding = (
        "both"
        if at_vol and at_gross
        else "vol"
        if at_vol
        else "leverage"
        if at_gross
        else "turnover"
    )
    return SolveResult(weights, str(problem.status), binding, vol, gross)
