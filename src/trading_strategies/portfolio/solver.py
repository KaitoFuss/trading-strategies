"""The optimizer's problem, as a pure function of a risk-model snapshot.

Two objectives, both under a gross-leverage cap (plus optional factor bounds):

- ``linear``: ``maximize alpha'w - turnover_penalty*||w - w_prev||_1``. The
  objective is linear, so the book sits on a limit almost every bar and its size
  ignores the signal's strength.
- ``mean_variance``: ``maximize alpha'w - (risk_aversion/2) w'Sigma w -
  turnover_penalty*||w - w_prev||_1``. The book's size follows the signal; the
  vol limit is a safety cap that should rarely bind.

Both are subject to ex-ante vol <= ``target_vol``.

Convex SOCP, solved with cvxpy/Clarabel. Kept free of engine types so it can be
tested on hand-built snapshots.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from backtester.core.events import Ticker
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR

from trading_strategies.risk.config import Objective, validate_objective
from trading_strategies.risk.model import RiskModelSnapshot
from trading_strategies.utils.streaming import FloatArray

_BINDING_TOLERANCE = 1e-4


@dataclass(frozen=True)
class OptimizerParams:
    """Tuning knobs for the optimizer's objective.

    ``information_coefficient`` is the assumed correlation between the score
    and next period's return. Grinold's ``alpha = IC * sigma * score`` uses it
    to turn a unitless score into a daily expected return.

    ``turnover_penalty_bps`` is the return required before one unit of
    one-way turnover is worth paying.

    Only the ratio of the two matters: scaling both leaves the solution
    unchanged, because the objective is ``alpha'w - turnover_penalty * ||w -
    w_prev||_1`` and multiplying the whole objective by a constant does not
    move its maximizer.
    """

    target_vol: float  # annualized
    max_gross: float
    information_coefficient: float = 0.05
    turnover_penalty_bps: float = 5.0
    factor_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    """Factor exposure bounds, keyed by factor name.

    The exposure is ``x = B^T w``, in units of the factor's own return,
    because ``B`` is vol-rescaled. So ``{"Equity": (-0.2, 0.2)}`` lets the
    book move at most 0.2x the Equity factor's return. Names must match
    ``snapshot.factors``; an unknown name raises. Empty means unconstrained.
    """
    objective: Objective = "linear"
    risk_aversion: float | None = None  # daily units; mean_variance only

    def __post_init__(self) -> None:
        validate_objective(self.objective, self.risk_aversion)


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
    alpha = params.information_coefficient * snapshot.sigma * score
    vol_limit = params.target_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    turnover_penalty = params.turnover_penalty_bps / 1e4

    values, vectors = np.linalg.eigh(snapshot.F)
    f_half = vectors * np.sqrt(np.clip(values, 0.0, None))

    w = cp.Variable(len(snapshot.tickers))
    stacked = cp.hstack(  # type: ignore[attr-defined]
        [f_half.T @ (snapshot.B.T @ w), cp.multiply(np.sqrt(snapshot.D), w)]  # type: ignore[attr-defined]
    )
    risk = cp.norm(stacked)  # type: ignore[attr-defined]
    # ||stacked||^2 == w'Sigma w, so the mean-variance penalty reuses the
    # factor form instead of an N x N quadratic form.
    turnover = turnover_penalty * cp.norm1(w - w_prev)  # type: ignore[attr-defined]
    if params.objective == "mean_variance":
        assert params.risk_aversion is not None
        variance_penalty = params.risk_aversion / 2 * cp.sum_squares(stacked)  # type: ignore[attr-defined]
        objective = cp.Maximize(alpha @ w - variance_penalty - turnover)
    else:
        objective = cp.Maximize(alpha @ w - turnover)
    constraints = [risk <= vol_limit, cp.norm1(w) <= params.max_gross]  # type: ignore[attr-defined]
    for name, (low, high) in params.factor_bounds.items():
        exposure = snapshot.B[:, snapshot.factors.index(name)] @ w
        constraints += [exposure >= low, exposure <= high]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]

    if w.value is None or problem.status != "optimal":
        return SolveResult(w_prev, str(problem.status), "none", 0.0, float(np.abs(w_prev).sum()))

    weights = np.asarray(w.value, dtype=np.float64)
    vol = float(np.sqrt(snapshot.portfolio_variance(weights)) * np.sqrt(TRADING_DAYS_PER_YEAR))
    gross = float(np.abs(weights).sum())
    binding = _binding_constraint(vol, gross, params)
    return SolveResult(weights, str(problem.status), binding, vol, gross)


def _binding_constraint(vol: float, gross: float, params: OptimizerParams) -> str:
    """Which limit stopped the optimizer: the vol limit, the leverage cap, both,
    or neither. Neither means the turnover penalty held the book where it was
    (``linear``), or the optimum is interior (``mean_variance``). Drives the
    "binding" row on the Optimizer Diagnostics page."""
    at_vol = vol >= params.target_vol * (1.0 - _BINDING_TOLERANCE)
    at_gross = gross >= params.max_gross * (1.0 - _BINDING_TOLERANCE)
    if at_vol and at_gross:
        return "both"
    if at_vol:
        return "vol"
    if at_gross:
        return "leverage"
    return "interior" if params.objective == "mean_variance" else "turnover"
