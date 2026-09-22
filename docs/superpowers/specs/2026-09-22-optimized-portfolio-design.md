# Optimized Portfolio — Design (Spec 2 of 2)

**Status:** approved (2026-09-22). Implementation starts after Spec 1 is merged.
**Date:** 2026-09-22
**Depends on:** Spec 1, `2026-09-22-factor-risk-model-design.md` (`FactorRiskModel`,
`RiskModelSnapshot`, factor attribution pages)

## Goal

Replace the scaling heuristics (Eq. 9 and the realized-vol rescale) with a
constrained optimizer:

- Maximize the alpha implied by the strategy scores.
- Regularize with a turnover penalty.
- Keep the ex-ante vol at or below the target, and the gross leverage at or below a cap.

The current portfolios stay as benchmarks. This is a new, additional portfolio type.

## Non-goals

- Changes to strategies or scores.
- Per-asset weight caps, asset-class budgets, or net-exposure limits. Vol and
  leverage are enough (decision from brainstorming).
- A multi-period optimization.

## Problem

On each bar, with snapshot `(B, F, D)` from Spec 1 and the current weights `w_prev`:

```
maximize    αᵀw − κ · ‖w − w_prev‖₁
subject to  ‖F^½ Bᵀ w‖² + ‖D^½ w‖² ≤ σ_tgt²      (ex-ante vol, daily units)
            ‖w‖₁ ≤ L                            (gross leverage = max_gross)
            lo_k ≤ (Bᵀ w)_k ≤ hi_k              (OPTIONAL factor bounds, off by default)
```

- This is a convex SOCP. Solve it with `cvxpy` and the default Clarabel solver.
- **Alpha from score (Grinold):** `α_i = IC · σ_i,daily · score_i`. With this, `α` has
  return units and is comparable with `κ`. Default `IC = 0.05`.
- **Turnover penalty:** `κ = turnover_penalty_bps / 1e4` per unit of one-way turnover.
  It is an explicit parameter (default `5.0` bp). It is not derived from `cost_bps`,
  because the current configs use `cost_bps = 0`, which would switch the penalty off.
- **Factor bounds (toggle):** config `factor_bounds: {"Equity": [-0.2, 0.2], ...}`.
  Each entry adds one pair of linear constraints. Factor names must match the names in
  the factor file. An unknown name gives an error at startup.
- A linear objective always pushes to a constraint. The portfolio logs which
  constraint binds on each bar: vol, leverage, or neither (turnover-limited).

## Behavior

- **Universe per bar:** tickers with a price and a model entry. Tickers without a
  score get `α = 0`. The optimizer can then reduce them, subject to the turnover
  penalty.
- **No price on this bar:** the ticker is left out of the problem. Its position is
  not changed.
- **Warm-up:** while `snapshot.ready` is `False`, the portfolio stays flat and logs it.
- **Solver failure or infeasible problem:** keep `w_prev` (no orders) and log a
  warning with the solver status. Count the failures for the report.
- **Orders:** target weights → rounded share deltas. Reuse the existing
  `_orders_from_targets` logic. Move it to a shared helper if needed.
- The model is updated inside `process_signal` before optimizing, in the same way that
  `TargetVolPortfolio._update_vol` works now.

## Code

| Location | Content |
|---|---|
| `src/trading_strategies/portfolio/optimized.py` | `OptimizedPortfolio(BasePortfolio)` |
| `src/trading_strategies/portfolio/problem.py` | Pure function `solve_weights(alpha, snapshot, w_prev, params) -> SolveResult` (weights, status, binding constraint) |
| `pyproject.toml` | Add `cvxpy` dependency |

`solve_weights` is pure and has no engine dependency, so it is easy to test.

**Config:** replace `rescale_to_portfolio_vol: bool` with
`portfolio: "target_vol" | "rescaled_target_vol" | "optimized"`, and migrate the two
existing configs. Add an optional section:

```json
"optimizer": {
    "ic": 0.05,
    "turnover_penalty_bps": 5.0,
    "factor_bounds": {}
}
```

`"optimized"` requires the `factor_model` section from Spec 1. It is an error at
startup if that section is missing.

## Report

- The factor attribution pages from Spec 1 apply automatically.
- One extra page, **Optimizer Diagnostics**: ex-ante vol vs target over time, gross
  leverage vs cap, the share of bars per binding constraint, solver failure count,
  and turnover.

## Testing

- **Closed form:** with no turnover penalty and only the vol constraint binding, the
  solution is `∝ Σ⁻¹ α`, scaled to the target vol.
- **Leverage binding:** with a very low `L`, gross = `L` and the vol constraint is
  slack.
- **Turnover:** with a very high `κ`, the result is `w = w_prev`.
- **Factor bounds:** the bounds are respected when on. The result is unchanged when
  off.
- **Failure path:** a forced solver failure leaves `w_prev` and logs a warning.
- **Warm-up:** there are no orders while the model is not ready.
- **Integration:** an `"optimized"` config runs end to end on the synthetic fixture.

## Evaluation (after implementation)

Compare `optimized` vs `target_vol` vs `rescaled_target_vol`, with a nonzero
`cost_bps`, on:

- Sharpe, net of costs
- Realized vol vs target (tracking)
- Annual turnover and maximum gross leverage
- Max drawdown
- The bias statistic from Spec 1

If `optimized` does not beat `rescaled_target_vol` net of costs, write down why
before tuning. Do not overfit `IC` or `κ` to the backtest.

## Decisions made in review

- **Defaults:** `IC = 0.05`, `turnover_penalty_bps = 5`. Only their ratio matters.
- **Warm-up:** stay flat. No fallback to Eq. 9 sizing.
- **Config:** the `portfolio` field replaces the `rescale_to_portfolio_vol` flag
  (a breaking change; migrate the two existing configs).
