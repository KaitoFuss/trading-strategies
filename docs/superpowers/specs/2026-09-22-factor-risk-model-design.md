# Factor Risk Model + Attribution Report — Design (Spec 1 of 2)

**Status:** approved (2026-09-22)
**Date:** 2026-09-22
**Follow-up:** Spec 2, `2026-09-22-optimized-portfolio-design.md` (uses this model)

## Goal

Build one factor risk model that serves two purposes:

1. **Attribution (this spec):** show how much of a strategy is equity beta, momentum,
   US vs. international, rates, and so on. The output goes into the existing backtest
   PDF report.
2. **Optimization (Spec 2):** supply the covariance `Σ = B F Bᵀ + D` to a constrained
   portfolio optimizer.

The model must be **streaming and causal**. It updates bar by bar and uses only past
data. Spec 2 can then run the same model inside a portfolio.

## Non-goals

- The optimizer, factor-exposure constraints and turnover control. These are in Spec 2.
- Style factors other than momentum, such as carry or value. The design leaves room
  for them. They are not built now.
- Returns-based regression attribution. It gives wrong results for a trend strategy,
  whose exposures change over time on purpose. Holdings-based attribution is used
  instead.

## Model

### Structure

```
Σ = B F Bᵀ + D                        (N assets, K factors)
x = Bᵀ w                              (portfolio factor exposures)
```

This is a **hybrid model**, like Barra:

- **Macro factors** use time-series betas. Each factor is a fixed portfolio of
  universe ETFs.
- **Style factors** (momentum only for now) use characteristic-based exposures.

### Macro factors (Two Sigma Factor Lens style, hierarchical)

Each factor is a weight vector `W_k` over universe tickers. Factors are
**residualized in level order**: a factor at level `n` is residualized against all
factors at earlier levels. A residualized factor is still a portfolio of ETFs:
`W_k' = W_k − Σ_j β_kj W_j'`, with `β` computed from the current `S`.

| Level | Factor | Proxy (weights) | Toggle |
|---|---|---|---|
| 1 Core | Equity | 0.6 SPY + 0.3 EFA + 0.1 EEM | always on |
| 1 Core | Rates | IEF | always on |
| 2 Core | Credit | 0.5 HYG + 0.5 LQD | always on |
| 2 Core | Commodities | DBC | always on |
| 3 Secondary | Emerging Markets | EEM | always on |
| 3 Secondary | Foreign Currency | −UUP | always on |
| 3 Secondary | Local Inflation | TIP | always on |
| 3 Secondary | US vs Intl | SPY − EFA | always on |
| 4 Granular | Size | IWM − SPY | `granular` |
| 4 Granular | Tech | QQQ − SPY | `granular` |
| 4 Granular | Cyclical vs Defensive | mean(XLI,XLB,XLY,XLF) − mean(XLP,XLU,XLV) | `granular` |
| 4 Granular | Curve | TLT − SHY | `granular` |
| 4 Granular | Energy | mean(XLE, OIH, USO) | `granular` |
| 4 Granular | Precious Metals | GLD | `granular` |
| 4 Granular | Yen | FXY | `granular` |

Level 1 and level 2 split the Two Sigma "core" group. Credit and Commodities are
residualized against Equity and Rates only, as in the Two Sigma Lens.

Factor definitions live in a JSON file (`configs/factors.json`), not in code.
Each entry has: `name`, `level`, `group` (`core` / `secondary` / `granular`),
`weights` (ticker → weight). The `granular` group can be switched on or off in the
config.

If a factor ticker has no data yet (not listed yet), the factor uses the tickers that
are available. If no ticker is available, the factor is left out of the model on that
bar.

### Style factor: momentum

- **Exposure:** `m_i = r_i(t−252 → t−21) / σ_i`, where `σ_i` is the annualized vol.
  This is the 12-1 month return, scaled by vol. Winsorize at median ± 3 robust sigmas
  (1.4826 · MAD), then z-score across the assets that are available on that bar.
- **Factor return:** a cross-sectional regression of the asset **macro-residual**
  returns on the exposures: `f_mom = Σ m_i ε_i / Σ m_i²`.
- **Factor variance:** an EWM variance of `f_mom`, with the same span as the
  correlations.
- `F` is block-diagonal: `[F_macro, 0; 0, var_mom]`. The macro-residual regression
  makes momentum approximately orthogonal to the macro factors.

### Estimation

One streaming estimator gives everything:

- **Asset vols `σ_i`:** EWM stdev of daily log returns, span 60. This matches the
  existing `TargetVolPortfolio`.
- **Asset correlations:** the EWM covariance `S` of **daily log returns**, span 250,
  converted to a correlation matrix. All ETFs trade in New York at the same close, so
  multi-day returns for non-synchronous closes are not needed.
- **Pairwise-complete:** each pair `(i, j)` keeps its own EWM weight sum. Assets with a
  late listing date still enter the model when they start trading.
- **Macro factors from `S`** (a deterministic projection, no extra state):
  ```
  F_macro = W'ᵀ S W'
  B_macro = S W' F_macro⁻¹
  ```
- **Combine:** `Σ = D_σ · C · D_σ`, where `C` is the factor-model correlation. Short-span
  vols and long-span correlations are therefore separate.
- **Specific risk:** `D_i = max(floor · S_ii, S_ii − (B F Bᵀ)_ii)`. `residual_floor`
  defaults to `0.10`. Assets that define a factor (IEF, DBC, GLD…) would otherwise get
  about zero specific risk.
- **PSD safety:** clip the eigenvalues of `F` at a small positive value before use.
- **Warm-up:** the model reports `ready = False` until the correlation EWM has
  `min_periods` observations (default 250). The vol estimate has its own 60-bar
  warm-up.

### Model output

`FactorRiskModel.snapshot()` returns a frozen `RiskModelSnapshot`:

- `tickers`, `factors` (names in level order, momentum last)
- `B` (N×K), `F` (K×K), `D` (N), `sigma` (N)
- `factor_returns` for the last bar (K). This is used for attribution.
- `ready: bool`

Methods on the snapshot: `covariance()`, `exposures(w)`, `portfolio_variance(w)`,
`risk_decomposition(w)`.

## Attribution

Attribution is holdings-based. For each bar `t`, with weights `w_t` at the close of
bar `t`:

| Output | Formula |
|---|---|
| Exposure | `x_t = B_tᵀ w_t` |
| Risk share, factor k | `x_k (F x)_k / σ²_p` |
| Risk share, specific | `wᵀ D w / σ²_p` |
| Return contribution, factor k over bar t+1 | `x_{t,k} · f_{t+1,k}` |
| Specific + residual | `r_p,t+1 − Σ_k x_{t,k} f_{t+1,k}` |

- The residual includes specific returns, cash, trading costs and intraday trading.
  The report labels it **"Specific + other"**.
- Contributions are **summed arithmetically** over time. This is exact per bar, and
  it is simple. Geometric linking is not used.
- **Identity test:** the factor contributions plus the residual equal the portfolio
  return on each bar, exactly.

Attribution runs **after** the backtest. It reads the tracker's weight history and
replays the factor model over the same price data. The model is causal, so the
replay gives the same values as a live run.

## Changes in `backtester` (separate PR, merged first)

1. **Per-bar weight history.** `PerformanceTracker` records
   `weights_history: list[tuple[datetime, dict[Ticker, float]]]` at the same point as
   the equity mark.
   - `weight_i = signed_qty_i · price_i / equity`.
   - Quantity comes from the tracker's own lot ledger. The price is the bar close,
     with a fallback to the last close the tracker has seen.
   - The plan must check the engine event order, so that the weights recorded at bar
     `t` are the holdings that earn the return of bar `t+1`.
2. **Public report API.**
   - `ReportPage` protocol: `def render(self, pdf: PdfPages) -> None`.
   - `save_report(..., extra_pages: Sequence[ReportPage] = ())`. The extra pages go
     after the monthly and cost-sweep pages and before the config page.
   - Make the page helpers public: `new_page`, `save_page`, `style_table`,
     `draw_heatmap`, and the theme colors. New pages then match the existing style.
3. The `trading-strategies` pin in `pyproject.toml` moves to the new `backtester`
   commit.

`backtester` stays factor-agnostic. It has no knowledge of factors.

## Changes in `trading-strategies`

New package `src/trading_strategies/risk/`:

| Module | Content |
|---|---|
| `covariance.py` | `EwmCovariance`: streaming, pairwise-complete EWM covariance of daily returns |
| `factors.py` | `FactorDefinition`, `load_factors(path, granular)`, hierarchical residualization |
| `momentum.py` | Streaming momentum exposures and factor return |
| `model.py` | `FactorRiskModel` (streaming `update(bar)`), `RiskModelSnapshot` |
| `attribution.py` | `run_attribution(weights_history, prices, model_config) -> AttributionResult` (pure) |
| `report_pages.py` | `ReportPage` implementations for the PDF |

**Config:** `MacdBacktestConfig` gets an optional nested section. If it is absent,
the report has no factor pages.

```json
"factor_model": {
    "factors_file": "configs/factors.json",
    "granular": false,
    "vol_span": 60,
    "corr_span": 250,
    "corr_min_periods": 250,
    "residual_floor": 0.10,
    "momentum": {"lookback": 252, "skip": 21}
}
```

**Runner / script:** `scripts/run_backtest.py` runs the attribution for each leg
(the strategy and buy-and-hold) and passes the pages to `save_report(extra_pages=...)`.

## Report pages

1. **Factor Exposures:** exposure per factor over time (one line per factor, grouped
   by level), and a table of mean / min / max exposure per leg.
2. **Risk Decomposition:** the mean share of ex-ante variance per factor + specific
   (bar chart per leg), and ex-ante vol vs 60-day realized vol over time.
3. **Return Attribution:** cumulative contribution per factor over time, and a table
   of annualized contribution per factor + "Specific + other" per leg.
4. **Model Health:** the bias statistic, i.e. the rolling std of
   `r_p,t+1 / σ̂_p,t`. The target is ≈ 1. The page shows the full-period value and
   the rolling series.

**Expected sanity results:**

- Buy-and-hold is mostly equity beta.
- The MACD strategy has large momentum exposure and time-varying equity exposure.

## Testing

- **Covariance:** a streaming EWM matches a pandas `ewm().cov()` reference. The
  pairwise-complete handling works with a late-listed asset.
- **Factors:** residualized factors have about zero covariance under `S`. Switching
  `granular` changes the number of factors from 9 to 16 (8 macro + momentum, then
  15 macro + momentum).
- **Model:** on synthetic data with a known `B`, the model recovers it within a
  tolerance. `Σ` is PSD. `D` respects the floor. `ready` is correct during warm-up.
- **Momentum:** the exposures are z-scored and winsorized. The factor return matches a
  hand-computed regression.
- **Attribution:** the identity `Σ contributions + residual = portfolio return` holds
  per bar. A single-asset SPY portfolio has an equity exposure ≈ its beta.
- **Backtester:** the weight history matches the positions. `extra_pages` are
  rendered in the correct place. The existing report tests still pass.
- **Integration:** `run_backtest.py` on the synthetic fixture produces a PDF with the
  factor pages.
- Coverage ≥ 95% (the repo gate). `mypy --strict`. `ruff`.

## Success criteria

- The report shows exposures, risk decomposition and return attribution for both legs.
- The bias statistic for buy-and-hold is within 0.8–1.2 over the full period.
- The model API is ready for Spec 2 without changes: a streaming `update` and a
  `snapshot` with `B`, `F`, `D`, `sigma`.

## Decisions made in review

- **Equity proxy** = 0.6 SPY + 0.3 EFA + 0.1 EEM (approximate global market-cap
  weights).
- **Correlation span** = 250 days.
- **Residual floor = 10%** of asset variance. Tune it with the bias statistic, not
  with Sharpe.
- **Daily returns for correlations.** All ETFs trade in New York at the same close.
