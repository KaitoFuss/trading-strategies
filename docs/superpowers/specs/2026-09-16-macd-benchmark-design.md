# MACD benchmark strategy: streaming features, Strategy, Portfolio

Date: 2026-09-16
Status: approved for planning

## Goal

Replicate the **MACD benchmark** from Poh, Wood, Roberts, Zohren, "Network
Momentum across Asset Classes" (arXiv:2308.11294), Section 4.1, Eq. 10.

This is one of three benchmarks the paper compares its network-momentum
strategy (GMOM) against. This spec covers MACD only. Long Only and LinReg
are separate, later efforts — LinReg in particular needs walk-forward OLS
retraining every 5 years, a materially different problem (training/rebalance
scheduling, not streaming features) and gets its own design.

## Why streaming, not batch

`features.py` (soon: `network_momentum/features.py`) currently takes a full
`pd.Series` of close prices and returns the whole feature history in one
call — fine for offline exploration, wrong shape for a `Strategy`, which
receives one `MarketEvent` (one bar, all tickers) at a time and must return
a `SignalEvent` before the next bar arrives. Existing strategies in
`backtester` (`ZScoreMovingAverageStrategy`) solve this with a small
per-ticker cache (`dict[Ticker, deque[float]]`) updated one value per bar.

This spec applies the same pattern to all 8 momentum/MACD features, using
`backtester`'s `Strategy` Protocol (`process_market(event) -> SignalEvent`).

**Not every feature is a pure EMA update**, which matters for what state
each one needs to cache:

| Feature component | Update kind | State needed |
|---|---|---|
| `ewm_vol` (60-day span) | EWM | O(1) scalars |
| MACD's two price EMAs (`m(t,S)`, `m(t,L)`) | EWM | O(1) scalars |
| `winsorize` (halflife=31) | EWM | O(1) scalars |
| MACD's `std(price, 63)` | plain rolling window | 63-value buffer |
| MACD's `std(MACDnorm, 252)` | plain rolling window | 252-value buffer |
| momentum's `diff(lookback)` | lag | `lookback`-value buffer |

The EWM pieces get an O(1) recursive update. The rolling-window and lag
pieces need a small bounded buffer (≤252 floats) — this is the one
correction to the original "no deque needed" assumption; a deque is
avoidable for the EWM state but not for these three.

Streaming output must match today's batch (pandas) output bar-for-bar,
within float tolerance, including pandas' exact `min_periods` warm-up
behavior. Since the batch functions are being deleted (not kept as a
reference), correctness is instead pinned by testing each primitive
directly against the equivalent pandas call (see Testing).

## Layout

```
src/trading_strategies/
  utils/
    __init__.py
    streaming.py        # EwmMean, EwmMoments, RollingStd, RollingLag
  network_momentum/
    __init__.py
    features.py          # StreamingVolScaledMomentum, StreamingMacd, StreamingFeatureSet
    strategies.py         # MacdBenchmarkStrategy
    portfolio.py           # TargetVolPortfolio
tests/
  utils/
    __init__.py
    test_streaming.py
  network_momentum/
    __init__.py
    test_features.py
    test_strategies.py
    test_portfolio.py
```

`src/trading_strategies/features.py` and `tests/test_features.py` are
deleted; their content moves into `network_momentum/features.py` /
`tests/network_momentum/test_features.py` in streaming form.
`utils/streaming.py` is deliberately outside `network_momentum/` — the four
primitives are generic (not paper-specific) and are exactly the kind of
thing a future strategy would want to reuse.

## A. `utils/streaming.py`

Four primitives, each replicating one pandas operation's *exact* semantics
one value at a time:

- **`EwmMean(span: int | None = None, halflife: float | None = None,
  min_periods: int)`** — matches `.ewm(span=..., min_periods=...).mean()`
  (pandas' *adjusted* EWM, default `adjust=True`). Recursion:
  `numer = x + (1-α)·numer_prev`, `denom = 1 + (1-α)·denom_prev`,
  `mean = numer/denom`. `.update(x) -> float | None`, `None` until
  `min_periods` observations seen. `α = 2/(span+1)` for span,
  `α = 1 - exp(-ln2/halflife)` for halflife (both match pandas' conventions).

- **`EwmMoments(span | halflife, min_periods)`** — matches
  `.ewm(...).mean()` and `.ewm(...).std()` together (they share the
  same decaying accumulators, so one class serves both). Tracks 4 decaying
  scalars: `Sx = Σw·x`, `Sxx = Σw·x²`, `Sw = Σw`, `Sw2 = Σw²`. `mean =
  Sx/Sw`; `variance = (Sw·Sxx − Sx²) / (Sw² − Sw2)` (pandas' bias-corrected
  weighted-variance formula for `bias=False`, the default `.std()` uses).
  `.update(x) -> None` (mutates state), `.mean`, `.std` properties (`None`
  until `min_periods`).

- **`RollingStd(window: int)`** — matches `.rolling(window,
  min_periods=window).std()`. A `deque(maxlen=window)` of raw values; each
  `.update(x) -> float | None` appends and, once full, calls
  `backtester.stats.mean_and_stdev` on the deque's contents (reusing the
  codebase's existing two-pass, `fsum`-based, sample-stdev convention rather
  than an incremental sum-of-squares).

- **`RollingLag(lookback: int)`** — a `deque(maxlen=lookback+1)`.
  `.update(x) -> float | None` appends and returns the value `lookback`
  steps back (the oldest element), or `None` until full.

Each primitive is a plain class with mutable internal state, no
inheritance/Protocol needed — they're used compositionally, not swapped
polymorphically.

## B. `network_momentum/features.py`

Per-ticker composed classes, each exposing `update(...) -> float | None`
(or a dict, for the aggregate):

- **`StreamingVolScaledMomentum(lookback: int, momentum_phi: bool)`** —
  owns an `EwmMoments(span=60)` for vol (fed log-returns) and a
  `RollingLag(lookback)` (fed raw close). `update(close) -> float | None`:
  computes the log-return internally (needs last close — one extra scalar),
  updates vol, computes `log(close/lag_close) / (vol * sqrt(lookback))`,
  applies `response_function` if `momentum_phi`, then winsorizes (see
  below). Returns `None` until every underlying piece is warm.

- **`StreamingMacd(short: int, long: int, macd_phi: bool)`** — owns
  `EwmMean(span=short)`, `EwmMean(span=long)`, `RollingStd(63)` (fed raw
  close), `RollingStd(252)` (fed the once-normalized MACD value).
  `update(close) -> float | None`: `macd = m_short - m_long`; `q =
  macd / std_63`; `y = q / std_252(q)`; applies `response_function` if
  `macd_phi`; then winsorizes.

- **Winsorizing**: each of the two classes above owns its own
  `EwmMoments(halflife=31)` over its own output history (matching today's
  `WINSOR_HALFLIFE = 31` constant, unchanged), clipping to `±5` EWM stdevs
  around the EWM mean — same as today's `winsorize`, just applied inline
  per-value instead of as a separate pass over a full series.

- **`StreamingFeatureSet`** — bundles 5 `StreamingVolScaledMomentum` +
  3 `StreamingMacd` per ticker. `update(close) -> dict[str, float] | None`,
  same column-name convention as today's `compute_all_features`
  (`mom_1`, ..., `macd_8_24`, ...), returns `None` only once *no* feature
  has produced a value yet (individual feature warm-up staggers, so most
  bars return a partial dict — same behavior a caller already has to handle
  with the batch version's leading `NaN`s, just shaped as "key absent"
  instead of "value is NaN").

`response_function` (the Baz et al. φ squashing) is unchanged — it's
already a pure, stateless, pointwise function; nothing to make streaming.

## C. `network_momentum/strategies.py`: `MacdBenchmarkStrategy`

```python
class MacdBenchmarkStrategy:
    def __init__(self) -> None:
        self._macd: dict[Ticker, tuple[StreamingMacd, StreamingMacd, StreamingMacd]] = {}

    def process_market(self, event: MarketEvent) -> SignalEvent:
        scores: dict[Ticker, float] = {}
        for ticker, bar in event.bars.items():
            states = self._macd.setdefault(
                ticker,
                tuple(StreamingMacd(s, l, macd_phi=True) for s, l in MACD_PAIRS),
            )
            values = [state.update(bar.close) for state in states]
            if all(v is not None for v in values):
                scores[ticker] = sum(values) / len(values)
        return SignalEvent(timestamp=event.timestamp, scores=scores)
```

Same shape as `ZScoreMovingAverageStrategy`: one cache dict keyed by
`Ticker`, `setdefault` + update, skip a ticker (no entry in `scores`) while
any of its 3 MACD pairs is still warming up.

**Paper-fidelity correction:** Eq. 10 (`x_i,t = mean of φ(y_i,t(Sk,Lk))`
over the 3 pairs) has **no winsorization step** — that's a Section 2.2
preprocessing detail for the GMOM regression's input features, not part of
the standalone MACD benchmark's definition. So `StreamingMacd` used here is
constructed with winsorizing disabled (a `winsorize: bool = True`
constructor flag, defaulting on for `StreamingFeatureSet`'s general use,
off here) — the strategy averages the three raw φ'd values directly.

## D. `network_momentum/portfolio.py`: `TargetVolPortfolio`

Implements the paper's Eq. 9 position-sizing directly:

```
weight_i,t = (1 / N_t) · score_i,t · (σ_tgt / σ_i,t)
```

- `σ_tgt = 0.15` (annualized), constructor default, overridable.
- `σ_i,t`: annualized EWM vol of daily log-returns, 60-day span — tracked
  independently inside the portfolio via one `EwmMoments(span=60)` per
  ticker (own `_returns`/`_last_price` bookkeeping, same pattern
  `InverseVolPortfolio` already uses). This duplicates the vol computation
  the `MacdBenchmarkStrategy`'s own `StreamingMacd` doesn't even need
  (MACD has no vol-scaling step) but the Portfolio does — Strategy and
  Portfolio stay decoupled per the existing architecture; nothing is
  shared between them beyond the `SignalEvent`.
- `N_t` = count of tickers that have both a score this bar and a warmed-up
  vol estimate.
- Uses the **raw score**, not its sign — MACD's signal is continuous
  (already φ-squashed to roughly `[-1, 1]`), and its magnitude is part of
  the paper's sizing, unlike `InverseVolPortfolio`'s gate-only score.
- **Gross is uncapped** — no `max_gross` renormalization, matching the
  paper exactly. `BasePortfolio.__init__` still takes `max_gross` (for
  interface consistency with the other portfolios / `factory.py`) but
  `TargetVolPortfolio` does not use it to scale anything.

## E. Testing

- `tests/utils/test_streaming.py`: each primitive vs. the equivalent pandas
  batch call, fed one value at a time over a random series, asserting
  elementwise match (including where pandas emits `NaN` for warm-up,
  `EwmMean`/`EwmMoments`/`RollingStd`/`RollingLag` return `None`).
- `tests/network_momentum/test_features.py`: `StreamingMacd` and
  `StreamingVolScaledMomentum` against hand-built pandas expressions for
  the same formula (not the old `compute_all_features` — it's gone), plus
  the existing property-style assertions ported over (e.g. "positive for a
  sustained uptrend", "bounded when phi applied").
- `tests/network_momentum/test_strategies.py`: `MacdBenchmarkStrategy`
  emits no score for a ticker until all 3 MACD pairs are warm; emits the
  correct 3-pair average once warm; drives a `MarketEvent` sequence and
  checks against a hand-computed expectation for a small synthetic series.
- `tests/network_momentum/test_portfolio.py`: `TargetVolPortfolio` weight
  formula directly (`weight == score/N * target_vol/realized_vol`), no
  gross capping under a large score/low-vol combination, `N_t` excludes
  tickers without warmed-up vol.

Coverage gate stays at 95% (`pyproject.toml`'s `--cov-fail-under=95`).

## F. Notebook

`notebooks/explore_features.ipynb` currently calls `compute_all_features`
once over a full close-price series. It's updated to loop bar-by-bar
through a `StreamingFeatureSet`, appending each bar's returned dict into a
list, then building the same plotting DataFrame from that list at the end.

## Out of scope (future specs)

- **Long Only** benchmark: trivial once the above exists — constant
  `score=1`, same `TargetVolPortfolio`. Small follow-up.
- **LinReg** benchmark and **GMOM** (the paper's actual proposed strategy):
  both need walk-forward OLS/graph-learning retraining on a 5-year
  schedule — a different problem (model lifecycle, not streaming features)
  needing its own design.
- `winsorize`'s halflife constant (`31`) vs. the paper's stated `252`: a
  pre-existing discrepancy in `features.py`, unrelated to this work,
  preserved as-is.
