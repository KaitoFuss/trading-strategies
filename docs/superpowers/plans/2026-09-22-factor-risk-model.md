# Factor Risk Model + Attribution Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a streaming hybrid factor risk model (hierarchical macro factors + momentum) and add holdings-based factor attribution pages to the existing backtest PDF report.

**Architecture:** Two repos.

- **`backtester`** (`~/GitHub/Projects/backtester`) gets two generic features:
  - a per-bar weight history on `PerformanceTracker`;
  - a public report-extension API (`ReportPage` + `save_report(extra_pages=...)` + public page helpers).
- **`trading-strategies`** gets a new package `trading_strategies/risk/`:
  - streaming EWM covariance → factor definitions/residualization → momentum → `FactorRiskModel` → `run_attribution` → report pages.
  - After a backtest, the script replays the causal model over the same price data, attributes each leg, and passes the pages to `save_report`.

**Tech Stack:** Python 3.12, numpy, pandas, matplotlib, `backtester` (git dependency), pytest, mypy --strict, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-22-factor-risk-model-design.md` (read it first). Spec 2 (`2026-09-22-optimized-portfolio-design.md`) is out of scope. Its plan is written after this plan is merged.

## Global Constraints

- Python `>=3.12`, `from __future__ import annotations` at the top of every new module in `trading-strategies`.
- `mypy --strict` on `src`, `tests` (and `scripts` in `trading-strategies`). Every new function has full annotations. Use `pd.Series[float]`, not bare `pd.Series`.
- `ruff format` + `ruff check` (rules `E, W, F, I, UP, B, C4, SIM, PT, RUF`), line length 100, double quotes.
- `pytest` with `--cov-fail-under=95` in **both** repos.
- `trading-strategies` commits: **no** `Co-Authored-By` / "Generated with" trailers.
- All risk quantities inside the model are in **daily** units. Annualize only for display, with `TRADING_DAYS_PER_YEAR` from `backtester.tracker.metrics`.
- Decisions fixed in the spec review:
  - `residual_floor = 0.10`
  - correlations from **daily** log returns, `corr_span = 250`, `corr_min_periods = 250`
  - `vol_span = 60`
  - equity proxy `0.6 SPY + 0.3 EFA + 0.1 EEM`
  - momentum `lookback = 252`, `skip = 21`

## Deviations from the spec (decided while planning)

- **Factor returns are a snapshot method, not a field.** `RiskModelSnapshot.factor_returns(asset_returns)` computes the returns of bar `t+1` from the snapshot of bar `t`. This keeps attribution causal.
- **Vol rescaling uses `S_ii`, not the model variance.** `B` and `D` are scaled by `σ_short / sqrt(S_ii)`. The residual floor therefore **adds** variance (conservative) and does not shrink betas. For example, a SPY-only portfolio keeps an Equity exposure ≈ 1. The factor side then takes its own vol regime: `F` is scaled by `g gᵀ` and `B` divided by `g`, where `g_k` = short-span / long-span vol of factor `k` (`W'ᵀ V W'` over `F_macro`; for Momentum, two EWMs of its return). `Σ` does not change, but exposures stay in real units after a vol spike.
- **Weight-history timing.** `weights_history[t]` = the holdings that **earned bar t's return**. The quantities are held from close `t−1` to close `t`. They are valued at close `t−1` and divided by the equity mark of `t−1`. The first entry is `{}`. The engine calls `track_market` before the strategy and portfolio on each bar, so the lot ledger at `track_market(t)` holds exactly those quantities.
- **Which tickers enter the model on a bar.** A ticker enters only when its vol and correlation estimates are warm. It also needs finite covariances with all other tickers in the model.
- **Missing factor legs.** Long and short legs are renormalized separately over the available tickers. If a leg has no available ticker, the factor is inactive on that bar.
- **Momentum winsorizing uses a robust scale.** The clip is at median ± 3 · 1.4826 · MAD, then a z-score. The spec's "winsorize ±3, then z-score" done with plain z-scores leaves a single outlier where it was.

---

## File Structure

**backtester** (`~/GitHub/Projects/backtester`, new branch `feature/weights-history-report-pages`)

| File | Change |
|---|---|
| `src/backtester/tracker/metrics.py` | `PerformanceTracker.weights_history` |
| `src/backtester/tracker/report.py` | `ReportPage` protocol, `extra_pages`, public helpers/colors |
| `tests/tracker/test_metrics.py` | weight-history tests |
| `tests/tracker/test_report.py` | extra-pages tests |

**trading-strategies** (new branch `feature/factor-risk-model`)

| File | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | bump the `backtester` pin |
| `src/trading_strategies/risk/__init__.py` | package marker |
| `src/trading_strategies/risk/config.py` | `FactorModelConfig`, `MomentumConfig` |
| `src/trading_strategies/risk/covariance.py` | `EwmCovariance`, `FloatArray` |
| `src/trading_strategies/risk/factors.py` | `FactorDefinition`, `load_factors`, `factor_weight_matrix`, `residualize` |
| `src/trading_strategies/risk/momentum.py` | `MomentumSignal`, `momentum_exposures` |
| `src/trading_strategies/risk/model.py` | `RiskModelSnapshot`, `FactorRiskModel`, `MOMENTUM`, `SPECIFIC` |
| `src/trading_strategies/risk/attribution.py` | `AttributionResult`, `run_attribution`, `iter_market_events`, `attribute_backtest`, `RESIDUAL` |
| `src/trading_strategies/risk/report_pages.py` | four `ReportPage` classes + `attribution_pages` |
| `src/trading_strategies/config.py` | `MacdBacktestConfig.factor_model` |
| `configs/factors.json` | factor definitions |
| `configs/backtest_macd.json`, `configs/backtest_macd_rescaled.json` | `factor_model` section |
| `scripts/run_backtest.py` | wire attribution into the report |
| `tests/risk/__init__.py` + one test file per module + `test_integration.py` | tests |

---

## Part A — `backtester`

Work in `~/GitHub/Projects/backtester`. Start with:

```bash
cd ~/GitHub/Projects/backtester
git checkout main && git pull
git checkout -b feature/weights-history-report-pages
uv sync --all-extras
```

### Task A1: Per-bar weight history on `PerformanceTracker`

**Files:**
- Modify: `src/backtester/tracker/metrics.py` (class `PerformanceTracker`, `__init__` and `track_market`, plus a new property and helper)
- Test: `tests/tracker/test_metrics.py`

**Interfaces:**
- Produces: `PerformanceTracker.weights_history -> list[tuple[datetime, dict[Ticker, float]]]`. One entry per recorded equity mark, with the same timestamps as `mark_to_market_history`. Entry `t` holds the weights of the holdings that earned bar `t`'s return.

- [ ] **Step 1: Write the failing tests** (append to `tests/tracker/test_metrics.py`)

```python
def test_weights_history_records_holdings_that_earned_each_bar() -> None:
    """Entry t = quantities held from close t-1 to close t, valued at close
    t-1 over the equity mark of t-1. A fill on bar 0 shows up from bar 1."""
    tracker = PerformanceTracker(portfolio=FakePortfolioView([1_000.0, 1_000.0, 1_100.0]))

    tracker.track_market(MarketEvent(timestamp=_ts(0), bars={"AAPL": Bar(close=100.0)}))
    tracker.track_fill(
        FillEvent(timestamp=_ts(0), ticker="AAPL", quantity=5, direction="BUY", fill_price=100.0)
    )
    tracker.track_market(MarketEvent(timestamp=_ts(1), bars={"AAPL": Bar(close=120.0)}))
    tracker.track_market(MarketEvent(timestamp=_ts(2), bars={"AAPL": Bar(close=130.0)}))

    assert tracker.weights_history == [
        (_ts(0), {}),
        (_ts(1), {"AAPL": 0.5}),
        (_ts(2), {"AAPL": 0.6}),
    ]


def test_weights_history_signs_short_positions_negative() -> None:
    tracker = PerformanceTracker(portfolio=FakePortfolioView([1_000.0, 1_000.0]))

    tracker.track_market(MarketEvent(timestamp=_ts(0), bars={"AAPL": Bar(close=100.0)}))
    tracker.track_fill(
        FillEvent(timestamp=_ts(0), ticker="AAPL", quantity=2, direction="SELL", fill_price=100.0)
    )
    tracker.track_market(MarketEvent(timestamp=_ts(1), bars={"AAPL": Bar(close=90.0)}))

    assert tracker.weights_history[-1] == (_ts(1), {"AAPL": -0.2})


def test_weights_history_values_a_missing_ticker_at_its_last_close() -> None:
    tracker = PerformanceTracker(portfolio=FakePortfolioView([1_000.0, 1_000.0, 1_000.0]))

    tracker.track_market(
        MarketEvent(timestamp=_ts(0), bars={"AAPL": Bar(close=100.0), "MSFT": Bar(close=50.0)})
    )
    tracker.track_fill(
        FillEvent(timestamp=_ts(0), ticker="MSFT", quantity=4, direction="BUY", fill_price=50.0)
    )
    tracker.track_market(MarketEvent(timestamp=_ts(1), bars={"AAPL": Bar(close=101.0)}))
    tracker.track_market(MarketEvent(timestamp=_ts(2), bars={"AAPL": Bar(close=102.0)}))

    assert tracker.weights_history[-1] == (_ts(2), {"MSFT": 0.2})


def test_weights_history_skips_bars_without_prices() -> None:
    tracker = PerformanceTracker(portfolio=FakePortfolioView([1_000.0]))

    tracker.track_market(MarketEvent(timestamp=_ts(0), bars={}))
    tracker.track_market(MarketEvent(timestamp=_ts(1), bars=LIVE_BARS))

    assert [ts for ts, _ in tracker.weights_history] == [_ts(1)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/tracker/test_metrics.py -k weights_history -v --no-cov`
Expected: FAIL with `AttributeError: 'PerformanceTracker' object has no attribute 'weights_history'`

- [ ] **Step 3: Implement**

In `PerformanceTracker.__init__`, add after `self._traded_notional = 0.0`:

```python
        self._weights_history: list[tuple[datetime, dict[Ticker, float]]] = []
        self._last_close: dict[Ticker, float] = {}
```

In `track_market`, replace the body after the `if not event.bars:` early return with:

```python
        equity = self._portfolio.mark_to_market()
        # Weights first: they read the *previous* equity mark and closes.
        self._weights_history.append((event.timestamp, self._held_weights()))
        self._mark_to_market_history.append((event.timestamp, equity))
        self._last_close.update({ticker: bar.close for ticker, bar in event.bars.items()})
        logger.debug("%s: equity=%.2f", event.timestamp, equity)
        if self._open_lots:
            self._bars_in_market += 1
```

Add after `track_market`:

```python
    def _held_weights(self) -> dict[Ticker, float]:
        """Weights of the holdings that earn the bar being marked: quantities
        from the lot ledger (every fill up to the previous bar, since the engine
        tracks a bar before its strategy trades on it), valued at the previous
        close over the previous equity mark. Empty on the first mark. A ticker
        missing from the previous bar keeps its last seen close."""
        if not self._mark_to_market_history:
            return {}
        previous_equity = self._mark_to_market_history[-1][1]
        if previous_equity <= 0:
            return {}
        return {
            ticker: lot.signed_qty * self._last_close[ticker] / previous_equity
            for ticker, lot in self._open_lots.items()
            if ticker in self._last_close
        }
```

Add next to the `mark_to_market_history` property:

```python
    @property
    def weights_history(self) -> list[tuple[datetime, dict[Ticker, float]]]:
        return [(timestamp, dict(weights)) for timestamp, weights in self._weights_history]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/tracker/test_metrics.py -v --no-cov`
Expected: all PASS (new and existing).

- [ ] **Step 5: Document**

Run `grep -n "mark_to_market_history" ARCHITECTURE.md README.md`. Wherever the tracker outputs are listed, add one sentence: "`weights_history` — per bar, the weights of the holdings that earned that bar's return (valued at the previous close over the previous equity mark)."

- [ ] **Step 6: Commit**

```bash
git add src/backtester/tracker/metrics.py tests/tracker/test_metrics.py ARCHITECTURE.md README.md
git commit -m "Record per-bar holding weights on PerformanceTracker"
```

### Task A2: Public report-extension API

**Files:**
- Modify: `src/backtester/tracker/report.py`
- Test: `tests/tracker/test_report.py`

**Interfaces:**
- Produces (all importable from `backtester.tracker.report`):
  - `class ReportPage(Protocol): def render(self, pdf: PdfPages) -> None`
  - `save_report(..., *, cost_sweep=None, extra_pages: Sequence[ReportPage] = (), config)`. Extra pages render after the monthly/cost pages and before the config page.
  - `new_page(title: str) -> Figure`, `save_page(pdf: PdfPages, fig: Figure) -> None`
  - `style_table(ax: Axes, cell_text: list[list[str]], col_labels: list[str], **kwargs: object) -> None`
  - `draw_heatmap(...)` (same signature as today's `_draw_heatmap`)
  - colors `SURFACE`, `INK`, `MUTED`, `GRID`, `NEGATIVE`

- [ ] **Step 1: Write the failing tests** (append to `tests/tracker/test_report.py`)

```python
import pytest
from matplotlib.backends.backend_pdf import PdfPages

from backtester.tracker import report
from backtester.tracker.report import new_page, save_page


class _RecordingPage:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def render(self, pdf: PdfPages) -> None:
        self._calls.append("extra")
        save_page(pdf, new_page("Extra"))


def test_extra_pages_render_before_the_config_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics, trade_metrics, histories, config = _sample_inputs()
    calls: list[str] = []
    original = report._add_config_page

    def _spy(pdf: PdfPages, config: report.ReportConfig) -> None:
        calls.append("config")
        original(pdf, config)

    monkeypatch.setattr(report, "_add_config_page", _spy)

    path = save_report(
        output_dir=tmp_path,
        histories=histories,
        metrics=metrics,
        trade_metrics=trade_metrics,
        monthly_tables={label: monthly_returns_table(h) for label, h in histories.items()},
        correlation=strategy_correlation_matrix(histories),
        extra_pages=[_RecordingPage(calls), _RecordingPage(calls)],
        config=config,
    )

    assert calls == ["extra", "extra", "config"]
    assert path.stat().st_size > 0


def test_public_page_helpers_and_colors_are_exported() -> None:
    for name in (
        "ReportPage",
        "new_page",
        "save_page",
        "style_table",
        "draw_heatmap",
        "SURFACE",
        "INK",
        "MUTED",
        "GRID",
        "NEGATIVE",
    ):
        assert hasattr(report, name), name
```

Put the new imports at the top of the file with the existing imports (ruff `I` sorts them).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/tracker/test_report.py -v --no-cov`
Expected: FAIL (`ImportError: cannot import name 'new_page'`)

- [ ] **Step 3: Rename the private helpers and colors to public names**

```bash
perl -pi -e 's/\b_new_page\b/new_page/g; s/\b_save_page\b/save_page/g; s/\b_style_table\b/style_table/g; s/\b_draw_heatmap\b/draw_heatmap/g; s/\b_SURFACE\b/SURFACE/g; s/\b_INK\b/INK/g; s/\b_MUTED\b/MUTED/g; s/\b_GRID\b/GRID/g; s/\b_NEGATIVE\b/NEGATIVE/g' src/backtester/tracker/report.py
grep -rnwE "_new_page|_save_page|_style_table|_draw_heatmap|_SURFACE|_INK|_MUTED|_GRID|_NEGATIVE" src tests
```

Expected: the `grep` prints nothing. Leave `_GOOD_TINT`, `_CRITICAL_TINT` and `_DIVERGING` private.

- [ ] **Step 4: Add the protocol and the `extra_pages` parameter**

Below `class ReportConfig(Protocol)`, add:

```python
class ReportPage(Protocol):
    """A caller-supplied page (or pages) for ``save_report``. ``render`` draws
    with the public helpers in this module (``new_page``, ``style_table``,
    ``draw_heatmap``, the theme colors) and writes with ``save_page``. It may
    write nothing when it has nothing to show. This keeps ``backtester``
    agnostic of what downstream reports add (e.g. factor attribution)."""

    def render(self, pdf: PdfPages) -> None: ...
```

Change `save_report`:

```python
def save_report(
    output_dir: Path,
    histories: Mapping[str, Sequence[tuple[datetime, float]]],
    metrics: Mapping[str, PerformanceMetrics],
    trade_metrics: Mapping[str, TradeMetrics],
    monthly_tables: Mapping[str, pd.DataFrame],
    correlation: pd.DataFrame,
    *,
    cost_sweep: Sequence[CostPoint] | None = None,
    extra_pages: Sequence[ReportPage] = (),
    config: ReportConfig,
) -> Path:
    path = _next_report_path(output_dir, stem=f"{_slugify(config.name)}_report")
    with PdfPages(path) as pdf:
        _add_overview_page(
            pdf,
            config.name,
            histories,
            metrics,
            trade_metrics,
            correlation,
            png_path=path.with_name(f"{path.stem}_overview.png"),
        )
        _add_monthly_page(pdf, monthly_tables)
        if cost_sweep is not None:
            _add_cost_sweep_page(pdf, cost_sweep)
        for page in extra_pages:
            page.render(pdf)
        _add_config_page(pdf, config)
    return path
```

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest && uv run mypy . && uv run ruff check . && uv run ruff format --check .`
Expected: all green, coverage ≥ 95%.

- [ ] **Step 6: Commit, push, open a PR**

```bash
git add src/backtester/tracker/report.py tests/tracker/test_report.py
git commit -m "Add a public report-page API and extra_pages to save_report"
git push -u origin feature/weights-history-report-pages
gh pr create --title "Weight history + report-page extension API" --body "Adds PerformanceTracker.weights_history and a public ReportPage/extra_pages API for downstream report pages (factor attribution in trading-strategies)."
git rev-parse HEAD   # note this SHA for Task B1
```

Ask the user before pushing if they have not already approved the push.

---

## Part B — `trading-strategies`

Work in `~/GitHub/Projects/trading-strategies`:

```bash
cd ~/GitHub/Projects/trading-strategies
git checkout main && git pull
git checkout -b feature/factor-risk-model
```

### Task B1: Bump the `backtester` pin

**Files:** Modify `pyproject.toml` (`[tool.uv.sources]`), `uv.lock`

- [ ] **Step 1:** In `pyproject.toml`, set `rev` to the SHA from Task A2 Step 6:

```toml
backtester = { git = "https://github.com/KaitoFuss/backtester", rev = "<SHA from A2>" }
```

- [ ] **Step 2:** Run `uv lock --upgrade-package backtester && uv sync --all-extras`
- [ ] **Step 3:** Run `uv run python -c "from backtester.tracker.report import ReportPage, new_page; from backtester.tracker.metrics import PerformanceTracker; assert hasattr(PerformanceTracker, 'weights_history')"`
  - Expected: no output, exit 0.
- [ ] **Step 4:** Run `uv run pytest && uv run mypy .`
  - Expected: green. The existing code does not use the renamed private helpers.
- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Bump backtester for weights_history and the report-page API"
```

After the backtester PR is merged, re-pin to the merge commit in the same way. Do this in the final PR of this plan (Task B10).

### Task B2: Factor-model config

**Files:**
- Create: `src/trading_strategies/risk/__init__.py` (empty), `src/trading_strategies/risk/config.py`, `tests/risk/__init__.py` (empty), `tests/risk/test_config.py`
- Modify: `src/trading_strategies/config.py`

**Interfaces:**
- Produces:
  - `MomentumConfig(lookback: int = 252, skip: int = 21)`
  - `FactorModelConfig(factors_file: str, granular: bool = False, vol_span: int = 60, corr_span: int = 250, corr_min_periods: int = 250, residual_floor: float = 0.10, momentum: MomentumConfig = MomentumConfig())`
  - `FactorModelConfig.from_dict(raw: dict[str, Any]) -> FactorModelConfig`
  - `MacdBacktestConfig.factor_model: FactorModelConfig | None = None`

- [ ] **Step 1: Write the failing tests** (`tests/risk/test_config.py`)

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.risk.config import FactorModelConfig, MomentumConfig


def test_from_dict_applies_defaults() -> None:
    config = FactorModelConfig.from_dict({"factors_file": "configs/factors.json"})

    assert config == FactorModelConfig(factors_file="configs/factors.json")
    assert config.momentum == MomentumConfig(lookback=252, skip=21)
    assert config.residual_floor == 0.10


def test_from_dict_parses_nested_momentum() -> None:
    config = FactorModelConfig.from_dict(
        {"factors_file": "f.json", "granular": True, "momentum": {"lookback": 126, "skip": 5}}
    )

    assert config.granular is True
    assert config.momentum == MomentumConfig(lookback=126, skip=5)


def _write(path: Path, extra: dict[str, object]) -> Path:
    path.write_text(json.dumps({"name": "X", "data": "d.parquet", "tickers": ["SPY"], **extra}))
    return path


def test_backtest_config_without_factor_model_is_none(tmp_path: Path) -> None:
    config = MacdBacktestConfig.from_json(_write(tmp_path / "c.json", {}))

    assert config.factor_model is None


def test_backtest_config_parses_factor_model(tmp_path: Path) -> None:
    config = MacdBacktestConfig.from_json(
        _write(tmp_path / "c.json", {"factor_model": {"factors_file": "f.json", "corr_span": 120}})
    )

    assert config.factor_model == FactorModelConfig(factors_file="f.json", corr_span=120)


def test_backtest_config_rejects_unknown_factor_model_field(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid config"):
        MacdBacktestConfig.from_json(
            _write(tmp_path / "c.json", {"factor_model": {"factors_file": "f.json", "bogus": 1}})
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/risk/test_config.py -v --no-cov`
Expected: FAIL (`ModuleNotFoundError: trading_strategies.risk`)

- [ ] **Step 3: Implement** (`src/trading_strategies/risk/config.py`)

```python
"""Config for the factor risk model (see
docs/superpowers/specs/2026-09-22-factor-risk-model-design.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self


@dataclass(frozen=True)
class MomentumConfig:
    lookback: int = 252
    skip: int = 21


@dataclass(frozen=True)
class FactorModelConfig:
    factors_file: str
    granular: bool = False
    vol_span: int = 60
    corr_span: int = 250
    corr_min_periods: int = 250
    residual_floor: float = 0.10
    momentum: MomentumConfig = field(default_factory=MomentumConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        fields = dict(raw)
        momentum = MomentumConfig(**fields.pop("momentum", {}))
        return cls(momentum=momentum, **fields)
```

In `src/trading_strategies/config.py`:
- Add `from trading_strategies.risk.config import FactorModelConfig`.
- Add the field `factor_model: FactorModelConfig | None = None` after `output_dir`.
- Replace `from_json`:

```python
    @classmethod
    def from_json(cls, path: Path) -> Self:
        try:
            raw = json.loads(path.read_text())
            if "factor_model" in raw:
                raw["factor_model"] = FactorModelConfig.from_dict(raw["factor_model"])
            return cls(**raw)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"invalid config at {path}: {error}") from error
```

Add one sentence to the module docstring: `factor_model` is optional, and when it is set the report gains factor-attribution pages.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/risk/test_config.py tests/test_config.py -v --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/risk src/trading_strategies/config.py tests/risk
git commit -m "Add the factor-model config section"
```

### Task B3: Streaming pairwise-complete EWM covariance

**Files:**
- Create: `src/trading_strategies/risk/covariance.py`, `tests/risk/test_covariance.py`

**Interfaces:**
- Produces:
  - `FloatArray = npt.NDArray[np.float64]`
  - `EwmCovariance(tickers: Sequence[Ticker], *, span: int, min_periods: int)`
  - `.tickers: tuple[Ticker, ...]`
  - `.update(values: Mapping[Ticker, float]) -> None`. Missing, NaN and unknown tickers are skipped.
  - `.covariance() -> FloatArray` (N×N, NaN where a pair has fewer than `min_periods` joint observations).
- Semantics: pandas `DataFrame.ewm(span=..., adjust=True).cov()` (bias-corrected). A pair's accumulators start at zero and only receive joint observations. Every accumulator decays every step. For a column that starts late, this equals pandas on the rows since it started.

- [ ] **Step 1: Write the failing tests** (`tests/risk/test_covariance.py`)

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_strategies.risk.covariance import EwmCovariance


def _frame(n: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, (n, 1))
    data = base + rng.normal(0, 0.01, (n, 3))
    return pd.DataFrame(data, columns=["A", "B", "C"])


def _feed(cov: EwmCovariance, frame: pd.DataFrame) -> None:
    for _, row in frame.iterrows():
        cov.update({str(k): float(v) for k, v in row.items()})


def test_matches_pandas_ewm_cov_on_complete_data() -> None:
    frame = _frame()
    cov = EwmCovariance(["A", "B", "C"], span=20, min_periods=2)
    _feed(cov, frame)

    expected = frame.ewm(span=20).cov().xs(frame.index[-1], level=0)
    np.testing.assert_allclose(cov.covariance(), expected.to_numpy(), rtol=1e-10)


def test_late_listed_column_matches_pandas_since_listing() -> None:
    frame = _frame()
    frame.loc[:39, "C"] = np.nan
    cov = EwmCovariance(["A", "B", "C"], span=20, min_periods=2)
    _feed(cov, frame)

    full = frame[["A", "B"]].ewm(span=20).cov().xs(frame.index[-1], level=0)
    since = frame.iloc[40:].ewm(span=20).cov().xs(frame.index[-1], level=0)
    result = cov.covariance()
    assert result[0, 1] == pytest.approx(full.loc["A", "B"], rel=1e-10)
    assert result[0, 2] == pytest.approx(since.loc["A", "C"], rel=1e-10)
    assert result[2, 2] == pytest.approx(since.loc["C", "C"], rel=1e-10)


def test_nan_until_min_periods_joint_observations() -> None:
    cov = EwmCovariance(["A", "B"], span=10, min_periods=5)
    for i in range(4):
        cov.update({"A": 0.01 * i, "B": -0.02 * i})
    assert np.isnan(cov.covariance()).all()

    cov.update({"A": 0.05, "B": 0.01})
    assert np.isfinite(cov.covariance()).all()


def test_unknown_and_missing_tickers_are_ignored() -> None:
    cov = EwmCovariance(["A", "B"], span=10, min_periods=2)
    cov.update({"A": 0.01, "ZZZ": 1.0})
    cov.update({"A": 0.02, "B": float("nan")})
    cov.update({"A": 0.03})

    result = cov.covariance()
    assert np.isfinite(result[0, 0])
    assert np.isnan(result[1, 1])
    assert np.isnan(result[0, 1])


def test_min_periods_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_periods"):
        EwmCovariance(["A"], span=10, min_periods=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/risk/test_covariance.py -v --no-cov`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement** (`src/trading_strategies/risk/covariance.py`)

```python
"""Streaming, pairwise-complete EWM covariance -- the one estimator the factor
risk model is built from (asset vols with a short span, the correlation
matrix with a long span).

Same math as ``DataFrame.ewm(span=..., adjust=True).cov()`` (bias-corrected,
``var = (Sw*Sxy - Sx*Sy) / (Sw^2 - Sw2)``; see ``utils.streaming.EwmMoments``)
kept per pair: a pair's accumulators only receive bars where both values are
present, and every accumulator decays every bar. A late-listed ticker
therefore enters the matrix exactly as pandas would on the rows since it
started trading.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from backtester.core.events import Ticker

FloatArray = npt.NDArray[np.float64]


class EwmCovariance:
    def __init__(self, tickers: Sequence[Ticker], *, span: int, min_periods: int) -> None:
        if min_periods < 2:
            raise ValueError("min_periods must be >= 2")
        n = len(tickers)
        self.tickers: tuple[Ticker, ...] = tuple(tickers)
        self._index = {ticker: i for i, ticker in enumerate(self.tickers)}
        self._decay = 1.0 - 2.0 / (span + 1)
        self._min_periods = min_periods
        self._sw = np.zeros((n, n))
        self._sw2 = np.zeros((n, n))
        self._sx = np.zeros((n, n))  # sx[i, j] = sum of w * x_i over joint (i, j) bars
        self._sxy = np.zeros((n, n))
        self._count = np.zeros((n, n), dtype=np.int64)

    def update(self, values: Mapping[Ticker, float]) -> None:
        x = np.full(len(self.tickers), np.nan)
        for ticker, value in values.items():
            index = self._index.get(ticker)
            if index is not None:
                x[index] = value
        valid = np.isfinite(x)
        joint = np.outer(valid, valid).astype(np.float64)
        x0 = np.where(valid, x, 0.0)
        decay = self._decay
        self._sw = decay * self._sw + joint
        self._sw2 = decay * decay * self._sw2 + joint
        self._sx = decay * self._sx + joint * x0[:, None]
        self._sxy = decay * self._sxy + np.outer(x0, x0)
        self._count += joint.astype(np.int64)

    def covariance(self) -> FloatArray:
        denom = self._sw**2 - self._sw2
        ready = (self._count >= self._min_periods) & (denom > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = (self._sw * self._sxy - self._sx * self._sx.T) / denom
        return np.where(ready, cov, np.nan)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/risk/test_covariance.py -v --no-cov`
Expected: PASS. If the late-listing test fails only on the `(A, C)` entry, check that `joint * x0[:, None]` (not `x0`) feeds `_sx`.

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/risk/covariance.py tests/risk/test_covariance.py
git commit -m "Add a streaming pairwise-complete EWM covariance"
```

### Task B4: Factor definitions, weight matrix, residualization

**Files:**
- Create: `src/trading_strategies/risk/factors.py`, `configs/factors.json`, `tests/risk/test_factors.py`

**Interfaces:**
- Consumes: `FloatArray` (B3)
- Produces:
  - `FactorGroup = Literal["core", "secondary", "granular"]`
  - `FactorDefinition(name: str, level: int, group: FactorGroup, weights: Mapping[Ticker, float])` (frozen)
  - `load_factors(path: Path, *, granular: bool) -> list[FactorDefinition]` (stable-sorted by level)
  - `factor_weight_matrix(factors: Sequence[FactorDefinition], tickers: Sequence[Ticker], available: Collection[Ticker]) -> FloatArray` (N×K; inactive factor = zero column)
  - `residualize(weights: FloatArray, levels: Sequence[int], covariance: FloatArray) -> FloatArray`

- [ ] **Step 1: Create `configs/factors.json`**

```json
{
    "factors": [
        {"name": "Equity", "level": 1, "group": "core", "weights": {"SPY": 0.6, "EFA": 0.3, "EEM": 0.1}},
        {"name": "Rates", "level": 1, "group": "core", "weights": {"IEF": 1.0}},
        {"name": "Credit", "level": 2, "group": "core", "weights": {"HYG": 0.5, "LQD": 0.5}},
        {"name": "Commodities", "level": 2, "group": "core", "weights": {"DBC": 1.0}},
        {"name": "Emerging Markets", "level": 3, "group": "secondary", "weights": {"EEM": 1.0}},
        {"name": "Foreign Currency", "level": 3, "group": "secondary", "weights": {"UUP": -1.0}},
        {"name": "Local Inflation", "level": 3, "group": "secondary", "weights": {"TIP": 1.0}},
        {"name": "US vs Intl", "level": 3, "group": "secondary", "weights": {"SPY": 1.0, "EFA": -1.0}},
        {"name": "Size", "level": 4, "group": "granular", "weights": {"IWM": 1.0, "SPY": -1.0}},
        {"name": "Tech", "level": 4, "group": "granular", "weights": {"QQQ": 1.0, "SPY": -1.0}},
        {"name": "Cyclical vs Defensive", "level": 4, "group": "granular", "weights": {
            "XLI": 0.25, "XLB": 0.25, "XLY": 0.25, "XLF": 0.25,
            "XLP": -0.3333333333, "XLU": -0.3333333333, "XLV": -0.3333333334}},
        {"name": "Curve", "level": 4, "group": "granular", "weights": {"TLT": 1.0, "SHY": -1.0}},
        {"name": "Energy", "level": 4, "group": "granular", "weights": {
            "XLE": 0.3333333333, "OIH": 0.3333333333, "USO": 0.3333333334}},
        {"name": "Precious Metals", "level": 4, "group": "granular", "weights": {"GLD": 1.0}},
        {"name": "Yen", "level": 4, "group": "granular", "weights": {"FXY": 1.0}}
    ]
}
```

- [ ] **Step 2: Write the failing tests** (`tests/risk/test_factors.py`)

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trading_strategies.risk.factors import (
    FactorDefinition,
    factor_weight_matrix,
    load_factors,
    residualize,
)

REPO_FACTORS = Path(__file__).resolve().parents[2] / "configs" / "factors.json"


def test_repo_factor_file_has_eight_macro_and_seven_granular() -> None:
    assert len(load_factors(REPO_FACTORS, granular=False)) == 8
    assert len(load_factors(REPO_FACTORS, granular=True)) == 15


def test_factors_are_sorted_by_level_stably(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(
        json.dumps(
            {
                "factors": [
                    {"name": "B", "level": 2, "group": "core", "weights": {"X": 1.0}},
                    {"name": "A1", "level": 1, "group": "core", "weights": {"Y": 1.0}},
                    {"name": "A2", "level": 1, "group": "core", "weights": {"Z": 1.0}},
                ]
            }
        )
    )

    assert [f.name for f in load_factors(path, granular=False)] == ["A1", "A2", "B"]


@pytest.mark.parametrize(
    ("factors", "message"),
    [
        ([{"name": "A", "level": 1, "group": "bogus", "weights": {"X": 1.0}}], "group"),
        ([{"name": "A", "level": 1, "group": "core", "weights": {}}], "weights"),
        (
            [
                {"name": "A", "level": 1, "group": "core", "weights": {"X": 1.0}},
                {"name": "A", "level": 2, "group": "core", "weights": {"Y": 1.0}},
            ],
            "duplicate",
        ),
    ],
)
def test_invalid_factor_files_are_rejected(
    tmp_path: Path, factors: list[dict[str, object]], message: str
) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"factors": factors}))
    with pytest.raises(ValueError, match=message):
        load_factors(path, granular=True)


def _factor(name: str, weights: dict[str, float], level: int = 1) -> FactorDefinition:
    return FactorDefinition(name=name, level=level, group="core", weights=weights)


def test_missing_long_ticker_renormalizes_the_leg() -> None:
    equity = _factor("Equity", {"SPY": 0.6, "EFA": 0.3, "EEM": 0.1})
    matrix = factor_weight_matrix([equity], ["SPY", "EFA"], available={"SPY", "EFA"})

    np.testing.assert_allclose(matrix[:, 0], [0.6 / 0.9, 0.3 / 0.9])


def test_spread_with_a_missing_leg_is_inactive() -> None:
    spread = _factor("US vs Intl", {"SPY": 1.0, "EFA": -1.0})
    matrix = factor_weight_matrix([spread], ["SPY", "EFA"], available={"SPY"})

    assert not matrix.any()


def test_short_only_factor_keeps_its_sign() -> None:
    fx = _factor("Foreign Currency", {"UUP": -1.0})
    matrix = factor_weight_matrix([fx], ["UUP"], available={"UUP"})

    np.testing.assert_allclose(matrix[:, 0], [-1.0])


def test_later_level_is_orthogonal_to_earlier_levels() -> None:
    covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
    weights = np.eye(2)

    residual = residualize(weights, [1, 2], covariance)

    assert residual[:, 0] @ covariance @ residual[:, 1] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(residual[:, 1], [-0.5, 1.0])


def test_same_level_factors_are_not_residualized() -> None:
    covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
    weights = np.eye(2)

    np.testing.assert_allclose(residualize(weights, [1, 1], covariance), weights)


def test_inactive_columns_are_skipped() -> None:
    covariance = np.eye(2)
    weights = np.array([[0.0, 1.0], [0.0, 0.0]])

    np.testing.assert_allclose(residualize(weights, [1, 2], covariance), weights)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/risk/test_factors.py -v --no-cov`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 4: Implement** (`src/trading_strategies/risk/factors.py`)

```python
"""Macro factor definitions (Two Sigma Factor Lens style) and their
hierarchical residualization. A factor is a fixed portfolio of universe ETFs;
a factor at level n is residualized against every active factor at an earlier
level, so it stays a portfolio of ETFs and carries only the risk the levels
above it do not explain."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from backtester.core.events import Ticker

from trading_strategies.risk.covariance import FloatArray

FactorGroup = Literal["core", "secondary", "granular"]
_GROUPS: frozenset[str] = frozenset({"core", "secondary", "granular"})


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    level: int
    group: FactorGroup
    weights: Mapping[Ticker, float]


def load_factors(path: Path, *, granular: bool) -> list[FactorDefinition]:
    entries = json.loads(path.read_text())["factors"]
    factors: list[FactorDefinition] = []
    seen: set[str] = set()
    for entry in entries:
        name = str(entry["name"])
        if entry["group"] not in _GROUPS:
            raise ValueError(f"factor {name!r}: unknown group {entry['group']!r}")
        if not entry["weights"]:
            raise ValueError(f"factor {name!r}: empty weights")
        if name in seen:
            raise ValueError(f"duplicate factor name {name!r}")
        seen.add(name)
        factors.append(
            FactorDefinition(
                name=name,
                level=int(entry["level"]),
                group=cast(FactorGroup, entry["group"]),
                weights={str(t): float(w) for t, w in entry["weights"].items()},
            )
        )
    if not granular:
        factors = [factor for factor in factors if factor.group != "granular"]
    return sorted(factors, key=lambda factor: factor.level)


def factor_weight_matrix(
    factors: Sequence[FactorDefinition],
    tickers: Sequence[Ticker],
    available: Collection[Ticker],
) -> FloatArray:
    """N x K raw factor weights over ``tickers``. The long and short legs are
    each renormalized over their available tickers to their original gross, so
    a missing EEM does not shrink the Equity factor. A leg with no available
    ticker makes the factor inactive (a zero column): a spread missing a leg is
    not the factor any more."""
    index = {ticker: i for i, ticker in enumerate(tickers)}
    matrix = np.zeros((len(tickers), len(factors)))
    for k, factor in enumerate(factors):
        column = np.zeros(len(tickers))
        for sign in (1.0, -1.0):
            leg = {t: w for t, w in factor.weights.items() if w * sign > 0}
            if not leg:
                continue
            live = {t: w for t, w in leg.items() if t in available and t in index}
            if not live:
                break
            scale = sum(leg.values()) / sum(live.values())
            for ticker, weight in live.items():
                column[index[ticker]] = weight * scale
        else:
            matrix[:, k] = column
    return matrix


def residualize(weights: FloatArray, levels: Sequence[int], covariance: FloatArray) -> FloatArray:
    """Residualize each factor portfolio, in column order, against the already
    residualized active factors at strictly earlier levels (GLS projection
    under ``covariance``). Same-level factors stay correlated; ``F`` carries
    that."""
    residual = weights.copy()
    active = np.any(weights != 0, axis=0)
    for k in range(weights.shape[1]):
        if not active[k]:
            continue
        earlier = [j for j in range(k) if active[j] and levels[j] < levels[k]]
        if not earlier:
            continue
        basis = residual[:, earlier]
        beta = np.linalg.lstsq(
            basis.T @ covariance @ basis, basis.T @ covariance @ weights[:, k], rcond=None
        )[0]
        residual[:, k] = weights[:, k] - basis @ beta
    return residual
```

Note the `for ... else` in `factor_weight_matrix`: the `else` runs only if no leg hit `break`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/risk/test_factors.py -v --no-cov`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_strategies/risk/factors.py configs/factors.json tests/risk/test_factors.py
git commit -m "Add hierarchical macro factor definitions"
```

### Task B5: Momentum signal and exposures

**Files:**
- Create: `src/trading_strategies/risk/momentum.py`, `tests/risk/test_momentum.py`

**Interfaces:**
- Produces:
  - `MomentumSignal(tickers: Sequence[Ticker], *, lookback: int, skip: int)`
  - `.update(closes: Mapping[Ticker, float]) -> None`
  - `.raw(ticker: Ticker) -> float | None` = `log(close[t-skip] / close[t-lookback])`, `None` until `lookback + 1` closes.
  - `momentum_exposures(raw: Mapping[Ticker, float], sigma: Mapping[Ticker, float]) -> dict[Ticker, float]`: vol-scale, clip at median ± 3 robust sigmas (`1.4826 · MAD`), then z-score. Returns `{}` with fewer than 3 names or zero dispersion.

- [ ] **Step 1: Write the failing tests** (`tests/risk/test_momentum.py`)

```python
from __future__ import annotations

import math

import numpy as np
import pytest

from trading_strategies.risk.momentum import MomentumSignal, momentum_exposures


def test_raw_is_none_until_lookback_plus_one_closes() -> None:
    signal = MomentumSignal(["A"], lookback=3, skip=1)
    for close in (100.0, 101.0, 102.0):
        signal.update({"A": close})
    assert signal.raw("A") is None

    signal.update({"A": 104.0})
    assert signal.raw("A") == pytest.approx(math.log(102.0 / 100.0))


def test_raw_rolls_forward_and_ignores_unknown_tickers() -> None:
    signal = MomentumSignal(["A"], lookback=2, skip=0)
    for close in (100.0, 110.0, 121.0, 133.1):
        signal.update({"A": close, "ZZZ": 1.0})

    assert signal.raw("A") == pytest.approx(math.log(133.1 / 110.0))


def test_exposures_are_standardized() -> None:
    raw = {f"T{i}": 0.01 * i for i in range(10)}
    sigma = dict.fromkeys(raw, 0.01)

    values = np.array(list(momentum_exposures(raw, sigma).values()))

    assert values.mean() == pytest.approx(0.0, abs=1e-12)
    assert values.std() == pytest.approx(1.0)


def test_single_outlier_is_winsorized() -> None:
    """Without the robust clip, one outlier among 101 names sits at z ~ 10."""
    rng = np.random.default_rng(0)
    raw = {f"T{i}": float(v) for i, v in enumerate(rng.normal(0, 1, 100))} | {"BIG": 1_000.0}
    sigma = dict.fromkeys(raw, 1.0)

    exposures = momentum_exposures(raw, sigma)

    assert exposures["BIG"] < 4.0
    assert exposures["BIG"] == max(exposures.values())


@pytest.mark.parametrize(
    "raw",
    [{"A": 0.1, "B": 0.2}, {"A": 0.1, "B": 0.1, "C": 0.1}],
)
def test_too_few_names_or_no_dispersion_gives_no_exposures(raw: dict[str, float]) -> None:
    assert momentum_exposures(raw, dict.fromkeys(raw, 0.01)) == {}


def test_names_without_a_positive_vol_are_dropped() -> None:
    raw = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}
    sigma = {"A": 0.01, "B": 0.01, "C": 0.01, "D": 0.0}

    assert set(momentum_exposures(raw, sigma)) == {"A", "B", "C"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/risk/test_momentum.py -v --no-cov`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement** (`src/trading_strategies/risk/momentum.py`)

```python
"""Characteristic-based momentum for the factor risk model: an asset's
exposure is its vol-scaled 12-1 month return, standardized across the assets
live on that bar (Barra-style), not a time-series beta."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence

import numpy as np
from backtester.core.events import Ticker

from trading_strategies.risk.covariance import FloatArray

_WINSOR = 3.0
_MAD_TO_SIGMA = 1.4826
_MIN_NAMES = 3


class MomentumSignal:
    def __init__(self, tickers: Sequence[Ticker], *, lookback: int, skip: int) -> None:
        self._lookback = lookback
        self._skip = skip
        self._closes: dict[Ticker, deque[float]] = {
            ticker: deque(maxlen=lookback + 1) for ticker in tickers
        }

    def update(self, closes: Mapping[Ticker, float]) -> None:
        for ticker, close in closes.items():
            history = self._closes.get(ticker)
            if history is not None:
                history.append(close)

    def raw(self, ticker: Ticker) -> float | None:
        history = self._closes[ticker]
        if len(history) < self._lookback + 1:
            return None
        return math.log(history[-1 - self._skip] / history[0])


def _zscore(values: FloatArray) -> FloatArray | None:
    std = float(values.std())
    if std == 0.0:
        return None
    return (values - values.mean()) / std


def momentum_exposures(
    raw: Mapping[Ticker, float], sigma: Mapping[Ticker, float]
) -> dict[Ticker, float]:
    """Winsorize with a robust scale (median / MAD), then z-score. A plain
    z-score -> clip -> z-score does not tame a single outlier: re-standardizing
    a lone clipped point puts it straight back at the same z."""
    scaled = {t: raw[t] / sigma[t] for t in raw if sigma.get(t, 0.0) > 0}
    if len(scaled) < _MIN_NAMES:
        return {}
    values = np.array(list(scaled.values()))
    median = float(np.median(values))
    robust_sigma = _MAD_TO_SIGMA * float(np.median(np.abs(values - median)))
    if robust_sigma > 0:
        values = np.clip(values, median - _WINSOR * robust_sigma, median + _WINSOR * robust_sigma)
    z = _zscore(values)
    if z is None:
        return {}
    return dict(zip(scaled, z.tolist(), strict=True))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/risk/test_momentum.py -v --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/risk/momentum.py tests/risk/test_momentum.py
git commit -m "Add the momentum style factor exposures"
```

### Task B6: `FactorRiskModel` and `RiskModelSnapshot`

**Files:**
- Create: `src/trading_strategies/risk/model.py`, `tests/risk/test_model.py`

**Interfaces:**
- Consumes:
  - `FactorModelConfig`, `MomentumConfig` (B2)
  - `EwmCovariance`, `FloatArray` (B3)
  - `load_factors`, `factor_weight_matrix`, `residualize` (B4)
  - `MomentumSignal`, `momentum_exposures` (B5)
  - `EwmMoments` (`trading_strategies.utils.streaming`, existing)
- Produces:
  - `MOMENTUM = "Momentum"`, `SPECIFIC = "Specific"`
  - `RiskModelSnapshot` (frozen) with fields `tickers, factors, B, F, D, sigma, factor_weights, macro_betas, momentum, ready`
  - Snapshot methods:
    - `.empty()` (classmethod)
    - `.weight_vector(weights: Mapping[Ticker, float]) -> FloatArray`
    - `.covariance() -> FloatArray`
    - `.exposures(w) -> FloatArray`
    - `.portfolio_variance(w) -> float`
    - `.risk_decomposition(w) -> dict[str, float]` (keys: factors + `SPECIFIC`)
    - `.factor_returns(asset_returns: Mapping[Ticker, float]) -> FloatArray` (aligned with `factors`)
    - `.momentum_return(asset_returns) -> float | None`
  - `FactorRiskModel(tickers: Sequence[Ticker], config: FactorModelConfig)` with `.update(closes: Mapping[Ticker, float]) -> None` and `.snapshot -> RiskModelSnapshot`

- [ ] **Step 1: Write the failing tests** (`tests/risk/test_model.py`)

```python
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from trading_strategies.risk.config import FactorModelConfig, MomentumConfig
from trading_strategies.risk.model import MOMENTUM, SPECIFIC, FactorRiskModel, RiskModelSnapshot

REPO_FACTORS = Path(__file__).resolve().parents[2] / "configs" / "factors.json"


def _config(tmp_path: Path, **overrides: object) -> FactorModelConfig:
    factors_file = tmp_path / "factors.json"
    factors_file.write_text(
        json.dumps(
            {
                "factors": [
                    {"name": "Market", "level": 1, "group": "core", "weights": {"MKT": 1.0}},
                ]
            }
        )
    )
    fields: dict[str, object] = {
        "factors_file": str(factors_file),
        "vol_span": 20,
        "corr_span": 100,
        "corr_min_periods": 50,
        "momentum": MomentumConfig(lookback=20, skip=2),
    }
    fields.update(overrides)
    return FactorModelConfig(**fields)  # type: ignore[arg-type]


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


def test_not_ready_until_correlation_warm(tmp_path: Path) -> None:
    path = _simulate(60, {"A": 0.5, "B": -1.0})
    model = FactorRiskModel(["MKT", "A", "B"], _config(tmp_path))

    _run(model, path[:50])  # 49 returns < corr_min_periods=50
    assert model.snapshot.ready is False

    model.update(path[50])
    assert model.snapshot.ready is True


def test_recovers_known_betas(tmp_path: Path) -> None:
    path = _simulate(600, {"A": 0.5, "B": -1.0})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B"], _config(tmp_path)), path)

    market = snapshot.factors.index("Market")
    betas = dict(zip(snapshot.tickers, snapshot.B[:, market], strict=True))
    assert betas["MKT"] == pytest.approx(1.0, abs=0.2)
    assert betas["A"] == pytest.approx(0.5, abs=0.2)
    assert betas["B"] == pytest.approx(-1.0, abs=0.2)


def test_covariance_is_psd_and_floor_holds(tmp_path: Path) -> None:
    path = _simulate(300, {"A": 0.5, "B": -1.0})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B"], _config(tmp_path)), path)

    assert np.linalg.eigvalsh(snapshot.covariance()).min() > 0
    # MKT *is* the Market factor: without the floor its specific risk is ~0.
    mkt = snapshot.tickers.index("MKT")
    assert snapshot.D[mkt] >= 0.10 * snapshot.sigma[mkt] ** 2 * 0.99


def test_momentum_joins_once_its_variance_is_warm(tmp_path: Path) -> None:
    path = _simulate(300, {"A": 0.5, "B": -1.0, "C": 0.2})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B", "C"], _config(tmp_path)), path)

    assert snapshot.factors == ("Market", MOMENTUM)
    assert snapshot.momentum is not None
    assert snapshot.B.shape == (4, 2)


def test_risk_decomposition_sums_to_one(tmp_path: Path) -> None:
    path = _simulate(300, {"A": 0.5, "B": -1.0, "C": 0.2})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B", "C"], _config(tmp_path)), path)
    w = snapshot.weight_vector({"A": 0.5, "B": 0.3, "C": -0.2})

    shares = snapshot.risk_decomposition(w)

    assert set(shares) == {*snapshot.factors, SPECIFIC}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert snapshot.portfolio_variance(w) == pytest.approx(w @ snapshot.covariance() @ w)


def test_zero_portfolio_has_zero_risk_shares(tmp_path: Path) -> None:
    path = _simulate(120, {"A": 0.5, "B": -1.0})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B"], _config(tmp_path)), path)

    shares = snapshot.risk_decomposition(snapshot.weight_vector({}))

    assert all(value == 0.0 for value in shares.values())


def test_macro_factor_returns_are_residualized_portfolio_returns(tmp_path: Path) -> None:
    path = _simulate(120, {"A": 0.5, "B": -1.0})
    snapshot = _run(FactorRiskModel(["MKT", "A", "B"], _config(tmp_path)), path)
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
        _config(tmp_path, factors_file=str(REPO_FACTORS), granular=granular),
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/risk/test_model.py -v --no-cov`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement** (`src/trading_strategies/risk/model.py`)

```python
"""Hybrid factor risk model (spec:
docs/superpowers/specs/2026-09-22-factor-risk-model-design.md).

Macro factors are fixed ETF portfolios, residualized level by level, with
time-series betas projected from one long-span EWM covariance ``S``:
``F = W'^T S W'``, ``B = S W' F^-1``. Momentum is characteristic-based: the
exposure is the standardized 12-1 month return, and the factor return is a
cross-sectional regression of macro residual returns on it. Specific risk is
floored at ``residual_floor * S_ii``. ``B`` and ``D`` are then rescaled by
``sigma_short / sqrt(S_ii)``, so short-span vols ride on long-span
correlations. Everything is in daily units, and streaming/causal: the
snapshot after ``update(bar t)`` uses data up to and including bar t only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from backtester.core.events import Ticker

from trading_strategies.risk.config import FactorModelConfig
from trading_strategies.risk.covariance import EwmCovariance, FloatArray
from trading_strategies.risk.factors import factor_weight_matrix, load_factors, residualize
from trading_strategies.risk.momentum import MomentumSignal, momentum_exposures
from trading_strategies.utils.streaming import EwmMoments

MOMENTUM = "Momentum"
SPECIFIC = "Specific"
_EIGEN_FLOOR = 1e-12


@dataclass(frozen=True)
class RiskModelSnapshot:
    tickers: tuple[Ticker, ...]
    factors: tuple[str, ...]
    B: FloatArray  # N x K, vol-rescaled betas / exposures
    F: FloatArray  # K x K factor covariance
    D: FloatArray  # N specific variances
    sigma: FloatArray  # N short-span daily vols
    factor_weights: FloatArray  # N x K_macro residualized factor portfolios
    macro_betas: FloatArray  # N x K_macro long-span betas (for macro residuals)
    momentum: FloatArray | None  # N momentum exposures, even before Momentum joins F
    ready: bool

    @classmethod
    def empty(cls) -> RiskModelSnapshot:
        matrix = np.zeros((0, 0))
        vector = np.zeros(0)
        return cls((), (), matrix, matrix, vector, vector, matrix, matrix, None, False)

    def weight_vector(self, weights: Mapping[Ticker, float]) -> FloatArray:
        return np.array([weights.get(t, 0.0) for t in self.tickers], dtype=np.float64)

    def covariance(self) -> FloatArray:
        return self.B @ self.F @ self.B.T + np.diag(self.D)

    def exposures(self, w: FloatArray) -> FloatArray:
        return self.B.T @ w

    def portfolio_variance(self, w: FloatArray) -> float:
        x = self.exposures(w)
        return float(x @ self.F @ x + np.sum(self.D * w**2))

    def risk_decomposition(self, w: FloatArray) -> dict[str, float]:
        variance = self.portfolio_variance(w)
        if variance <= 0:
            return dict.fromkeys((*self.factors, SPECIFIC), 0.0)
        x = self.exposures(w)
        marginal = self.F @ x
        shares = {name: float(x[k] * marginal[k]) / variance for k, name in enumerate(self.factors)}
        shares[SPECIFIC] = float(np.sum(self.D * w**2)) / variance
        return shares

    def factor_returns(self, asset_returns: Mapping[Ticker, float]) -> FloatArray:
        r = self._return_vector(asset_returns)
        macro = self.factor_weights.T @ r
        if MOMENTUM not in self.factors:
            return macro
        return np.append(macro, self._momentum_return(r, macro))

    def momentum_return(self, asset_returns: Mapping[Ticker, float]) -> float | None:
        if self.momentum is None:
            return None
        r = self._return_vector(asset_returns)
        return self._momentum_return(r, self.factor_weights.T @ r)

    def _return_vector(self, asset_returns: Mapping[Ticker, float]) -> FloatArray:
        r = np.array([asset_returns.get(t, 0.0) for t in self.tickers], dtype=np.float64)
        return np.nan_to_num(r)

    def _momentum_return(self, r: FloatArray, macro: FloatArray) -> float:
        assert self.momentum is not None
        residual = r - self.macro_betas @ macro
        denom = float(self.momentum @ self.momentum)
        return float(self.momentum @ residual) / denom if denom > 0 else 0.0


class FactorRiskModel:
    def __init__(self, tickers: Sequence[Ticker], config: FactorModelConfig) -> None:
        self._tickers = tuple(tickers)
        self._known = frozenset(self._tickers)
        self._config = config
        self._definitions = load_factors(Path(config.factors_file), granular=config.granular)
        self._vol = EwmCovariance(self._tickers, span=config.vol_span, min_periods=config.vol_span)
        self._corr = EwmCovariance(
            self._tickers, span=config.corr_span, min_periods=config.corr_min_periods
        )
        self._momentum = MomentumSignal(
            self._tickers, lookback=config.momentum.lookback, skip=config.momentum.skip
        )
        self._momentum_var = EwmMoments(span=config.corr_span, min_periods=config.vol_span)
        self._last_close: dict[Ticker, float] = {}
        self._snapshot = RiskModelSnapshot.empty()

    @property
    def snapshot(self) -> RiskModelSnapshot:
        return self._snapshot

    def update(self, closes: Mapping[Ticker, float]) -> None:
        live = {t: c for t, c in closes.items() if t in self._known}
        returns = {
            t: math.log(c / self._last_close[t]) for t, c in live.items() if t in self._last_close
        }
        self._last_close.update(live)
        # Previous snapshot + this bar's returns: causal.
        momentum_return = self._snapshot.momentum_return(returns)
        if momentum_return is not None:
            self._momentum_var.update(momentum_return)
        self._vol.update(returns)
        self._corr.update(returns)
        self._momentum.update(live)
        self._snapshot = self._build()

    def _build(self) -> RiskModelSnapshot:
        corr = self._corr.covariance()
        vol = self._vol.covariance()
        universe = _complete_universe(corr, vol)
        if len(universe) < 2:
            return RiskModelSnapshot.empty()
        tickers = tuple(self._tickers[i] for i in universe)
        s = corr[np.ix_(universe, universe)]
        sigma = np.sqrt(np.diag(vol)[universe])

        raw_weights = factor_weight_matrix(self._definitions, tickers, available=set(tickers))
        active = np.flatnonzero(np.any(raw_weights != 0, axis=0))
        if active.size == 0:
            return RiskModelSnapshot.empty()
        definitions = [self._definitions[k] for k in active]
        factor_weights = residualize(raw_weights[:, active], [d.level for d in definitions], s)
        f_macro = _clip_psd(factor_weights.T @ s @ factor_weights)
        macro_betas = s @ factor_weights @ np.linalg.pinv(f_macro)
        names = [d.name for d in definitions]

        b, f = macro_betas, f_macro
        momentum = self._momentum_exposure(tickers, sigma)
        momentum_std = self._momentum_var.std
        if momentum is not None and momentum_std:
            k = len(names)
            b = np.column_stack([macro_betas, momentum])
            f = np.block(
                [
                    [f_macro, np.zeros((k, 1))],
                    [np.zeros((1, k)), np.array([[momentum_std**2]])],
                ]
            )
            names.append(MOMENTUM)

        s_ii = np.diag(s)
        common = np.einsum("ik,kl,il->i", b, f, b)
        specific = np.maximum(self._config.residual_floor * s_ii, s_ii - common)
        scale = sigma / np.sqrt(s_ii)
        return RiskModelSnapshot(
            tickers=tickers,
            factors=tuple(names),
            B=b * scale[:, None],
            F=f,
            D=specific * scale**2,
            sigma=sigma,
            factor_weights=factor_weights,
            macro_betas=macro_betas,
            momentum=momentum,
            ready=True,
        )

    def _momentum_exposure(
        self, tickers: tuple[Ticker, ...], sigma: FloatArray
    ) -> FloatArray | None:
        raw = {t: value for t in tickers if (value := self._momentum.raw(t)) is not None}
        exposures = momentum_exposures(raw, dict(zip(tickers, sigma.tolist(), strict=True)))
        if not exposures:
            return None
        return np.array([exposures.get(t, 0.0) for t in tickers])


def _complete_universe(corr: FloatArray, vol: FloatArray) -> list[int]:
    """Tickers with warm vol and correlation, and a finite covariance with
    every other such ticker."""
    candidates = [
        i
        for i in range(corr.shape[0])
        if np.isfinite(corr[i, i]) and corr[i, i] > 0 and np.isfinite(vol[i, i]) and vol[i, i] > 0
    ]
    block = corr[np.ix_(candidates, candidates)]
    return [c for j, c in enumerate(candidates) if np.isfinite(block[j]).all()]


def _clip_psd(matrix: FloatArray) -> FloatArray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.maximum(values, _EIGEN_FLOOR)) @ vectors.T
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/risk/test_model.py -v --no-cov`
Expected: PASS.
- If `test_not_ready_until_correlation_warm` is off by one, recount: the first `update` produces no return, so 50 updates = 49 returns.
- If `test_recovers_known_betas` is flaky, change the seed. Do **not** widen the tolerance beyond 0.2.

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/risk/model.py tests/risk/test_model.py
git commit -m "Add the streaming hybrid factor risk model"
```

### Task B7: Holdings-based attribution

**Files:**
- Create: `src/trading_strategies/risk/attribution.py`, `tests/risk/test_attribution.py`

**Interfaces:**
- Consumes:
  - `FactorRiskModel`, `RiskModelSnapshot`, `SPECIFIC` (B6)
  - `FactorModelConfig` (B2)
  - `PerformanceTracker.weights_history` / `.mark_to_market_history` (A1)
  - `FrameMarketData` (`backtester.data.frame_market_data`)
- Produces:
  - `RESIDUAL = "Specific + other"`
  - `AttributionResult` (frozen) with fields:
    - `exposures: pd.DataFrame`
    - `risk_shares: pd.DataFrame`
    - `contributions: pd.DataFrame` (factor columns, then `RESIDUAL` last)
    - `portfolio_returns: pd.Series[float]`
    - `predicted_vol: pd.Series[float]` (daily ex-ante)
  - `AttributionResult` methods: `standardized_returns() -> pd.Series[float]`, `bias_statistic() -> float`, `rolling_bias(window: int = 250) -> pd.Series[float]`
  - `run_attribution(events: Iterable[MarketEvent], weights_history: Sequence[tuple[datetime, Mapping[Ticker, float]]], equity_history: Sequence[tuple[datetime, float]], model: FactorRiskModel) -> AttributionResult`
  - `iter_market_events(data: Path, tickers: Sequence[Ticker]) -> Iterator[MarketEvent]`
  - `attribute_backtest(data: Path, tickers: Sequence[Ticker], model_config: FactorModelConfig, trackers: Mapping[str, PerformanceTracker]) -> dict[str, AttributionResult]`

- [ ] **Step 1: Write the failing tests** (`tests/risk/test_attribution.py`)

```python
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from backtester.core.events import Bar, MarketEvent

from trading_strategies.risk.attribution import RESIDUAL, AttributionResult, run_attribution
from trading_strategies.risk.config import FactorModelConfig, MomentumConfig
from trading_strategies.risk.model import FactorRiskModel

_START = datetime(2020, 1, 1)
_WEIGHTS = {"MKT": 0.5, "A": 0.5}


def _config(tmp_path: Path) -> FactorModelConfig:
    factors_file = tmp_path / "factors.json"
    factors_file.write_text(
        json.dumps(
            {
                "factors": [
                    {"name": "Market", "level": 1, "group": "core", "weights": {"MKT": 1.0}},
                ]
            }
        )
    )
    return FactorModelConfig(
        factors_file=str(factors_file),
        vol_span=20,
        corr_span=100,
        corr_min_periods=50,
        momentum=MomentumConfig(lookback=20, skip=2),
    )


def _scenario(
    n_bars: int = 500, seed: int = 0
) -> tuple[
    list[MarketEvent], list[tuple[datetime, dict[str, float]]], list[tuple[datetime, float]]
]:
    rng = np.random.default_rng(seed)
    closes = {"MKT": 100.0, "A": 100.0, "B": 100.0}
    events: list[MarketEvent] = []
    weights: list[tuple[datetime, dict[str, float]]] = []
    equity: list[tuple[datetime, float]] = []
    value = 1_000_000.0
    for i in range(n_bars):
        ts = _START + timedelta(days=i)
        if i > 0:
            market = rng.normal(0, 0.01)
            previous = dict(closes)
            closes["MKT"] *= math.exp(market)
            closes["A"] *= math.exp(0.5 * market + rng.normal(0, 0.005))
            closes["B"] *= math.exp(-market + rng.normal(0, 0.005))
            value *= 1 + sum(w * (closes[t] / previous[t] - 1) for t, w in _WEIGHTS.items())
        events.append(MarketEvent(timestamp=ts, bars={t: Bar(close=c) for t, c in closes.items()}))
        weights.append((ts, {} if i == 0 else dict(_WEIGHTS)))
        equity.append((ts, value))
    return events, weights, equity


def _result(tmp_path: Path) -> AttributionResult:
    events, weights, equity = _scenario()
    model = FactorRiskModel(["MKT", "A", "B"], _config(tmp_path))
    return run_attribution(events, weights, equity, model)


def test_contributions_add_up_to_the_portfolio_return(tmp_path: Path) -> None:
    result = _result(tmp_path)

    np.testing.assert_allclose(
        result.contributions.sum(axis=1).to_numpy(), result.portfolio_returns.to_numpy()
    )
    assert result.contributions.columns[-1] == RESIDUAL


def test_rows_start_only_after_the_model_is_ready(tmp_path: Path) -> None:
    result = _result(tmp_path)

    # corr_min_periods=50 returns -> first ready snapshot after bar 50,
    # used for bar 51.
    assert result.exposures.index[0] == _START + timedelta(days=51)


def test_market_exposure_matches_the_blended_beta(tmp_path: Path) -> None:
    result = _result(tmp_path)

    # 0.5 * beta(MKT)=1 + 0.5 * beta(A)=0.5
    assert result.exposures["Market"].iloc[-100:].mean() == pytest.approx(0.75, abs=0.15)


def test_bias_statistic_is_close_to_one(tmp_path: Path) -> None:
    result = _result(tmp_path)

    assert 0.7 < result.bias_statistic() < 1.3
    assert result.rolling_bias(window=100).dropna().size > 0


def test_empty_history_gives_empty_frames(tmp_path: Path) -> None:
    events, _, _ = _scenario(n_bars=30)
    model = FactorRiskModel(["MKT", "A", "B"], _config(tmp_path))

    result = run_attribution(events, [], [], model)

    assert result.exposures.empty
    assert result.contributions.empty
    assert math.isnan(result.bias_statistic())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/risk/test_attribution.py -v --no-cov`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement** (`src/trading_strategies/risk/attribution.py`)

```python
"""Holdings-based factor attribution. Per bar t, with the weights that earned
bar t's return (``PerformanceTracker.weights_history``) and the model
snapshot built from data up to bar t-1:

- exposures ``x = B^T w``
- risk shares ``x_k (F x)_k / var`` and ``w^T D w / var``
- return contributions ``x_k * f_k,t``, plus ``RESIDUAL`` = portfolio return
  minus their sum (specific returns, cash, costs, intraday trading).

Contributions sum arithmetically over time; per bar the decomposition is
exact by construction. Holdings-based rather than a returns regression,
because a trend strategy's exposures move on purpose."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from backtester.core.events import MarketEvent, Ticker
from backtester.data.frame_market_data import FrameMarketData
from backtester.tracker.metrics import PerformanceTracker

from trading_strategies.risk.config import FactorModelConfig
from trading_strategies.risk.model import FactorRiskModel

RESIDUAL = "Specific + other"


@dataclass(frozen=True)
class AttributionResult:
    exposures: pd.DataFrame
    risk_shares: pd.DataFrame
    contributions: pd.DataFrame
    portfolio_returns: pd.Series[float]
    predicted_vol: pd.Series[float]

    def standardized_returns(self) -> pd.Series[float]:
        live = self.predicted_vol > 0
        return self.portfolio_returns[live] / self.predicted_vol[live]

    def bias_statistic(self) -> float:
        z = self.standardized_returns()
        return float(z.std()) if len(z) > 1 else math.nan

    def rolling_bias(self, window: int = 250) -> pd.Series[float]:
        return self.standardized_returns().rolling(window, min_periods=window // 2).std()


def run_attribution(
    events: Iterable[MarketEvent],
    weights_history: Sequence[tuple[datetime, Mapping[Ticker, float]]],
    equity_history: Sequence[tuple[datetime, float]],
    model: FactorRiskModel,
) -> AttributionResult:
    weights_at = dict(weights_history)
    returns_at = {
        timestamp: equity / previous - 1.0
        for (_, previous), (timestamp, equity) in itertools.pairwise(equity_history)
        if previous > 0
    }
    index: list[datetime] = []
    exposures: list[dict[str, float]] = []
    shares: list[dict[str, float]] = []
    contributions: list[dict[str, float]] = []
    portfolio: list[float] = []
    predicted: list[float] = []
    last_close: dict[Ticker, float] = {}

    for event in events:
        closes = {ticker: bar.close for ticker, bar in event.bars.items()}
        snapshot = model.snapshot  # data up to the previous bar
        timestamp = event.timestamp
        if snapshot.ready and timestamp in weights_at and timestamp in returns_at:
            asset_returns = {
                t: c / last_close[t] - 1.0 for t, c in closes.items() if t in last_close
            }
            w = snapshot.weight_vector(weights_at[timestamp])
            x = snapshot.exposures(w)
            explained = x * snapshot.factor_returns(asset_returns)
            portfolio_return = returns_at[timestamp]
            index.append(timestamp)
            exposures.append(dict(zip(snapshot.factors, x.tolist(), strict=True)))
            shares.append(snapshot.risk_decomposition(w))
            contributions.append(
                dict(zip(snapshot.factors, explained.tolist(), strict=True))
                | {RESIDUAL: portfolio_return - float(explained.sum())}
            )
            portfolio.append(portfolio_return)
            predicted.append(math.sqrt(snapshot.portfolio_variance(w)))
        last_close.update(closes)
        model.update(closes)

    dates = pd.DatetimeIndex(index)
    contribution_frame = pd.DataFrame(contributions, index=dates).fillna(0.0)
    if not contribution_frame.empty:
        ordered = [c for c in contribution_frame.columns if c != RESIDUAL] + [RESIDUAL]
        contribution_frame = contribution_frame[ordered]
    return AttributionResult(
        exposures=pd.DataFrame(exposures, index=dates).fillna(0.0),
        risk_shares=pd.DataFrame(shares, index=dates).fillna(0.0),
        contributions=contribution_frame,
        portfolio_returns=pd.Series(portfolio, index=dates, dtype=float),
        predicted_vol=pd.Series(predicted, index=dates, dtype=float),
    )


def iter_market_events(data: Path, tickers: Sequence[Ticker]) -> Iterator[MarketEvent]:
    market_data = FrameMarketData(data, tickers=list(tickers))
    while (event := market_data.get_next_bar()) is not None:
        yield event


def attribute_backtest(
    data: Path,
    tickers: Sequence[Ticker],
    model_config: FactorModelConfig,
    trackers: Mapping[str, PerformanceTracker],
) -> dict[str, AttributionResult]:
    """Replay the causal model over the backtest's price data once per leg.
    The replay gives the same snapshots a live run would."""
    return {
        label: run_attribution(
            iter_market_events(data, tickers),
            tracker.weights_history,
            tracker.mark_to_market_history,
            FactorRiskModel(tickers, model_config),
        )
        for label, tracker in trackers.items()
    }
```

`iter_market_events` and `attribute_backtest` are covered by the integration test in Task B9.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/risk/test_attribution.py -v --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/risk/attribution.py tests/risk/test_attribution.py
git commit -m "Add holdings-based factor attribution"
```

### Task B8: Report pages

**Files:**
- Create: `src/trading_strategies/risk/report_pages.py`, `tests/risk/test_report_pages.py`

Load the `dataviz` skill before writing the chart code in this task. Keep the colors from `backtester.tracker.report` (`SURFACE`, `INK`, `MUTED`, `GRID`, `NEGATIVE`) so the pages match the rest of the PDF.

**Interfaces:**
- Consumes:
  - `AttributionResult`, `RESIDUAL` (B7), `SPECIFIC` (B6)
  - `ReportPage`, `new_page`, `save_page`, `style_table` and the colors (A2)
  - `TRADING_DAYS_PER_YEAR`
- Produces:
  - `FactorExposurePage`, `RiskDecompositionPage`, `ReturnAttributionPage`, `ModelHealthPage`: frozen dataclasses with a field `results: Mapping[str, AttributionResult]` and `render(pdf: PdfPages) -> None`
  - `attribution_pages(results: Mapping[str, AttributionResult]) -> list[ReportPage]`
- Behavior: one row per leg (skip legs with empty results), a chart on the left and a table or chart on the right. If no leg has data, write no page.

- [ ] **Step 1: Write the failing tests** (`tests/risk/test_report_pages.py`)

```python
from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from trading_strategies.risk.attribution import RESIDUAL, AttributionResult
from trading_strategies.risk.model import SPECIFIC
from trading_strategies.risk.report_pages import attribution_pages


class _CountingPdf:
    def __init__(self) -> None:
        self.pages = 0

    def savefig(self, figure: Figure, **_: Any) -> None:
        self.pages += 1
        plt.close(figure)


def _result(n: int = 300) -> AttributionResult:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, n), index=dates)
    return AttributionResult(
        exposures=pd.DataFrame(
            {"Equity": rng.normal(0.5, 0.1, n), "Momentum": rng.normal(1.0, 0.2, n)}, index=dates
        ),
        risk_shares=pd.DataFrame({"Equity": 0.5, "Momentum": 0.3, SPECIFIC: 0.2}, index=dates),
        contributions=pd.DataFrame(
            {"Equity": returns * 0.5, "Momentum": returns * 0.3, RESIDUAL: returns * 0.2},
            index=dates,
        ),
        portfolio_returns=returns,
        predicted_vol=pd.Series(0.01, index=dates),
    )


def _empty() -> AttributionResult:
    dates = pd.DatetimeIndex([])
    return AttributionResult(
        exposures=pd.DataFrame(index=dates),
        risk_shares=pd.DataFrame(index=dates),
        contributions=pd.DataFrame(index=dates),
        portfolio_returns=pd.Series([], index=dates, dtype=float),
        predicted_vol=pd.Series([], index=dates, dtype=float),
    )


def test_four_pages_render_for_two_legs() -> None:
    pdf = _CountingPdf()

    for page in attribution_pages({"Strategy": _result(), "Buy & Hold": _result()}):
        page.render(cast(PdfPages, pdf))

    assert pdf.pages == 4


def test_empty_legs_are_skipped_and_all_empty_writes_nothing() -> None:
    pdf = _CountingPdf()

    for page in attribution_pages({"Strategy": _empty()}):
        page.render(cast(PdfPages, pdf))

    assert pdf.pages == 0


def test_pages_render_into_a_real_pdf(tmp_path: Any) -> None:
    path = tmp_path / "pages.pdf"
    with PdfPages(path) as pdf:
        for page in attribution_pages({"Strategy": _result(), "Empty": _empty()}):
            page.render(pdf)

    assert path.stat().st_size > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/risk/test_report_pages.py -v --no-cov`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement** (`src/trading_strategies/risk/report_pages.py`)

```python
"""Factor-attribution pages for ``backtester``'s PDF report, via its
``ReportPage`` / ``extra_pages`` API. One row per leg; legs without
attribution rows are skipped, and a page with no legs is not written."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR
from backtester.tracker.report import (
    GRID,
    INK,
    MUTED,
    NEGATIVE,
    SURFACE,
    ReportPage,
    new_page,
    save_page,
    style_table,
)
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

from trading_strategies.risk.attribution import AttributionResult
from trading_strategies.risk.model import SPECIFIC

_ANNUAL = math.sqrt(TRADING_DAYS_PER_YEAR)
_BIAS_BAND = (0.8, 1.2)

_Draw = Callable[[Axes, Axes, str, AttributionResult], None]


def _style(ax: Axes, title: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=6)


def _colors(names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return {name: cmap(i % 20) for i, name in enumerate(names)}


def _lines(ax: Axes, frame: pd.DataFrame, title: str) -> None:
    names = [str(c) for c in frame.columns]
    colors = _colors(names)
    for name in names:
        series = frame[name]
        ax.plot(series.index, series.to_numpy(), color=colors[name], linewidth=1.0, label=name)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.legend(fontsize=6.5, ncol=3, frameon=False, loc="upper left")
    _style(ax, title)


def _render_legs(
    pdf: PdfPages, title: str, results: Mapping[str, AttributionResult], draw: _Draw
) -> None:
    legs = {label: r for label, r in results.items() if not r.exposures.empty}
    if not legs:
        return
    fig = new_page(title)
    grid = fig.add_gridspec(
        len(legs),
        2,
        width_ratios=[1.6, 1],
        left=0.06,
        right=0.975,
        top=0.86,
        bottom=0.07,
        hspace=0.45,
        wspace=0.18,
    )
    for row, (label, result) in enumerate(legs.items()):
        draw(fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1]), label, result)
    save_page(pdf, fig)


def _draw_exposures(left: Axes, right: Axes, label: str, result: AttributionResult) -> None:
    exposures = result.exposures
    _lines(left, exposures, label)
    rows = [
        [str(name), f"{col.mean():+.2f}", f"{col.min():+.2f}", f"{col.max():+.2f}"]
        for name, col in exposures.items()
    ]
    style_table(right, rows, ["Factor", "Mean", "Min", "Max"], cellLoc="center")


def _draw_risk(left: Axes, right: Axes, label: str, result: AttributionResult) -> None:
    shares = result.risk_shares.mean()
    names = [str(n) for n in shares.index]
    left.barh(names, shares.to_numpy(), color=[MUTED if n == SPECIFIC else INK for n in names])
    left.invert_yaxis()
    left.xaxis.set_major_formatter(PercentFormatter(1.0))
    _style(left, f"{label} - mean share of ex-ante variance")

    ex_ante = result.predicted_vol * _ANNUAL
    realized = result.portfolio_returns.rolling(60, min_periods=20).std() * _ANNUAL
    right.plot(ex_ante.index, ex_ante.to_numpy(), color=INK, linewidth=1.0, label="Ex-ante")
    right.plot(
        realized.index, realized.to_numpy(), color=NEGATIVE, linewidth=1.0, label="Realized 60d"
    )
    right.yaxis.set_major_formatter(PercentFormatter(1.0))
    right.legend(fontsize=7, frameon=False, loc="upper left")
    _style(right, "Annualized vol")


def _draw_returns(left: Axes, right: Axes, label: str, result: AttributionResult) -> None:
    _lines(left, result.contributions.cumsum(), f"{label} - cumulative contribution")
    left.yaxis.set_major_formatter(PercentFormatter(1.0))
    annual = result.contributions.mean() * TRADING_DAYS_PER_YEAR
    rows = [[str(name), f"{value:+.2%}"] for name, value in annual.items()]
    style_table(right, rows, ["Source", "Annualized"], cellLoc="center")


def _draw_health(left: Axes, right: Axes, label: str, result: AttributionResult) -> None:
    rolling = result.rolling_bias()
    left.axhspan(*_BIAS_BAND, color=GRID, alpha=0.6)
    left.axhline(1.0, color=MUTED, linewidth=0.8)
    left.plot(rolling.index, rolling.to_numpy(), color=INK, linewidth=1.0)
    _style(left, f"{label} - rolling bias statistic (target 1.0)")

    rows = [
        ["Bias statistic", f"{result.bias_statistic():.2f}"],
        ["Bars", f"{len(result.portfolio_returns):,}"],
        ["Mean ex-ante vol", f"{result.predicted_vol.mean() * _ANNUAL:.2%}"],
        ["Realized vol", f"{result.portfolio_returns.std() * _ANNUAL:.2%}"],
    ]
    style_table(right, rows, ["Metric", "Value"], cellLoc="center")


@dataclass(frozen=True)
class FactorExposurePage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        _render_legs(pdf, "Factor Exposures", self.results, _draw_exposures)


@dataclass(frozen=True)
class RiskDecompositionPage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        _render_legs(pdf, "Risk Decomposition", self.results, _draw_risk)


@dataclass(frozen=True)
class ReturnAttributionPage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        _render_legs(pdf, "Return Attribution", self.results, _draw_returns)


@dataclass(frozen=True)
class ModelHealthPage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        _render_legs(pdf, "Factor Model Health", self.results, _draw_health)


def attribution_pages(results: Mapping[str, AttributionResult]) -> list[ReportPage]:
    return [
        FactorExposurePage(results),
        RiskDecompositionPage(results),
        ReturnAttributionPage(results),
        ModelHealthPage(results),
    ]
```

- [ ] **Step 4: Run the tests and look at the pages**

Run: `uv run pytest tests/risk/test_report_pages.py -v --no-cov`
Expected: PASS.
Open the `pages.pdf` from a manual run (for example, copy the last test body into a scratch script). Check that the legends and tables do not overlap at 2 legs. Adjust `hspace`/font sizes if needed.

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/risk/report_pages.py tests/risk/test_report_pages.py
git commit -m "Add factor attribution report pages"
```

### Task B9: Wire into the backtest script and configs

**Files:**
- Modify: `scripts/run_backtest.py`, `configs/backtest_macd.json`, `configs/backtest_macd_rescaled.json`
- Create: `tests/risk/test_integration.py`

**Interfaces:**
- Consumes: `attribute_backtest` (B7), `attribution_pages` (B8), `save_report(extra_pages=...)` (A2), `MacdBacktestConfig.factor_model` (B2)

- [ ] **Step 1: Write the failing integration test** (`tests/risk/test_integration.py`)

```python
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from backtester.tracker.metrics import monthly_returns_table, strategy_correlation_matrix
from backtester.tracker.report import save_report

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.network_momentum.runner import run_macd_benchmark
from trading_strategies.risk.attribution import attribute_backtest
from trading_strategies.risk.config import FactorModelConfig
from trading_strategies.risk.report_pages import attribution_pages

REPO_FACTORS = Path(__file__).resolve().parents[2] / "configs" / "factors.json"
_TICKERS = ("SPY", "QQQ", "TLT")
_NUM_BARS = 500


def _write_synthetic_parquet(path: Path) -> None:
    rng = np.random.default_rng(7)
    start = datetime(2020, 1, 1)
    rows = [
        {"date": start + timedelta(days=i), "ticker": ticker, "close": float(close)}
        for ticker in _TICKERS
        for i, close in enumerate(100 + np.cumsum(rng.normal(0, 1, _NUM_BARS)))
    ]
    pd.DataFrame(rows).to_parquet(path)


def test_backtest_report_includes_factor_attribution(tmp_path: Path) -> None:
    data = tmp_path / "raw.parquet"
    _write_synthetic_parquet(data)
    config = MacdBacktestConfig(
        name="MACD Benchmark",
        data=str(data),
        tickers=list(_TICKERS),
        output_dir=str(tmp_path / "output"),
        factor_model=FactorModelConfig(factors_file=str(REPO_FACTORS)),
    )
    assert config.factor_model is not None

    trackers = run_macd_benchmark(config)
    results = attribute_backtest(Path(config.data), config.tickers, config.factor_model, trackers)

    buy_and_hold = results["Buy & Hold"]
    assert not buy_and_hold.exposures.empty
    assert "Equity" in buy_and_hold.exposures.columns  # SPY-only Equity leg
    histories = {label: t.mark_to_market_history for label, t in trackers.items()}
    path = save_report(
        Path(config.output_dir),
        histories,
        {label: t.metrics() for label, t in trackers.items()},
        {label: t.trade_metrics() for label, t in trackers.items()},
        {label: monthly_returns_table(h) for label, h in histories.items()},
        strategy_correlation_matrix(histories),
        extra_pages=attribution_pages(results),
        config=config,
    )
    assert path.stat().st_size > 0
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/risk/test_integration.py -v --no-cov`
Expected: PASS right away (B2–B8 are done). It covers `iter_market_events` and `attribute_backtest`. If it fails, fix the cause in the relevant module. Do not weaken the test.

- [ ] **Step 3: Wire the script** (`scripts/run_backtest.py`)

Add the imports:

```python
from backtester.tracker.report import ReportPage, save_report

from trading_strategies.risk.attribution import attribute_backtest
from trading_strategies.risk.report_pages import attribution_pages
```

In `main()`, add before `report_path = save_report(`:

```python
extra_pages: list[ReportPage] = []
if config.factor_model is not None:
    results = attribute_backtest(Path(config.data), config.tickers, config.factor_model, trackers)
    for label, result in results.items():
        logger.info("%s: factor-model bias statistic %.2f", label, result.bias_statistic())
    extra_pages = attribution_pages(results)
```

Pass `extra_pages=extra_pages,` to `save_report` (before `config=config`).

- [ ] **Step 4: Add the section to both backtest configs**

Add to `configs/backtest_macd.json` and `configs/backtest_macd_rescaled.json`, after `"output_dir"`:

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

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest && uv run mypy . && uv run ruff check . && uv run ruff format --check .`
Expected: all green, coverage ≥ 95%.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_backtest.py configs/backtest_macd.json configs/backtest_macd_rescaled.json tests/risk/test_integration.py
git commit -m "Add factor attribution pages to the backtest report"
```

### Task B10: Real-data check and PR

- [ ] **Step 1: Run on real data**

Run: `uv run scripts/run_backtest.py configs/backtest_macd.json`

If `data/raw.parquet` is missing, first run `uv run scripts/fetch_data.py configs/fetch_data.json`.

Expected:
- The log shows a `factor-model bias statistic` line per leg.
- The new PDF in `output/` has the four factor pages.

- [ ] **Step 2: Check the success criteria from the spec**
  - The Buy & Hold bias statistic is in **0.8–1.2**. If it is not, report the value to the user. Do not tune it silently. `residual_floor` and `corr_span` are the tuning knobs, and the bias statistic (not Sharpe) is the guide.
  - Buy & Hold is mostly Equity in the risk-decomposition page.
  - The MACD leg shows a clear Momentum exposure and an Equity exposure that changes over time.
- [ ] **Step 3: Re-pin `backtester`** to the merge commit of its PR (Task B1 Steps 1–4). Skip this if the PR is not merged yet, and say so in the PR description.
- [ ] **Step 4: Push and open the PR** (ask the user first if pushing was not approved yet)

```bash
git push -u origin feature/factor-risk-model
gh pr create --title "Factor risk model and attribution report pages" --body "Implements docs/superpowers/specs/2026-09-22-factor-risk-model-design.md (Spec 1). Adds trading_strategies.risk (streaming hybrid factor model + holdings-based attribution) and four report pages. Real-data bias statistic: <fill in from Step 1>."
```

No `Co-Authored-By` / "Generated with" lines (repo rule).
