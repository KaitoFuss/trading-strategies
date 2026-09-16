# MACD Benchmark Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate the MACD benchmark strategy from Poh, Wood, Roberts & Zohren (2023), "Network Momentum across Asset Classes" (arXiv:2308.11294), Eq. 10, as a streaming `Strategy` + `Portfolio` pair against the `backtester` library's Protocols.

**Architecture:** Four O(1)/bounded-memory streaming primitives (`utils/streaming.py`) that reproduce pandas' `.ewm()`/`.rolling()` semantics one value at a time, composed into the 8 momentum/MACD features (`utils/features.py`), consumed by a new `MacdBenchmarkStrategy` (`network_momentum/strategies.py`) and a new `TargetVolPortfolio` (`network_momentum/portfolio.py`) implementing the paper's Eq. 9 position sizing.

**Tech Stack:** Python 3.12, pandas/numpy (tests only — production code is plain-float, no pandas), `backtester` (git dependency, not modified), pytest, mypy --strict, ruff.

**Spec:** `docs/superpowers/specs/2026-09-16-macd-benchmark-design.md`

## Global Constraints

- Streaming output must match pandas batch output bar-for-bar, including exact `min_periods`/warm-up behavior — verified against pandas directly in tests, not against a from-scratch reimplementation.
- `WINSOR_HALFLIFE = 31`, `WINSOR_Z = 5.0`, `VOL_SPAN = 60`, `MOMENTUM_LOOKBACKS = (1, 21, 63, 126, 252)`, `MACD_PAIRS = ((8, 24), (16, 48), (32, 96))` — copied verbatim from today's `features.py`, unchanged.
- `TargetVolPortfolio`: `target_vol = 0.15` annualized default, gross **uncapped** (no renormalization to `max_gross`).
- `MacdBenchmarkStrategy` does **not** winsorize its MACD inputs (paper's Eq. 10 has no winsorization step).
- Coverage gate stays at 95% (`pyproject.toml`'s `--cov-fail-under=95`). Every new module needs tests before commit.
- mypy runs in strict mode (`pyproject.toml`) — all new code must be fully typed.
- `src/trading_strategies/features.py` and `tests/test_features.py` are deleted; nothing may still import from `trading_strategies.features` when this plan is done.

---

## Task 1: Scaffold packages, `EwmMean` primitive

**Files:**
- Create: `src/trading_strategies/utils/__init__.py` (empty)
- Create: `src/trading_strategies/utils/streaming.py`
- Create: `src/trading_strategies/network_momentum/__init__.py` (empty)
- Test: `tests/utils/__init__.py` (empty)
- Test: `tests/utils/test_streaming.py`

**Interfaces:**
- Produces: `EwmMean(*, span: int | None = None, halflife: float | None = None, min_periods: int)` with `.update(value: float) -> float | None`.

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p src/trading_strategies/utils src/trading_strategies/network_momentum tests/utils tests/network_momentum
touch src/trading_strategies/utils/__init__.py
touch src/trading_strategies/network_momentum/__init__.py
touch tests/utils/__init__.py
touch tests/network_momentum/__init__.py
```

- [ ] **Step 2: Write the failing test for `EwmMean`**

```python
# tests/utils/test_streaming.py
import numpy as np
import pandas as pd
import pytest

from trading_strategies.utils.streaming import EwmMean


def test_ewm_mean_matches_pandas_span() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 40)
    span, min_periods = 5, 5
    expected = pd.Series(values).ewm(span=span, min_periods=min_periods).mean()

    ewm = EwmMean(span=span, min_periods=min_periods)
    actual = [ewm.update(v) for v in values]

    for a, e in zip(actual, expected, strict=True):
        if np.isnan(e):
            assert a is None
        else:
            assert a == pytest.approx(e)


def test_ewm_mean_matches_pandas_halflife() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, 40)
    halflife, min_periods = 10.0, 10
    expected = pd.Series(values).ewm(halflife=halflife, min_periods=min_periods).mean()

    ewm = EwmMean(halflife=halflife, min_periods=min_periods)
    actual = [ewm.update(v) for v in values]

    for a, e in zip(actual, expected, strict=True):
        if np.isnan(e):
            assert a is None
        else:
            assert a == pytest.approx(e)


def test_ewm_mean_rejects_both_span_and_halflife() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        EwmMean(span=5, halflife=5.0, min_periods=5)


def test_ewm_mean_rejects_neither_span_nor_halflife() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        EwmMean(min_periods=5)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/utils/test_streaming.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_strategies.utils.streaming'`

- [ ] **Step 4: Implement `EwmMean`**

```python
# src/trading_strategies/utils/streaming.py
"""Streaming primitives that reproduce pandas' EWM/rolling semantics one
value at a time, so a `Strategy` can compute features bar-by-bar instead of
recomputing a whole history on every `MarketEvent`.

Each primitive's output is verified against the equivalent pandas batch
call in tests -- these are not independent formulas, they are the same
pandas math, restated as an O(1) or bounded-memory recursion.
"""

from __future__ import annotations

import math
from collections import deque


def _alpha_from_span(span: int) -> float:
    return 2.0 / (span + 1)


def _alpha_from_halflife(halflife: float) -> float:
    return 1.0 - math.exp(math.log(0.5) / halflife)


def _resolve_alpha(span: int | None, halflife: float | None) -> float:
    if (span is None) == (halflife is None):
        raise ValueError("pass exactly one of span or halflife")
    return _alpha_from_span(span) if span is not None else _alpha_from_halflife(halflife)  # type: ignore[arg-type]


class EwmMean:
    """Matches ``Series.ewm(span=..., min_periods=...).mean()`` (pandas'
    default ``adjust=True``), one value at a time."""

    def __init__(
        self, *, span: int | None = None, halflife: float | None = None, min_periods: int
    ) -> None:
        self._alpha = _resolve_alpha(span, halflife)
        self._min_periods = min_periods
        self._numer = 0.0
        self._denom = 0.0
        self._count = 0

    def update(self, value: float) -> float | None:
        a = self._alpha
        self._numer = value + (1 - a) * self._numer
        self._denom = 1 + (1 - a) * self._denom
        self._count += 1
        if self._count < self._min_periods:
            return None
        return self._numer / self._denom
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/utils/test_streaming.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/trading_strategies/utils src/trading_strategies/network_momentum tests/utils tests/network_momentum
git commit -m "Add EwmMean streaming primitive"
```

---

## Task 2: `EwmMoments` primitive (mean + std)

**Files:**
- Modify: `src/trading_strategies/utils/streaming.py`
- Test: `tests/utils/test_streaming.py`

**Interfaces:**
- Consumes: `_resolve_alpha` from Task 1.
- Produces: `EwmMoments(*, span: int | None = None, halflife: float | None = None, min_periods: int)` with `.update(value: float) -> None`, `.mean -> float | None`, `.std -> float | None`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/utils/test_streaming.py
from trading_strategies.utils.streaming import EwmMoments


def test_ewm_moments_matches_pandas_mean_and_std_span() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(0, 1, 60)
    span, min_periods = 8, 8
    expected_mean = pd.Series(values).ewm(span=span, min_periods=min_periods).mean()
    expected_std = pd.Series(values).ewm(span=span, min_periods=min_periods).std()

    ewm = EwmMoments(span=span, min_periods=min_periods)
    means, stds = [], []
    for v in values:
        ewm.update(v)
        means.append(ewm.mean)
        stds.append(ewm.std)

    for a, e in zip(means, expected_mean, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e)
    for a, e in zip(stds, expected_std, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e)


def test_ewm_moments_matches_pandas_std_halflife() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(0, 1, 60)
    halflife, min_periods = 31, 31
    expected_std = pd.Series(values).ewm(halflife=halflife, min_periods=min_periods).std()

    ewm = EwmMoments(halflife=halflife, min_periods=min_periods)
    stds = []
    for v in values:
        ewm.update(v)
        stds.append(ewm.std)

    for a, e in zip(stds, expected_std, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e)


def test_ewm_moments_zero_for_constant_input() -> None:
    ewm = EwmMoments(span=10, min_periods=10)
    for _ in range(20):
        ewm.update(1.0)
    assert ewm.std == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/utils/test_streaming.py -v -k ewm_moments`
Expected: FAIL with `ImportError: cannot import name 'EwmMoments'`

- [ ] **Step 3: Implement `EwmMoments`**

```python
# append to src/trading_strategies/utils/streaming.py
class EwmMoments:
    """Matches ``Series.ewm(...).mean()`` and ``Series.ewm(...).std()``
    together (they share the same decaying accumulators). ``.std`` uses
    pandas' bias-corrected weighted-variance formula (``bias=False``, the
    default): ``var = (Sw*Sxx - Sx^2) / (Sw^2 - Sw2)`` where ``Sx = sum(w*x)``,
    ``Sxx = sum(w*x^2)``, ``Sw = sum(w)``, ``Sw2 = sum(w^2)``, all decayed by
    ``(1-alpha)`` each step."""

    def __init__(
        self, *, span: int | None = None, halflife: float | None = None, min_periods: int
    ) -> None:
        self._alpha = _resolve_alpha(span, halflife)
        self._min_periods = min_periods
        self._sx = 0.0
        self._sxx = 0.0
        self._sw = 0.0
        self._sw2 = 0.0
        self._count = 0

    def update(self, value: float) -> None:
        a = self._alpha
        self._sx = value + (1 - a) * self._sx
        self._sxx = value * value + (1 - a) * self._sxx
        self._sw = 1 + (1 - a) * self._sw
        self._sw2 = 1 + (1 - a) ** 2 * self._sw2
        self._count += 1

    @property
    def mean(self) -> float | None:
        if self._count < self._min_periods:
            return None
        return self._sx / self._sw

    @property
    def std(self) -> float | None:
        if self._count < self._min_periods or self._count < 2:
            return None
        denom = self._sw**2 - self._sw2
        if denom <= 0:
            return None
        variance = (self._sw * self._sxx - self._sx**2) / denom
        return math.sqrt(variance) if variance > 0 else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/utils/test_streaming.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/utils/streaming.py tests/utils/test_streaming.py
git commit -m "Add EwmMoments streaming primitive"
```

---

## Task 3: `RollingStd` and `RollingLag` primitives

**Files:**
- Modify: `src/trading_strategies/utils/streaming.py`
- Test: `tests/utils/test_streaming.py`

**Interfaces:**
- Produces: `RollingStd(window: int)` with `.update(value: float) -> float | None`.
- Produces: `RollingLag(lookback: int)` with `.update(value: float) -> float | None`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/utils/test_streaming.py
from trading_strategies.utils.streaming import RollingLag, RollingStd


def test_rolling_std_matches_pandas_rolling_std() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(0, 1, 30)
    window = 6
    expected = pd.Series(values).rolling(window, min_periods=window).std()

    roller = RollingStd(window)
    actual = [roller.update(v) for v in values]

    for a, e in zip(actual, expected, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e)


def test_rolling_lag_returns_value_from_lookback_steps_ago() -> None:
    lag = RollingLag(lookback=3)
    results = [lag.update(v) for v in [10.0, 20.0, 30.0, 40.0, 50.0]]
    assert results == [None, None, None, 10.0, 20.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/utils/test_streaming.py -v -k "rolling"`
Expected: FAIL with `ImportError: cannot import name 'RollingStd'`

- [ ] **Step 3: Implement both primitives**

```python
# append to src/trading_strategies/utils/streaming.py
class RollingStd:
    """Matches ``Series.rolling(window, min_periods=window).std()``: a
    genuine fixed window, not an EMA -- needs the actual last ``window``
    values, not just decaying scalars. Recomputes two-pass with ``math.fsum``
    each step (matching ``backtester.stats.mean_and_stdev``'s numerical-
    accuracy convention) rather than an incremental sum-of-squares; the
    window is at most a few hundred floats, so the O(window) recompute is
    cheap."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._buffer: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> float | None:
        self._buffer.append(value)
        if len(self._buffer) < self._window:
            return None
        n = len(self._buffer)
        mean = math.fsum(self._buffer) / n
        variance = math.fsum((v - mean) ** 2 for v in self._buffer) / (n - 1)
        return math.sqrt(variance)


class RollingLag:
    """The value ``lookback`` steps back, or ``None`` until ``lookback + 1``
    values have been seen -- what a plain ``Series.diff(lookback)`` needs."""

    def __init__(self, lookback: int) -> None:
        self._buffer: deque[float] = deque(maxlen=lookback + 1)

    def update(self, value: float) -> float | None:
        self._buffer.append(value)
        assert self._buffer.maxlen is not None
        if len(self._buffer) < self._buffer.maxlen:
            return None
        return self._buffer[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/utils/test_streaming.py -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/utils/streaming.py tests/utils/test_streaming.py
git commit -m "Add RollingStd and RollingLag streaming primitives"
```

---

## Task 4: `response_function` and `_Winsorizer`

**Files:**
- Create: `src/trading_strategies/utils/features.py`
- Create: `tests/utils/test_features.py`

**Interfaces:**
- Consumes: `EwmMoments` from Task 2.
- Produces: `response_function(normalized: float) -> float`; module constants `MOMENTUM_LOOKBACKS`, `MACD_PAIRS`, `VOL_SPAN`, `WINSOR_HALFLIFE`, `WINSOR_Z`; `_Winsorizer` (private, used by Tasks 5-6).

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/test_features.py
import math

import numpy as np
import pandas as pd
import pytest

from trading_strategies.utils.features import WINSOR_HALFLIFE, WINSOR_Z, response_function


def test_response_function_matches_baz_formula() -> None:
    y = 1.5
    expected = y * math.exp(-(y**2) / 4) / 0.89
    assert response_function(y) == pytest.approx(expected)


def test_response_function_bounded_near_one() -> None:
    for y in (-10.0, -1.0, 0.0, 1.0, 10.0):
        assert abs(response_function(y)) <= 1.1


def test_winsorizer_passes_value_through_during_warmup() -> None:
    from trading_strategies.utils.features import _Winsorizer

    winsorizer = _Winsorizer()
    # WINSOR_HALFLIFE=31 min_periods -- well before that, clip is a no-op,
    # matching Series.clip(lower=NaN, upper=NaN) leaving the value unchanged.
    assert winsorizer.apply(1000.0) == 1000.0


def test_winsorizer_matches_pandas_clip_bit_for_bit() -> None:
    from trading_strategies.utils.features import _Winsorizer

    rng = np.random.default_rng(5)
    values = rng.normal(0, 1, 100)
    values[-1] = 1000.0  # extreme outlier at the end

    series = pd.Series(values)
    mean = series.ewm(halflife=WINSOR_HALFLIFE, min_periods=WINSOR_HALFLIFE).mean()
    std = series.ewm(halflife=WINSOR_HALFLIFE, min_periods=WINSOR_HALFLIFE).std()
    expected = series.clip(lower=mean - WINSOR_Z * std, upper=mean + WINSOR_Z * std)

    winsorizer = _Winsorizer()
    actual = [winsorizer.apply(v) for v in values]

    for a, e in zip(actual, expected, strict=True):
        assert a == pytest.approx(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/utils/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_strategies.utils.features'`

- [ ] **Step 3: Implement**

```python
# src/trading_strategies/utils/features.py
"""Streaming momentum/MACD features from Poh, Lim, Zohren & Roberts (2022)
and Poh, Wood, Roberts & Zohren (2023) / network-momentum.

Each streaming class consumes one price at a time, in date order, and
returns that day's feature value (or ``None`` while still warming up) --
the shape a live ``Strategy.process_market`` needs, one bar per call.
"""

from __future__ import annotations

import math

from trading_strategies.utils.streaming import EwmMean, EwmMoments, RollingLag, RollingStd

MOMENTUM_LOOKBACKS: tuple[int, ...] = (1, 21, 63, 126, 252)
MACD_PAIRS: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96))
VOL_SPAN = 60
WINSOR_HALFLIFE = 31
WINSOR_Z = 5.0


def response_function(normalized: float) -> float:
    """Baz et al. (2015) position-sizing response: y*exp(-y^2/4)/0.89,
    squashed to ~[-1, 1]."""
    return normalized * math.exp(-(normalized**2) / 4) / 0.89


class _Winsorizer:
    """Clips a value to +/- z EWM stdevs around its own EWM mean. Passes the
    raw value through unchanged while the EWM is still warming up --
    ``Series.clip(lower=NaN, upper=NaN)`` is a no-op in pandas, not NaN, so
    this adds no warmup delay of its own."""

    def __init__(self, halflife: float = WINSOR_HALFLIFE, z: float = WINSOR_Z) -> None:
        self._ewm = EwmMoments(halflife=halflife, min_periods=int(halflife))
        self._z = z

    def apply(self, value: float) -> float:
        self._ewm.update(value)
        mean, std = self._ewm.mean, self._ewm.std
        if mean is None or std is None:
            return value
        return min(max(value, mean - self._z * std), mean + self._z * std)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/utils/test_features.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/utils/features.py tests/utils/test_features.py
git commit -m "Add response_function and _Winsorizer"
```

---

## Task 5: `StreamingMacd`

**Files:**
- Modify: `src/trading_strategies/utils/features.py`
- Modify: `tests/utils/test_features.py`

**Interfaces:**
- Consumes: `EwmMean`, `RollingStd` (Tasks 1, 3); `response_function`, `_Winsorizer` (Task 4).
- Produces: `StreamingMacd(short: int, long: int, apply_phi: bool = False, winsorize: bool = True)` with `.update(close: float) -> float | None`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/utils/test_features.py
from trading_strategies.utils.features import StreamingMacd


def _batch_macd(close: pd.Series, short: int, long: int, apply_phi: bool) -> pd.Series:
    macd = close.ewm(span=short, min_periods=short).mean() - close.ewm(
        span=long, min_periods=long
    ).mean()
    q = macd / close.rolling(63, min_periods=63).std()
    normalized = q / q.rolling(252, min_periods=252).std()
    if apply_phi:
        return normalized * np.exp(-(normalized**2) / 4) / 0.89
    return normalized


def test_streaming_macd_matches_batch_formula_unwinsorized() -> None:
    rng = np.random.default_rng(6)
    n = 400
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    expected = _batch_macd(close, short=8, long=24, apply_phi=True)

    macd = StreamingMacd(short=8, long=24, apply_phi=True, winsorize=False)
    actual = [macd.update(c) for c in close]

    for a, e in zip(actual, expected, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e, abs=1e-6)


def test_streaming_macd_positive_for_sustained_uptrend() -> None:
    rng = np.random.default_rng(7)
    n = 400
    trend = np.linspace(100, 200, n)
    noise = rng.normal(0, 0.5, n)
    macd = StreamingMacd(short=8, long=24)
    last = None
    for v in trend + noise:
        last = macd.update(v)
    assert last is not None
    assert last > 0


def test_streaming_macd_winsorizes_by_default() -> None:
    # A single extreme one-day price jump partway through an otherwise calm
    # random walk -- winsorize=True should clip the resulting outlier
    # reading, winsorize=False should pass it through unchanged, so the two
    # must diverge at (or shortly after) the jump.
    rng = np.random.default_rng(8)
    n = 500
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close[400] += 500.0  # one huge one-day jump

    unwinsorized = StreamingMacd(short=8, long=24, winsorize=False)
    winsorized = StreamingMacd(short=8, long=24, winsorize=True)

    raw_values = [unwinsorized.update(c) for c in close]
    clipped_values = [winsorized.update(c) for c in close]

    diverged = any(
        raw is not None and clipped is not None and raw != pytest.approx(clipped)
        for raw, clipped in zip(raw_values, clipped_values, strict=True)
    )
    assert diverged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/utils/test_features.py -v -k macd`
Expected: FAIL with `ImportError: cannot import name 'StreamingMacd'`

- [ ] **Step 3: Implement**

```python
# append to src/trading_strategies/utils/features.py
class StreamingMacd:
    """Normalized MACD signal (Baz et al. 2015), one price at a time.
    ``short``/``long`` are the two EMA spans of one (S, L) pair from
    ``MACD_PAIRS``."""

    def __init__(
        self, short: int, long: int, apply_phi: bool = False, winsorize: bool = True
    ) -> None:
        self._short = EwmMean(span=short, min_periods=short)
        self._long = EwmMean(span=long, min_periods=long)
        self._price_std = RollingStd(63)
        self._macd_std = RollingStd(252)
        self._apply_phi = apply_phi
        self._winsorizer = _Winsorizer() if winsorize else None

    def update(self, close: float) -> float | None:
        m_short = self._short.update(close)
        m_long = self._long.update(close)
        price_std = self._price_std.update(close)

        q = None
        if m_short is not None and m_long is not None and price_std:
            q = (m_short - m_long) / price_std
        if q is None:
            return None

        macd_std = self._macd_std.update(q)
        if not macd_std:
            return None

        normalized = q / macd_std
        value = response_function(normalized) if self._apply_phi else normalized
        return self._winsorizer.apply(value) if self._winsorizer is not None else value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/utils/test_features.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/utils/features.py tests/utils/test_features.py
git commit -m "Add StreamingMacd feature"
```

---

## Task 6: `StreamingVolScaledMomentum`

**Files:**
- Modify: `src/trading_strategies/utils/features.py`
- Modify: `tests/utils/test_features.py`

**Interfaces:**
- Consumes: `EwmMoments` (Task 2), `RollingLag` (Task 3), `response_function`, `_Winsorizer` (Task 4).
- Produces: `StreamingVolScaledMomentum(lookback: int, apply_phi: bool = False, winsorize: bool = True)` with `.update(close: float) -> float | None`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/utils/test_features.py
from trading_strategies.utils.features import VOL_SPAN, StreamingVolScaledMomentum


def _batch_vol_scaled_momentum(close: pd.Series, lookback: int) -> pd.Series:
    returns = np.log(close).diff()
    vol = returns.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
    raw_return = np.log(close).diff(lookback)
    return raw_return / (vol * np.sqrt(lookback))


def test_streaming_momentum_matches_batch_formula_unwinsorized() -> None:
    rng = np.random.default_rng(9)
    n = 400
    returns = rng.normal(0.0005, 0.01, n)
    close = pd.Series((1 + returns).cumprod() * 100.0)
    expected = _batch_vol_scaled_momentum(close, lookback=21)

    momentum = StreamingVolScaledMomentum(lookback=21, winsorize=False)
    actual = [momentum.update(c) for c in close]

    for a, e in zip(actual, expected, strict=True):
        assert (a is None) == np.isnan(e)
        if a is not None:
            assert a == pytest.approx(e, abs=1e-6)


def test_streaming_momentum_sign_matches_return_direction() -> None:
    rng = np.random.default_rng(10)
    n = 300
    returns = rng.normal(0.001, 0.01, n)
    close = (1 + returns).cumprod() * 100.0
    momentum = StreamingVolScaledMomentum(lookback=21, winsorize=False)

    values = [momentum.update(c) for c in close]
    lookback = 21
    for i in range(lookback, n):
        if values[i] is None:
            continue
        realized = close[i] / close[i - lookback] - 1
        if realized == 0:
            continue
        assert (values[i] > 0) == (realized > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/utils/test_features.py -v -k momentum`
Expected: FAIL with `ImportError: cannot import name 'StreamingVolScaledMomentum'`

- [ ] **Step 3: Implement**

```python
# append to src/trading_strategies/utils/features.py
class StreamingVolScaledMomentum:
    """Cumulative log return over ``lookback`` days, scaled by EWM vol and
    sqrt(lookback), one price at a time."""

    def __init__(self, lookback: int, apply_phi: bool = False, winsorize: bool = True) -> None:
        self._lookback = lookback
        self._apply_phi = apply_phi
        self._vol = EwmMoments(span=VOL_SPAN, min_periods=VOL_SPAN)
        self._lag = RollingLag(lookback)
        self._winsorizer = _Winsorizer() if winsorize else None
        self._last_close: float | None = None

    def update(self, close: float) -> float | None:
        if self._last_close is not None:
            self._vol.update(math.log(close / self._last_close))
        self._last_close = close

        lag_close = self._lag.update(close)
        vol = self._vol.std
        if lag_close is None or not vol:
            return None

        raw = math.log(close / lag_close) / (vol * math.sqrt(self._lookback))
        value = response_function(raw) if self._apply_phi else raw
        return self._winsorizer.apply(value) if self._winsorizer is not None else value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/utils/test_features.py -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/utils/features.py tests/utils/test_features.py
git commit -m "Add StreamingVolScaledMomentum feature"
```

---

## Task 7: `StreamingFeatureSet`

**Files:**
- Modify: `src/trading_strategies/utils/features.py`
- Modify: `tests/utils/test_features.py`

**Interfaces:**
- Consumes: `StreamingMacd` (Task 5), `StreamingVolScaledMomentum` (Task 6), `MOMENTUM_LOOKBACKS`, `MACD_PAIRS` (module constants).
- Produces: `StreamingFeatureSet(momentum_phi: bool = False, macd_phi: bool = False)` with `.update(close: float) -> dict[str, float]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/utils/test_features.py
from trading_strategies.utils.features import MACD_PAIRS, MOMENTUM_LOOKBACKS, StreamingFeatureSet


def test_streaming_feature_set_has_expected_columns_once_warm() -> None:
    rng = np.random.default_rng(11)
    n = 600
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    feature_set = StreamingFeatureSet()

    last_row: dict[str, float] = {}
    for c in close:
        last_row = feature_set.update(c)

    expected_columns = {f"mom_{lb}" for lb in MOMENTUM_LOOKBACKS} | {
        f"macd_{short}_{long}" for short, long in MACD_PAIRS
    }
    assert set(last_row) == expected_columns


def test_streaming_feature_set_returns_empty_dict_before_any_feature_is_warm() -> None:
    feature_set = StreamingFeatureSet()
    assert feature_set.update(100.0) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/utils/test_features.py -v -k feature_set`
Expected: FAIL with `ImportError: cannot import name 'StreamingFeatureSet'`

- [ ] **Step 3: Implement**

```python
# append to src/trading_strategies/utils/features.py
class StreamingFeatureSet:
    """All 8 momentum/MACD features for one ticker, one price at a time.
    ``momentum_phi``/``macd_phi`` toggle the Baz et al. response-function
    squashing for each feature family, independently, matching today's
    ``compute_all_features`` toggles."""

    def __init__(self, momentum_phi: bool = False, macd_phi: bool = False) -> None:
        self._momentum = {
            lookback: StreamingVolScaledMomentum(lookback, apply_phi=momentum_phi)
            for lookback in MOMENTUM_LOOKBACKS
        }
        self._macd = {
            (short, long): StreamingMacd(short, long, apply_phi=macd_phi)
            for short, long in MACD_PAIRS
        }

    def update(self, close: float) -> dict[str, float]:
        values: dict[str, float] = {}
        for lookback, feature in self._momentum.items():
            value = feature.update(close)
            if value is not None:
                values[f"mom_{lookback}"] = value
        for (short, long), feature in self._macd.items():
            value = feature.update(close)
            if value is not None:
                values[f"macd_{short}_{long}"] = value
        return values
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/utils/test_features.py -v`
Expected: PASS (11 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/utils/features.py tests/utils/test_features.py
git commit -m "Add StreamingFeatureSet"
```

---

## Task 8: `MacdBenchmarkStrategy`

**Files:**
- Create: `src/trading_strategies/network_momentum/strategies.py`
- Create: `tests/network_momentum/test_strategies.py`

**Interfaces:**
- Consumes: `StreamingMacd` (Task 5), `MACD_PAIRS` (from `trading_strategies.utils.features`); `MarketEvent`, `SignalEvent`, `Ticker` from `backtester.core.events`.
- Produces: `MacdBenchmarkStrategy` implementing `backtester.core.engine.Strategy` (`.process_market(event: MarketEvent) -> SignalEvent`).

- [ ] **Step 1: Write the failing test**

```python
# tests/network_momentum/test_strategies.py
from datetime import datetime, timedelta

import numpy as np

from backtester.core.events import Bar, MarketEvent
from trading_strategies.network_momentum.strategies import MacdBenchmarkStrategy
from trading_strategies.utils.features import MACD_PAIRS, StreamingMacd


def _market_events(closes: dict[str, list[float]]) -> list[MarketEvent]:
    n = len(next(iter(closes.values())))
    start = datetime(2020, 1, 1)
    events = []
    for i in range(n):
        bars = {ticker: Bar(close=series[i]) for ticker, series in closes.items()}
        events.append(MarketEvent(timestamp=start + timedelta(days=i), bars=bars))
    return events


def test_macd_benchmark_strategy_scores_nothing_while_warming_up() -> None:
    rng = np.random.default_rng(12)
    closes = {"SPY": list(100 + np.cumsum(rng.normal(0, 1, 5)))}
    strategy = MacdBenchmarkStrategy()

    for event in _market_events(closes):
        signal = strategy.process_market(event)

    assert signal.scores == {}


def test_macd_benchmark_strategy_matches_mean_of_three_macd_pairs() -> None:
    rng = np.random.default_rng(13)
    n = 400
    series = list(100 + np.cumsum(rng.normal(0, 1, n)))
    closes = {"SPY": series}

    strategy = MacdBenchmarkStrategy()
    reference = [StreamingMacd(short, long, apply_phi=True, winsorize=False) for short, long in MACD_PAIRS]

    last_score = None
    for i, event in enumerate(_market_events(closes)):
        signal = strategy.process_market(event)
        expected_parts = [state.update(series[i]) for state in reference]
        if all(p is not None for p in expected_parts):
            expected = sum(expected_parts) / len(expected_parts)
            assert signal.scores["SPY"] == pytest.approx(expected, abs=1e-9)
            last_score = expected
        else:
            assert "SPY" not in signal.scores

    assert last_score is not None
```

Add `import pytest` at the top of the test file alongside the other imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/network_momentum/test_strategies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_strategies.network_momentum.strategies'`

- [ ] **Step 3: Implement**

```python
# src/trading_strategies/network_momentum/strategies.py
"""Strategies from Poh, Wood, Roberts & Zohren (2023), "Network Momentum
across Asset Classes" (arXiv:2308.11294)."""

from __future__ import annotations

import logging

from backtester.core.events import MarketEvent, SignalEvent, Ticker

from trading_strategies.utils.features import MACD_PAIRS, StreamingMacd

logger = logging.getLogger(__name__)


class MacdBenchmarkStrategy:
    """The paper's MACD benchmark (Eq. 10, Section 4.1): the mean of the
    response-function-squashed, normalized MACD indicator over the three
    (short, long) time-scale pairs in ``MACD_PAIRS``.

    No winsorization here -- that's a Section 2.2 preprocessing step for the
    GMOM regression's input features, not part of this benchmark's own
    definition.
    """

    def __init__(self) -> None:
        self._macd: dict[Ticker, tuple[StreamingMacd, ...]] = {}

    def process_market(self, event: MarketEvent) -> SignalEvent:
        scores: dict[Ticker, float] = {}
        for ticker, bar in event.bars.items():
            states = self._macd.setdefault(
                ticker,
                tuple(
                    StreamingMacd(short, long, apply_phi=True, winsorize=False)
                    for short, long in MACD_PAIRS
                ),
            )
            values = [state.update(bar.close) for state in states]
            if all(value is not None for value in values):
                scores[ticker] = sum(values) / len(values)  # type: ignore[arg-type]
            else:
                logger.debug("%s  %s: warming up MACD pairs", event.timestamp, ticker)
        return SignalEvent(timestamp=event.timestamp, scores=scores)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/network_momentum/test_strategies.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/network_momentum/strategies.py tests/network_momentum/test_strategies.py
git commit -m "Add MacdBenchmarkStrategy"
```

---

## Task 9: `TargetVolPortfolio`

**Files:**
- Create: `src/trading_strategies/network_momentum/portfolio.py`
- Create: `tests/network_momentum/test_portfolio.py`

**Interfaces:**
- Consumes: `EwmMoments` (Task 2); `BasePortfolio` from `backtester.portfolio.base`; `PriceSource` from `backtester.core.engine`; `OrderEvent`, `SignalEvent`, `Ticker` from `backtester.core.events`; `TRADING_DAYS_PER_YEAR` from `backtester.tracker.metrics`; `log_trade` from `backtester.core.trade_log`.
- Produces: `TargetVolPortfolio(price_source, initial_cash=100_000.0, target_vol=0.15, max_gross=1.0)` implementing `backtester.core.engine.Portfolio`.

- [ ] **Step 1: Write the failing test**

A minimal fake `PriceSource` drives prices without needing `FrameMarketData`/parquet data.

```python
# tests/network_momentum/test_portfolio.py
from datetime import datetime, timedelta

import numpy as np

from backtester.core.events import SignalEvent
from trading_strategies.network_momentum.portfolio import TargetVolPortfolio


class _FakePriceSource:
    def __init__(self) -> None:
        self._prices: dict[str, float] = {}

    def set_price(self, ticker: str, price: float) -> None:
        self._prices[ticker] = price

    def get_price(self, ticker: str) -> float | None:
        return self._prices.get(ticker)


def _signal(timestamp: datetime, scores: dict[str, float]) -> SignalEvent:
    return SignalEvent(timestamp=timestamp, scores=scores)


def test_target_vol_portfolio_skips_tickers_without_warmed_up_vol() -> None:
    prices = _FakePriceSource()
    prices.set_price("SPY", 100.0)
    portfolio = TargetVolPortfolio(price_source=prices, initial_cash=100_000.0)

    orders = portfolio.process_signal(_signal(datetime(2020, 1, 1), {"SPY": 0.5}))
    assert orders == []  # no return history yet -> vol not ready


def test_target_vol_portfolio_sizes_by_eq9_formula_once_warm() -> None:
    prices = _FakePriceSource()
    portfolio = TargetVolPortfolio(price_source=prices, initial_cash=100_000.0, target_vol=0.15)

    start = datetime(2020, 1, 1)
    price = 100.0
    rng = np.random.default_rng(0)
    for i in range(65):
        price *= 1 + rng.normal(0, 0.01)
        prices.set_price("SPY", price)
        # SPY must appear in `scores` every bar for `_update_vol` to track it
        # at all -- a ticker absent from every signal is invisible to this
        # portfolio, same as `InverseVolPortfolio`. Score of 0.0 here is a
        # placeholder just to keep SPY visible while vol warms up.
        portfolio.process_signal(_signal(start + timedelta(days=i), {"SPY": 0.0}))

    # one more bar, now meaningfully scored: check the order matches Eq. 9
    price *= 1.01
    prices.set_price("SPY", price)
    equity_before = portfolio.mark_to_market()
    orders = portfolio.process_signal(_signal(start + timedelta(days=65), {"SPY": 0.5}))

    assert len(orders) == 1
    order = orders[0]
    vol = portfolio._annualized_vol("SPY")  # white-box check of internal state; ruff has no private-access rule enabled here
    assert vol is not None
    expected_weight = (0.5 / 1) * (0.15 / vol)
    expected_qty = round(expected_weight * equity_before / price)
    assert order.quantity == abs(expected_qty)
    assert order.direction == ("BUY" if expected_qty > 0 else "SELL")


def test_target_vol_portfolio_gross_is_uncapped() -> None:
    """A very-low-vol ticker with a strong score should be able to size
    past max_gross=1.0 -- Eq. 9 has no cross-sectional renormalization."""
    prices = _FakePriceSource()
    portfolio = TargetVolPortfolio(
        price_source=prices, initial_cash=100_000.0, target_vol=0.15, max_gross=1.0
    )

    start = datetime(2020, 1, 1)
    price = 100.0
    for i in range(65):
        # tiny, near-constant wiggle -> very low realized vol
        price *= 1 + (0.0001 if i % 2 == 0 else -0.0001)
        prices.set_price("SPY", price)
        # SPY must be present in `scores` every bar to stay visible to
        # `_update_vol` (see the previous test) -- 0.0 is a placeholder.
        portfolio.process_signal(_signal(start + timedelta(days=i), {"SPY": 0.0}))

    price *= 1.0001
    prices.set_price("SPY", price)
    equity = portfolio.mark_to_market()
    orders = portfolio.process_signal(_signal(start + timedelta(days=65), {"SPY": 1.0}))

    assert len(orders) == 1
    notional = orders[0].quantity * price
    assert notional / equity > 1.0  # gross > 100% of equity, i.e. uncapped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/network_momentum/test_portfolio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_strategies.network_momentum.portfolio'`

- [ ] **Step 3: Implement**

```python
# src/trading_strategies/network_momentum/portfolio.py
"""Portfolios from Poh, Wood, Roberts & Zohren (2023), "Network Momentum
across Asset Classes" (arXiv:2308.11294)."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from backtester.core.engine import PriceSource
from backtester.core.events import OrderEvent, SignalEvent, Ticker
from backtester.core.trade_log import log_trade
from backtester.portfolio.base import BasePortfolio
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR

from trading_strategies.utils.streaming import EwmMoments

logger = logging.getLogger(__name__)

_VOL_SPAN = 60


class TargetVolPortfolio(BasePortfolio):
    """Equal-1/N, target-volatility position sizing -- the paper's Eq. 9:
    ``weight_i = (1/N_t) * score_i * (sigma_tgt/sigma_i)``, with
    ``sigma_i`` the annualized 60-day-span EWM stdev of daily log returns
    and ``sigma_tgt`` the annualized target (``0.15`` by default).

    Uses the raw score, not its sign -- unlike ``InverseVolPortfolio``,
    a signal here is continuous (e.g. MACD's phi-squashed average) and its
    magnitude is part of the paper's sizing.

    Gross is deliberately uncapped: the paper applies no cross-sectional
    renormalization to fit a leverage budget, unlike every other
    ``Portfolio`` in this codebase. ``max_gross`` is accepted for
    constructor-signature consistency with ``factory.py`` but is not used
    to scale anything.
    """

    def __init__(
        self,
        price_source: PriceSource,
        initial_cash: float = 100_000.0,
        target_vol: float = 0.15,
        max_gross: float = 1.0,
    ) -> None:
        super().__init__(price_source=price_source, initial_cash=initial_cash, max_gross=max_gross)
        self._target_vol = target_vol
        self._vol: dict[Ticker, EwmMoments] = {}
        self._last_price: dict[Ticker, float] = {}

    def process_signal(self, event: SignalEvent) -> Sequence[OrderEvent]:
        self._update_vol(set(event.scores) | set(self._positions), event.timestamp)
        equity = self.mark_to_market()
        if equity <= 0:
            return []

        weights = self._target_weights(event)
        return self._orders_from_targets(weights, equity, event.timestamp)

    def _update_vol(self, tickers: set[Ticker], timestamp: datetime) -> None:
        for ticker in tickers:
            price = self._price_source.get_price(ticker)
            if price is None:
                continue
            prev = self._last_price.get(ticker)
            self._last_price[ticker] = price
            if prev is not None:
                ewm = self._vol.setdefault(ticker, EwmMoments(span=_VOL_SPAN, min_periods=_VOL_SPAN))
                ewm.update(math.log(price / prev))
                logger.debug("%s  %s: vol_ewm updated", timestamp, ticker)

    def _annualized_vol(self, ticker: Ticker) -> float | None:
        ewm = self._vol.get(ticker)
        if ewm is None or not ewm.std:
            return None
        return ewm.std * math.sqrt(TRADING_DAYS_PER_YEAR)

    def _target_weights(self, event: SignalEvent) -> dict[Ticker, float]:
        ready: dict[Ticker, tuple[float, float]] = {}
        for ticker, score in event.scores.items():
            if self._price_source.get_price(ticker) is None:
                continue
            vol = self._annualized_vol(ticker)
            if vol is None:
                logger.debug("%s  %s: vol not ready, skipping", event.timestamp, ticker)
                continue
            ready[ticker] = (score, vol)

        n = len(ready)
        if n == 0:
            return {}
        return {
            ticker: (score / n) * (self._target_vol / vol) for ticker, (score, vol) in ready.items()
        }

    def _orders_from_targets(
        self, weights: dict[Ticker, float], equity: float, timestamp: datetime
    ) -> list[OrderEvent]:
        orders: list[OrderEvent] = []
        for ticker, weight in weights.items():
            price = self._price_source.get_price(ticker)
            if price is None:
                continue
            position = self._positions.get(ticker)
            current_qty = position.quantity if position else 0
            delta = round(weight * equity / price) - current_qty
            if delta == 0:
                continue
            direction: Literal["BUY", "SELL"] = "BUY" if delta > 0 else "SELL"
            log_trade(
                logger,
                timestamp,
                "REBALANCE",
                direction,
                ticker,
                abs(delta),
                price,
                f"weight={weight:.5f}",
            )
            orders.append(
                OrderEvent(timestamp=timestamp, ticker=ticker, quantity=abs(delta), direction=direction)
            )
        return orders
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/network_momentum/test_portfolio.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading_strategies/network_momentum/portfolio.py tests/network_momentum/test_portfolio.py
git commit -m "Add TargetVolPortfolio"
```

---

## Task 10: Delete the old batch `features.py`

**Files:**
- Delete: `src/trading_strategies/features.py`
- Delete: `tests/test_features.py`

**Interfaces:** None — this task removes the module Tasks 1-9 have fully superseded.

- [ ] **Step 1: Confirm nothing still imports the old module**

Run: `grep -rn "trading_strategies.features\b" src tests scripts notebooks || echo "no matches outside the notebook"`
Expected: only `notebooks/explore_features.ipynb` matches (handled in Task 11) — no `src`/`tests`/`scripts` hits.

- [ ] **Step 2: Delete the old files**

```bash
git rm src/trading_strategies/features.py tests/test_features.py
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest`
Expected: PASS, coverage >= 95% (the deleted module's tests are replaced 1:1 by Tasks 1-9's streaming tests)

- [ ] **Step 4: Commit**

```bash
git commit -m "Delete batch features.py, superseded by streaming utils/features.py"
```

---

## Task 11: Update `notebooks/explore_features.ipynb`

**Files:**
- Modify: `notebooks/explore_features.ipynb`

**Interfaces:**
- Consumes: `StreamingFeatureSet` (Task 7).

The notebook has two parts today:
1. Build the 8-feature panel via `compute_all_features`, then sanity-check NaN warm-up, correlation, and one ticker's time series.
2. A "winsorize halflife sensitivity" analysis that already concluded `halflife=31` (parametrizing the batch `winsorize` function at 5 different halflives) — this is a settled, historical decision, not something the streaming classes need to keep re-runnable (they hardcode `WINSOR_HALFLIFE=31`, matching what part 2 already picked).

- [ ] **Step 1: Replace the panel-building cell**

Find the cell:
```python
from trading_strategies.features import MACD_PAIRS, MOMENTUM_LOOKBACKS, compute_all_features
```
Replace the import and the `_features_for_one_ticker`/`panel` cell with:

```python
from trading_strategies.utils.features import MACD_PAIRS, MOMENTUM_LOOKBACKS, StreamingFeatureSet


def _features_for_one_ticker(group: pd.DataFrame) -> pd.DataFrame:
    feature_set = StreamingFeatureSet(macd_phi=False)
    rows = [feature_set.update(close) for close in group["close"]]
    return pd.DataFrame(rows, index=group.index)


panel = raw.groupby("ticker", group_keys=True).apply(_features_for_one_ticker)
panel.index.names = ["ticker", "date"]
panel = panel.reorder_levels(["date", "ticker"]).sort_index()

print(panel.shape)
panel.tail()
```

This keeps every downstream cell (NaN sanity check, correlation heatmap, per-ticker time series plot) working unchanged — they all consume `panel`, whose shape and column names are identical to before (a bar with no feature yet ready now produces `NaN` via `pd.DataFrame(rows, ...)` filling missing dict keys, matching the old leading-`NaN` behavior).

- [ ] **Step 2: Replace the winsorize-sensitivity section**

Delete the cells under "Winsorize halflife sensitivity" (the `_raw_features_for_one_ticker` cell through the correlation-between-halflives cell) and replace the section's markdown intro with:

```markdown
## Winsorize halflife

`winsorize` clips each feature to +/- 5 EWM std around its EWM mean, with
`halflife=31`. This was chosen by a one-off sensitivity sweep (see git
history for this notebook prior to the streaming-features migration):
clipping barely happens at any halflife tested, so the choice doesn't
matter much in practice. Re-running that sweep against the streaming
features would need `_Winsorizer`'s halflife exposed as a per-run
parameter, which isn't needed for the MACD benchmark this repo has --
not done here.
```

- [ ] **Step 3: Run the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/explore_features.ipynb`
Expected: completes without error (requires `data/raw.parquet` to exist locally — run `uv run scripts/fetch_data.py configs/fetch_data.json` first if it doesn't).

- [ ] **Step 4: Commit**

```bash
git add notebooks/explore_features.ipynb
git commit -m "Update exploration notebook for streaming features"
```

---

## Task 12: Full verification pass

**Files:** None (verification only).

- [ ] **Step 1: Run the full test suite with coverage**

Run: `uv run pytest`
Expected: all tests pass, coverage >= 95%

- [ ] **Step 2: Run mypy strict**

Run: `uv run mypy .`
Expected: no errors. If `MacdBenchmarkStrategy.process_market`'s `sum(values) / len(values)` line raises an `arg-type` complaint beyond the inline `# type: ignore[arg-type]` already placed (mypy can't narrow `all(v is not None for v in values)` to `list[float]`), leave the `# type: ignore[arg-type]` as-is — it's the same narrowing limitation `ZScoreMovingAverageStrategy` doesn't hit only because it checks length rather than `all(... is not None ...)`.

- [ ] **Step 3: Run ruff**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: no remaining issues after autofix; review the diff if `--fix` changed anything beyond formatting.

- [ ] **Step 4: Confirm the old module is fully gone**

Run: `grep -rn "trading_strategies.features\b" src tests scripts notebooks`
Expected: no matches at all now (Task 11 removed the notebook's last reference).

- [ ] **Step 5: Commit any lint/format fixes**

```bash
git add -A
git commit -m "Lint and format fixes"
```

(Skip this commit if step 3 made no changes.)
