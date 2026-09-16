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
