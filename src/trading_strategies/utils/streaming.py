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
