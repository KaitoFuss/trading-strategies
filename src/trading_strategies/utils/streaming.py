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
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from backtester.core.events import Ticker

FloatArray = npt.NDArray[np.float64]


def _alpha_from_span(span: int) -> float:
    return 2.0 / (span + 1)


def _alpha_from_halflife(halflife: float) -> float:
    return 1.0 - math.exp(math.log(0.5) / halflife)


def _resolve_alpha(span: int | None, halflife: float | None) -> float:
    if (span is None) == (halflife is None):
        raise ValueError("pass exactly one of span or halflife")
    return _alpha_from_span(span) if span is not None else _alpha_from_halflife(halflife)  # type: ignore[arg-type]


class EwmMoments:
    """Matches ``Series.ewm(...).mean()`` and ``Series.ewm(...).std()``
    together (they share the same decaying accumulators). ``.std`` uses
    pandas' bias-corrected weighted-variance formula (``bias=False``, the
    default): ``var = (Sw*Sxx - Sx^2) / (Sw^2 - Sw2)`` where ``Sx = sum(w*x)``,
    ``Sxx = sum(w*x^2)``, ``Sw = sum(w)``, ``Sw2 = sum(w^2)``, all decayed by
    ``(1-alpha)`` each step.

    ``min_periods`` is required with ``span``. With ``halflife`` it may be
    omitted, defaulting to ``math.ceil(halflife)`` -- ``ceil`` rather than
    ``int()`` truncation so a fractional halflife doesn't warm up slightly
    early."""

    def __init__(
        self,
        *,
        span: int | None = None,
        halflife: float | None = None,
        min_periods: int | None = None,
    ) -> None:
        self._alpha = _resolve_alpha(span, halflife)
        if min_periods is None:
            if halflife is None:
                raise ValueError("min_periods is required unless halflife is given")
            min_periods = math.ceil(halflife)
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


class EwmMean:
    """Matches ``Series.ewm(span=..., min_periods=...).mean()`` (pandas'
    default ``adjust=True``), one value at a time.

    A thin wrapper around ``EwmMoments`` -- the recursion math lives in one
    place -- that returns ``.mean`` directly from ``update`` for callers that
    only need the mean and want it inline, not via a separate property read."""

    def __init__(
        self,
        *,
        span: int | None = None,
        halflife: float | None = None,
        min_periods: int | None = None,
    ) -> None:
        self._moments = EwmMoments(span=span, halflife=halflife, min_periods=min_periods)

    def update(self, value: float) -> float | None:
        self._moments.update(value)
        return self._moments.mean


class RollingStd:
    """Matches ``Series.rolling(window, min_periods=window).std()``: a
    genuine fixed window, not an EMA -- needs the actual last ``window``
    values, not just decaying scalars. Recomputes two-pass with ``math.fsum``
    each step for numerical accuracy, rather than an incremental
    sum-of-squares; the window is at most a few hundred floats, so the
    O(window) recompute is cheap."""

    def __init__(self, window: int) -> None:
        self._max_len = window
        self._buffer: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> float | None:
        self._buffer.append(value)
        if len(self._buffer) < self._max_len:
            return None
        n = len(self._buffer)
        mean = math.fsum(self._buffer) / n
        variance = math.fsum((v - mean) ** 2 for v in self._buffer) / (n - 1)
        return math.sqrt(variance)


class RollingLag:
    """The value ``lookback`` steps back, or ``None`` until ``lookback + 1``
    values have been seen -- what a plain ``Series.diff(lookback)`` needs."""

    def __init__(self, lookback: int) -> None:
        # +1: the window holds `lookback` bars of history plus the current bar itself.
        self._max_len = lookback + 1
        self._buffer: deque[float] = deque(maxlen=self._max_len)

    def update(self, value: float) -> float | None:
        self._buffer.append(value)
        if len(self._buffer) < self._max_len:
            return None
        return self._buffer[0]


class EwmCovariance:
    """Streaming, pairwise-complete EWM covariance over several series at
    once -- the many-series generalization of ``EwmMoments``: a 1x1
    ``EwmCovariance`` is exactly an ``EwmMoments(span=...)`` (same bias-
    corrected weighted-variance recursion, same decay from span), restated as
    matrices so an N x N covariance costs one vectorized update instead of
    N^2 separate accumulators.

    A pair's accumulators only receive bars where both series are present,
    and every accumulator decays every bar -- matches
    ``DataFrame.ewm(span=..., adjust=True).cov()``. A late-listed series
    therefore enters the matrix exactly as pandas would, on the rows since it
    started."""

    def __init__(self, tickers: Sequence[Ticker], *, span: int, min_periods: int) -> None:
        if min_periods < 2:
            raise ValueError("min_periods must be >= 2")
        n = len(tickers)
        self.tickers: tuple[Ticker, ...] = tuple(tickers)
        self._index = {ticker: i for i, ticker in enumerate(self.tickers)}
        self._decay = 1.0 - _alpha_from_span(span)
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
