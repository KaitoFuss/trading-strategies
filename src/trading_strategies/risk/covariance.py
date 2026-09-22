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
