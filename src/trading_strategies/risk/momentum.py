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
