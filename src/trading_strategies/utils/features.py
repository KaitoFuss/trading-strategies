"""Streaming momentum/MACD features from Poh, Lim, Zohren & Roberts (2022)
and Poh, Wood, Roberts & Zohren (2023) / network-momentum.

Each streaming class consumes one price at a time, in date order, and
returns that day's feature value (or ``None`` while still warming up) --
the shape a live ``Strategy.process_market`` needs, one bar per call.
"""

from __future__ import annotations

import math

from trading_strategies.utils.streaming import EwmMoments

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
