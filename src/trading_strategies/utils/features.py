"""Streaming momentum/MACD features from Poh, Lim, Zohren & Roberts (2022)
and Poh, Wood, Roberts & Zohren (2023) / network-momentum.

Each streaming class consumes one price at a time, in date order, and
returns that day's feature value (or ``None`` while still warming up) --
the shape a live ``Strategy.process_market`` needs, one bar per call.
"""

from __future__ import annotations

import math
from typing import Protocol

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


class MomentsSource(Protocol):
    """What ``_Winsorizer`` needs from its mean/std source -- satisfied
    structurally by ``EwmMoments`` (no inheritance required)."""

    def update(self, value: float) -> None: ...

    @property
    def mean(self) -> float | None: ...

    @property
    def std(self) -> float | None: ...


class _Winsorizer:
    """Clips a value to +/- z stdevs around its own mean, both drawn from
    ``source`` (updated once per ``apply()`` call). Passes the raw value
    through unchanged while ``source`` is still warming up --
    ``Series.clip(lower=NaN, upper=NaN)`` is a no-op in pandas, not NaN, so
    this adds no warmup delay of its own."""

    def __init__(self, source: MomentsSource, z: float = WINSOR_Z) -> None:
        self._source = source
        self._z = z

    def apply(self, value: float) -> float:
        self._source.update(value)
        mean, std = self._source.mean, self._source.std
        if mean is None or std is None:
            return value
        return min(max(value, mean - self._z * std), mean + self._z * std)


def _finalize(raw: float, apply_phi: bool, winsorizer: _Winsorizer | None) -> float:
    """Shared tail of every feature's ``update``: optionally squash through
    ``response_function``, then optionally winsorize."""
    value = response_function(raw) if apply_phi else raw
    return winsorizer.apply(value) if winsorizer is not None else value


class Feature(Protocol):
    """A single streaming feature: consumes one price at a time and returns
    its value under its own ``name`` (or ``None`` while warming up).
    Satisfied structurally by ``StreamingMacd`` and
    ``StreamingVolScaledMomentum`` -- ``name`` is what ``StreamingFeatureSet``
    keys its output dict by."""

    name: str

    def update(self, close: float) -> float | None: ...


class FeatureSet(Protocol):
    """Every feature for one ticker, one price at a time, keyed by each
    feature's own ``name``."""

    def update(self, close: float) -> dict[str, float]: ...


class StreamingMacd:
    """Normalized MACD signal (Baz et al. 2015), one price at a time.
    ``short``/``long`` are the two EMA spans of one (S, L) pair from
    ``MACD_PAIRS``."""

    def __init__(
        self, short: int, long: int, apply_phi: bool = False, winsorize: bool = True
    ) -> None:
        self.name = f"macd_{short}_{long}"
        self._short = EwmMean(span=short, min_periods=short)
        self._long = EwmMean(span=long, min_periods=long)
        self._price_std = RollingStd(63)
        self._macd_std = RollingStd(252)
        self._apply_phi = apply_phi
        self._winsorizer = _Winsorizer(EwmMoments(halflife=WINSOR_HALFLIFE)) if winsorize else None

    def update(self, close: float) -> float | None:
        m_short = self._short.update(close)
        m_long = self._long.update(close)
        price_std = self._price_std.update(close)

        q = None
        # price_std/macd_std > 0 (not just "is not None") guards the division below --
        # a std of exactly 0.0 (e.g. a flat price stretch) is a ready value, not a
        # warm-up state, but dividing by it would raise ZeroDivisionError.
        if m_short is not None and m_long is not None and price_std is not None and price_std > 0:
            q = (m_short - m_long) / price_std
        if q is None:
            return None

        macd_std = self._macd_std.update(q)
        if macd_std is None or macd_std <= 0:
            return None

        normalized = q / macd_std
        return _finalize(normalized, self._apply_phi, self._winsorizer)


class StreamingVolScaledMomentum:
    """Cumulative log return over ``lookback`` days, scaled by EWM vol and
    sqrt(lookback), one price at a time."""

    def __init__(self, lookback: int, apply_phi: bool = False, winsorize: bool = True) -> None:
        self.name = f"mom_{lookback}"
        self._lookback = lookback
        self._apply_phi = apply_phi
        self._vol = EwmMoments(span=VOL_SPAN, min_periods=VOL_SPAN)
        self._lag = RollingLag(lookback)
        self._winsorizer = _Winsorizer(EwmMoments(halflife=WINSOR_HALFLIFE)) if winsorize else None
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
        return _finalize(raw, self._apply_phi, self._winsorizer)


class StreamingFeatureSet:
    """All 8 momentum/MACD features for one ticker, one price at a time.
    ``momentum_phi``/``macd_phi`` toggle the Baz et al. response-function
    squashing for each feature family, independently, matching today's
    ``compute_all_features`` toggles."""

    def __init__(self, momentum_phi: bool = False, macd_phi: bool = False) -> None:
        features: list[Feature] = [
            StreamingVolScaledMomentum(lookback, apply_phi=momentum_phi)
            for lookback in MOMENTUM_LOOKBACKS
        ]
        features += [StreamingMacd(short, long, apply_phi=macd_phi) for short, long in MACD_PAIRS]
        self._features: dict[str, Feature] = {feature.name: feature for feature in features}

    def update(self, close: float) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, feature in self._features.items():
            value = feature.update(close)
            if value is not None:
                values[name] = value
        return values
