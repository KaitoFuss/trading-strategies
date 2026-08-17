import math
from collections import deque

from backtester.stats import mean_and_stdev

from trading_strategies.features.momentum_features import MACD_PAIRS, VOL_SCALED_RETURN_DELTAS


class RecursiveEma:
    """m(t, J) = EWMA(price, alpha=1/J), recursive form (pandas adjust=False).

    m_t = alpha*x_t + (1-alpha)*m_{t-1}, seeded by the first observation.
    Exact by construction, O(1) per update, no history retained.
    """

    def __init__(self, j: int) -> None:
        self._alpha = 1.0 / j
        self._value: float | None = None

    def update(self, price: float) -> float:
        self._value = (
            price if self._value is None else self._alpha * price + (1 - self._alpha) * self._value
        )
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


class RecursiveEwVariance:
    """Exponentially-weighted variance, recursive (Finch's incremental form).

    Not bit-identical to pandas' ewm(span=...).std() (that uses a different
    bias-correction/initialization), but converges closely once well past
    the span's effective memory — verified by an equivalence test rather
    than assumed. None until a second observation exists, matching
    ``backtester.stats.mean_and_stdev``'s own convention that a single point
    has no defined variance.
    """

    def __init__(self, span: int) -> None:
        self._alpha = 2.0 / (span + 1)
        self._mean: float | None = None
        self._variance = 0.0
        self._n = 0

    def update(self, x: float) -> float | None:
        self._n += 1
        if self._mean is None:
            self._mean = x
            return None
        diff = x - self._mean
        incr = self._alpha * diff
        self._mean += incr
        self._variance = (1 - self._alpha) * (self._variance + diff * incr)
        return self.std

    @property
    def std(self) -> float | None:
        if self._n < 2:
            return None
        return math.sqrt(self._variance)


_EMA_SPANS = sorted({j for pair in MACD_PAIRS for j in pair})
_MAX_DELTA = max(VOL_SCALED_RETURN_DELTAS)
_PRICE_STD_WINDOW = 63
_Q_STD_WINDOW = 252


class OnlineTickerFeatures:
    """Incrementally maintained 8-feature loading (B) for one ticker.

    Same formulas as ``momentum_features.py``, computed bar by bar with O(1)
    or O(window) state instead of recomputing over the whole growing price
    series each call — the pandas version is too slow at real backtest
    scale (recompute-per-bar over a growing series is effectively O(T^2)).
    Validated against ``momentum_features.py`` via an equivalence test
    rather than assumed correct from the formulas alone.
    """

    def __init__(self) -> None:
        self._price_history: deque[float] = deque(maxlen=_MAX_DELTA + 1)
        self._price_std_window: deque[float] = deque(maxlen=_PRICE_STD_WINDOW)
        self._emas = {j: RecursiveEma(j=j) for j in _EMA_SPANS}
        self._sigma = RecursiveEwVariance(span=60)
        self._q_std_windows: dict[tuple[int, int], deque[float]] = {
            pair: deque(maxlen=_Q_STD_WINDOW) for pair in MACD_PAIRS
        }
        self._last_price: float | None = None

    def update(self, price: float) -> list[float] | None:
        """Feed today's close; return today's 8 loadings, or None while any
        of them is still short of its warm-up."""
        sigma = None
        if self._last_price is not None:
            daily_return = price / self._last_price - 1
            self._sigma.update(daily_return)
            sigma = self._sigma.std
        self._last_price = price

        self._price_history.append(price)
        self._price_std_window.append(price)
        for ema in self._emas.values():
            ema.update(price)

        vol_scaled = self._vol_scaled_returns(sigma)
        macd = self._macd_features()
        if vol_scaled is None or macd is None:
            return None
        return [*vol_scaled, *macd]

    def _vol_scaled_returns(self, sigma: float | None) -> list[float] | None:
        if not sigma:
            return None
        current_price = self._price_history[-1]
        features = []
        for delta in VOL_SCALED_RETURN_DELTAS:
            if len(self._price_history) <= delta:
                return None
            past_price = self._price_history[-(delta + 1)]
            window_return = current_price / past_price - 1
            features.append(window_return / (sigma * math.sqrt(delta)))
        return features

    def _macd_features(self) -> list[float] | None:
        if len(self._price_std_window) < _PRICE_STD_WINDOW:
            return None
        _, price_std = mean_and_stdev(list(self._price_std_window))
        if not price_std:
            return None

        # Append to every pair's window unconditionally before checking any
        # of them for completeness — an early return here would starve later
        # pairs of an append on this call, staggering their warm-up relative
        # to earlier ones instead of all three filling in lockstep.
        qs = {}
        for short, long in MACD_PAIRS:
            short_ema, long_ema = self._emas[short].value, self._emas[long].value
            assert short_ema is not None
            assert long_ema is not None
            q = (short_ema - long_ema) / price_std
            self._q_std_windows[(short, long)].append(q)
            qs[(short, long)] = q

        features = []
        for pair, q in qs.items():
            window = self._q_std_windows[pair]
            if len(window) < _Q_STD_WINDOW:
                return None
            _, q_std = mean_and_stdev(list(window))
            if not q_std:
                return None
            features.append(max(-5.0, min(5.0, q / q_std)))
        return features
