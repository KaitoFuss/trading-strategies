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
                ewm = self._vol.setdefault(
                    ticker, EwmMoments(span=_VOL_SPAN, min_periods=_VOL_SPAN)
                )
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
                OrderEvent(
                    timestamp=timestamp, ticker=ticker, quantity=abs(delta), direction=direction
                )
            )
        return orders
