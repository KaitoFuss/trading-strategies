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
    constructor-signature consistency with ``BasePortfolio`` and the other
    portfolio implementations but is not used to scale anything.

    Warm-up is compounded, not additive: this Portfolio only sees a ticker
    once its ``Strategy`` starts scoring it (e.g. ``MacdBenchmarkStrategy``
    needs ~347 bars for all 3 MACD pairs to warm), and only from that point
    does this Portfolio's own ``EwmMoments(span=60, min_periods=60)`` start
    counting toward its own ~60-bar warm-up -- so the first trade lands
    around bar ~410, not ~350. This is a consequence of the deliberate
    Strategy/Portfolio decoupling, not a bug.

    A held position that stops being scored, loses its price, or hits a
    zero vol estimate is dropped from the weight dict in
    ``_target_weights``/``_orders_from_targets`` and is therefore left
    completely untouched -- never rebalanced or exited -- until end-of-run
    liquidation. This mirrors ``ScoreProportionalPortfolio``'s documented
    behavior.

    Known risk: ``_update_vol`` reads prices via ``PriceSource.get_price``,
    which returns a sticky last price for a ticker missing from a bar
    (e.g. a holiday misalignment or a data gap). That injects a spurious
    zero return into the vol EWM, biasing ``sigma_i`` downward -- and
    because this Portfolio's gross is uncapped with no ``max_gross``
    renormalization backstop (unlike ``InverseVolPortfolio``), a
    downward-biased ``sigma_i`` directly inflates leverage with no safety
    net. Low probability, but worth knowing before feeding this real data
    with gaps.
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


class RescaledTargetVolPortfolio(TargetVolPortfolio):
    """``TargetVolPortfolio`` with one extra step: the per-asset Eq. 9
    weights are rescaled by ``target_vol / portfolio_vol``, where
    ``portfolio_vol`` is this portfolio's own trailing annualized vol (an
    ``EwmMoments(span=60, min_periods=60)`` on its realized daily equity
    returns). Unlike the base class, ``target_vol`` here is a genuine
    portfolio-level target: Eq. 9 alone only guarantees that the *sum* of
    each asset's own vol contribution equals ``target_vol``, which is far
    above the realized portfolio vol once diversification/correlation
    across many assets is accounted for -- this rescale corrects for that.

    Warm-up sample gating: a bar where this portfolio holds no position
    (``self._positions`` empty) produces a mechanically zero equity
    return -- nothing was held for a price move to act on, not a genuine
    zero-vol observation. Such bars are excluded from the ``portfolio_vol``
    EWM entirely, rather than counted as evidence of near-zero vol; without
    this, the EWM would warm up during the strategy's own pre-trade
    warm-up (all flat, all zero-return) and the first live rescale would
    divide by a near-zero vol, spiking leverage. Until ``portfolio_vol`` is
    warm, weights pass through unscaled -- identical to the base class --
    which in practice means the first ~60 bars after this portfolio's first
    real position match ``TargetVolPortfolio`` exactly.

    Known feedback property, not a bug: once warm, the rescale is driven by
    the vol of this portfolio's own past *rescaled* returns, not an
    independent signal. This is the standard, pro-cyclical dynamic of every
    realized-vol-targeting overlay (a calm stretch understates vol, sizing
    up leverage right before a regime shift) -- documented here so it
    isn't mistaken for an oversight.
    """

    def __init__(
        self,
        price_source: PriceSource,
        initial_cash: float = 100_000.0,
        target_vol: float = 0.15,
        max_gross: float = 1.0,
    ) -> None:
        super().__init__(
            price_source=price_source,
            initial_cash=initial_cash,
            target_vol=target_vol,
            max_gross=max_gross,
        )
        self._portfolio_vol = EwmMoments(span=_VOL_SPAN, min_periods=_VOL_SPAN)
        self._last_equity: float | None = None

    def _annualized_portfolio_vol(self) -> float | None:
        if not self._portfolio_vol.std:
            return None
        return self._portfolio_vol.std * math.sqrt(TRADING_DAYS_PER_YEAR)

    def process_signal(self, event: SignalEvent) -> Sequence[OrderEvent]:
        self._update_vol(set(event.scores) | set(self._positions), event.timestamp)
        equity = self.mark_to_market()
        if equity <= 0:
            return []

        if self._positions and self._last_equity is not None:
            self._portfolio_vol.update(math.log(equity / self._last_equity))
        self._last_equity = equity

        weights = self._target_weights(event)
        portfolio_vol = self._annualized_portfolio_vol()
        if portfolio_vol:
            scale = self._target_vol / portfolio_vol
            weights = {ticker: weight * scale for ticker, weight in weights.items()}
        return self._orders_from_targets(weights, equity, event.timestamp)
