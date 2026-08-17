import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import numpy as np
from backtester.core.engine import PriceSource
from backtester.core.events import OrderEvent, SignalEvent, Ticker
from backtester.core.trade_log import log_trade
from backtester.portfolio.base import BasePortfolio, existing_gross

from trading_strategies.factor_risk import FactorRiskModel
from trading_strategies.optimize import mean_variance_weights

logger = logging.getLogger(__name__)


class MeanVariancePortfolio(BasePortfolio):
    """Classic Grinold-Kahn mean-variance optimizer:
    max_w w'scores - (risk_aversion/2) w'Sigma w, s.t. sum(w)=0, ||w||_1<=budget.

    ``scores`` (the SignalEvent) is each ticker's expected return, B.mu_f --
    this portfolio never needs B or mu_f separately, only that one number.
    Sigma comes from the shared ``FactorRiskModel`` as the structural
    ``B Sigma_f B' + D`` -- positive-definite by construction (D is the
    eigenvalue floor), and far better conditioned than any sample-covariance
    estimate of the 26-name universe, so no statistical shrinkage is needed.
    The strategy updates that model with this bar's factor structure before
    this method reads it (backtester runs process_market -> process_signal).

    A ticker absent from today's optimization set (too little price history
    yet, or absent from ``scores``) is held untouched, not force-closed --
    same "flat/held is a valid state, not touched" rule every other portfolio
    here follows. Its gross is reserved via ``existing_gross`` so new opens
    only size into what's actually left of the budget.

    ``drift_band`` is the no-trade region around each ticker's target
    weight, same mechanism as ``ScoreProportionalPortfolio``: solving the QP
    fresh every bar with no dampening means day-to-day noise in scores/Sigma
    gets traded on constantly, which is exactly what an empirical run
    showed (turnover ~758x/yr even with costs off) -- this is the lever
    against it.
    """

    def __init__(
        self,
        price_source: PriceSource,
        initial_cash: float = 100_000.0,
        max_gross: float = 1.0,
        risk_aversion: float = 1.0,
        min_periods: int = 60,
        drift_band: float = 0.0,
        *,
        risk_model: FactorRiskModel,
    ) -> None:
        super().__init__(price_source=price_source, initial_cash=initial_cash, max_gross=max_gross)
        self._risk_aversion = risk_aversion
        self._min_periods = min_periods
        self._drift_band = drift_band
        self._risk_model = risk_model
        self._return_counts: dict[Ticker, int] = {}
        self._last_price: dict[Ticker, float] = {}

    def process_signal(self, event: SignalEvent) -> Sequence[OrderEvent]:
        self._record_return_counts(set(event.scores) | set(self._positions))
        equity = self.mark_to_market()
        if equity <= 0:
            return []

        tickers = [
            ticker
            for ticker in event.scores
            if self._return_counts.get(ticker, 0) >= self._min_periods
        ]
        if len(tickers) < 2:
            return []

        held_elsewhere = set(self._positions) - set(tickers)
        reserved = existing_gross(
            {ticker: self._positions[ticker] for ticker in held_elsewhere},
            self._price_source,
            equity,
        )
        budget = max(0.0, self._max_gross - reserved)
        if reserved > 0:
            logger.info(
                "%s  Gross budget reduced to %.4f (max_gross=%.4f) by unscored held positions",
                event.timestamp,
                budget,
                self._max_gross,
            )

        covariance = self._risk_model.covariance(tickers)
        expected_returns = np.array([event.scores[ticker] for ticker in tickers])
        weights = mean_variance_weights(
            expected_returns, covariance, risk_aversion=self._risk_aversion, gross_cap=budget
        )

        targets = dict(zip(tickers, weights, strict=True))
        return self._orders_from_targets(targets, equity, event.timestamp)

    def _record_return_counts(self, tickers: set[Ticker]) -> None:
        """Count how many consecutive-price return observations each ticker has
        seen -- the ``min_periods`` warm-up gate below is the only consumer, so
        only the count matters, not the return values (the covariance now comes
        from the factor risk model, not a sample of these returns)."""
        for ticker in tickers:
            price = self._price_source.get_price(ticker)
            if price is None:
                continue
            prev = self._last_price.get(ticker)
            self._last_price[ticker] = price
            if prev is not None:
                self._return_counts[ticker] = self._return_counts.get(ticker, 0) + 1

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
            current_weight = current_qty * price / equity
            if abs(weight - current_weight) < self._drift_band:
                continue

            delta = round(weight * equity / price) - current_qty
            if delta == 0:
                continue

            direction: Literal["BUY", "SELL"] = "BUY" if delta > 0 else "SELL"
            log_trade(
                logger,
                timestamp,
                "REBALANCE" if current_qty != 0 else "OPEN",
                direction,
                ticker,
                abs(delta),
                price,
                f"weight={weight:.5f} qty {current_qty} -> {current_qty + delta}",
            )
            orders.append(
                OrderEvent(
                    timestamp=timestamp,
                    ticker=ticker,
                    quantity=abs(delta),
                    direction=direction,
                )
            )
        return orders
