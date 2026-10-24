"""Portfolio that sizes positions with a constrained optimizer instead of the
paper's per-asset vol scaling.

Runs the factor risk model itself (one ``update`` per bar, before optimizing,
mirroring ``TargetVolPortfolio._update_vol``), turns the strategy's scores into
alphas, and asks ``solve_weights`` for the target weights. Stays flat while the
model warms up, and keeps its previous weights whenever the solver fails.

Known sticky-price caveat, inherited from ``TargetVolPortfolio``: ``get_price``
returns the last seen price for a ticker missing from the current bar, which
feeds a spurious zero return into the model for that ticker. Documented here,
not fixed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
from backtester.core.engine import PriceSource
from backtester.core.events import OrderEvent, SignalEvent, Ticker
from backtester.core.trade_log import log_trade
from backtester.portfolio.base import BasePortfolio

from trading_strategies.portfolio.solver import OptimizerParams, solve_weights
from trading_strategies.risk.config import FactorModelConfig, OptimizerConfig
from trading_strategies.risk.model import FactorRiskModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolveLogEntry:
    timestamp: datetime
    status: str
    binding: str
    ex_ante_vol: float
    gross: float
    turnover: float


class OptimizedPortfolio(BasePortfolio):
    def __init__(
        self,
        price_source: PriceSource,
        *,
        initial_cash: float = 100_000.0,
        target_vol: float = 0.15,
        max_gross: float = 1.0,
        model_config: FactorModelConfig,
        optimizer_config: OptimizerConfig,
        tickers: Sequence[Ticker],
    ) -> None:
        super().__init__(price_source=price_source, initial_cash=initial_cash, max_gross=max_gross)
        self._tickers = tuple(tickers)
        self._model = FactorRiskModel(self._tickers, model_config)
        self._params = OptimizerParams(
            target_vol=target_vol,
            max_gross=max_gross,
            information_coefficient=optimizer_config.information_coefficient,
            turnover_penalty_bps=optimizer_config.turnover_penalty_bps,
            factor_bounds=optimizer_config.factor_bounds,
        )
        self.solve_log: list[SolveLogEntry] = []

    def process_signal(self, event: SignalEvent) -> Sequence[OrderEvent]:
        # The engine only ever calls process_signal / process_fill on a
        # portfolio (see backtester.core.engine.Engine._dispatch), so the model
        # is updated from the price source here, exactly once per bar -- the
        # same route TargetVolPortfolio._update_vol takes.
        self._update_model()
        snapshot = self._model.snapshot
        equity = self.mark_to_market()
        if not snapshot.ready or equity <= 0:
            logger.debug("%s: model not ready, staying flat", event.timestamp)
            return []

        w_prev = np.array(
            [self._current_weight(t, equity) for t in snapshot.tickers], dtype=np.float64
        )
        result = solve_weights(event.scores, snapshot, w_prev, self._params)
        if result.status != "optimal":
            logger.warning("%s: solver status %s, keeping weights", event.timestamp, result.status)
        self.solve_log.append(
            SolveLogEntry(
                timestamp=event.timestamp,
                status=result.status,
                binding=result.binding,
                ex_ante_vol=result.ex_ante_vol,
                gross=result.gross,
                turnover=float(np.abs(result.weights - w_prev).sum()),
            )
        )
        if result.status != "optimal":
            return []
        weights = dict(zip(snapshot.tickers, result.weights.tolist(), strict=True))
        return self._orders_from_targets(weights, equity, event.timestamp)

    def _update_model(self) -> None:
        closes = {
            ticker: price
            for ticker in self._tickers
            if (price := self._price_source.get_price(ticker)) is not None
        }
        self._model.update(closes)

    def _current_weight(self, ticker: Ticker, equity: float) -> float:
        position = self._positions.get(ticker)
        price = self._price_source.get_price(ticker)
        if position is None or price is None:
            return 0.0
        return position.quantity * price / equity

    def _orders_from_targets(
        self, weights: Mapping[Ticker, float], equity: float, timestamp: datetime
    ) -> list[OrderEvent]:
        orders: list[OrderEvent] = []
        for ticker, weight in weights.items():
            price = self._price_source.get_price(ticker)
            if price is None:
                continue
            position = self._positions.get(ticker)
            current = position.quantity if position else 0
            delta = round(weight * equity / price) - current
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
