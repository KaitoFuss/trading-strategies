"""Strategies from Poh, Wood, Roberts & Zohren (2023), "Network Momentum
across Asset Classes" (arXiv:2308.11294)."""

from __future__ import annotations

import logging

from backtester.core.events import MarketEvent, SignalEvent, Ticker

from trading_strategies.utils.features import MACD_PAIRS, StreamingMacd

logger = logging.getLogger(__name__)


class MacdBenchmarkStrategy:
    """The paper's MACD benchmark (Eq. 10, Section 4.1): the mean of the
    response-function-squashed, normalized MACD indicator over the three
    (short, long) time-scale pairs in ``MACD_PAIRS``.

    No winsorization here -- that's a Section 2.2 preprocessing step for the
    GMOM regression's input features, not part of this benchmark's own
    definition.
    """

    def __init__(self) -> None:
        self._macd: dict[Ticker, tuple[StreamingMacd, ...]] = {}

    def process_market(self, event: MarketEvent) -> SignalEvent:
        scores: dict[Ticker, float] = {}
        for ticker, bar in event.bars.items():
            states = self._macd.setdefault(
                ticker,
                tuple(
                    StreamingMacd(short, long, apply_phi=True, winsorize=False)
                    for short, long in MACD_PAIRS
                ),
            )
            values = [state.update(bar.close) for state in states]
            if all(value is not None for value in values):
                scores[ticker] = sum(values) / len(values)  # type: ignore[arg-type]
            else:
                logger.debug("%s  %s: warming up MACD pairs", event.timestamp, ticker)
        return SignalEvent(timestamp=event.timestamp, scores=scores)
