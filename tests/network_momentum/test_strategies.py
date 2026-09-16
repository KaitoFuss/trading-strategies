from datetime import datetime, timedelta

import numpy as np
import pytest
from backtester.core.events import Bar, MarketEvent

from trading_strategies.network_momentum.strategies import MacdBenchmarkStrategy
from trading_strategies.utils.features import MACD_PAIRS, StreamingMacd


def _market_events(closes: dict[str, list[float]]) -> list[MarketEvent]:
    n = len(next(iter(closes.values())))
    start = datetime(2020, 1, 1)
    events = []
    for i in range(n):
        bars = {ticker: Bar(close=series[i]) for ticker, series in closes.items()}
        events.append(MarketEvent(timestamp=start + timedelta(days=i), bars=bars))
    return events


def test_macd_benchmark_strategy_scores_nothing_while_warming_up() -> None:
    rng = np.random.default_rng(12)
    closes = {"SPY": [float(x) for x in 100 + np.cumsum(rng.normal(0, 1, 5))]}
    strategy = MacdBenchmarkStrategy()

    for event in _market_events(closes):
        signal = strategy.process_market(event)

    assert signal.scores == {}


def test_macd_benchmark_strategy_matches_mean_of_three_macd_pairs() -> None:
    rng = np.random.default_rng(13)
    n = 400
    series = [float(x) for x in 100 + np.cumsum(rng.normal(0, 1, n))]
    closes = {"SPY": series}

    strategy = MacdBenchmarkStrategy()
    reference = [
        StreamingMacd(short, long, apply_phi=True, winsorize=False) for short, long in MACD_PAIRS
    ]

    last_score = None
    for i, event in enumerate(_market_events(closes)):
        signal = strategy.process_market(event)
        expected_parts = [state.update(series[i]) for state in reference]
        if all(p is not None for p in expected_parts):
            expected = sum(expected_parts) / len(expected_parts)  # type: ignore[arg-type]
            assert signal.scores["SPY"] == pytest.approx(expected, abs=1e-9)
            last_score = expected
        else:
            assert "SPY" not in signal.scores

    assert last_score is not None
