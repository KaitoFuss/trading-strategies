from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.network_momentum.runner import run_macd_benchmark

_TICKERS = ("SPY", "QQQ", "TLT")
_NUM_BARS = 500


def _write_synthetic_parquet(path: Path) -> None:
    rng = np.random.default_rng(7)
    start = datetime(2020, 1, 1)
    rows: list[dict[str, object]] = []
    for ticker in _TICKERS:
        closes = 100 + np.cumsum(rng.normal(0, 1, _NUM_BARS))
        for i, close in enumerate(closes):
            rows.append(
                {
                    "date": start + timedelta(days=i),
                    "ticker": ticker,
                    "close": float(close),
                }
            )
    pd.DataFrame(rows).to_parquet(path)


@pytest.fixture
def config(tmp_path: Path) -> MacdBacktestConfig:
    data_path = tmp_path / "raw.parquet"
    _write_synthetic_parquet(data_path)
    return MacdBacktestConfig(
        name="MACD Benchmark",
        data=str(data_path),
        tickers=list(_TICKERS),
        target_vol=0.15,
        initial_cash=100_000.0,
        output_dir=str(tmp_path / "output"),
    )


def test_run_macd_benchmark_returns_both_legs(config: MacdBacktestConfig) -> None:
    trackers = run_macd_benchmark(config)

    assert set(trackers) == {"MACD Benchmark", "Buy & Hold"}
    for tracker in trackers.values():
        assert len(tracker.mark_to_market_history) == _NUM_BARS
        metrics = tracker.metrics()
        assert not math.isnan(metrics.total_return)


def test_run_macd_benchmark_both_legs_actually_trade(config: MacdBacktestConfig) -> None:
    """With cost_bps=commission_bps=0.0, equity can only move away from
    initial_cash by holding a position through a price change -- so a flat
    final-vs-initial equity for the whole run would mean the strategy never
    actually traded, which is the wiring bug this test exists to catch
    (e.g. wrong price source, tickers never reaching the portfolio)."""
    trackers = run_macd_benchmark(config)

    for label, tracker in trackers.items():
        final_equity = tracker.mark_to_market_history[-1][1]
        assert final_equity != pytest.approx(config.initial_cash), f"{label} leg never traded"
        assert tracker.trade_metrics().num_trades > 0, f"{label} leg recorded no trades"
