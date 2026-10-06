from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from tests.synthetic_data import write_synthetic_parquet
from trading_strategies.config import MacdBacktestConfig
from trading_strategies.network_momentum.portfolio import RescaledTargetVolPortfolio
from trading_strategies.network_momentum.runner import run_macd_benchmark

_TICKERS = ("SPY", "QQQ", "TLT")
_NUM_BARS = 500


@pytest.fixture
def config(tmp_path: Path) -> MacdBacktestConfig:
    data_path = tmp_path / "raw.parquet"
    write_synthetic_parquet(data_path, _TICKERS, _NUM_BARS)
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


def test_run_macd_benchmark_uses_rescaled_portfolio_when_configured(
    config: MacdBacktestConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rescale_to_portfolio_vol=True`` must route through
    ``RescaledTargetVolPortfolio`` instead of the plain ``TargetVolPortfolio``
    -- checked by spying on the constructor, since the two portfolios'
    trades are otherwise hard to tell apart from the outside on a short
    synthetic run."""
    seen: list[type] = []
    original_init = RescaledTargetVolPortfolio.__init__

    def _spy_init(self: RescaledTargetVolPortfolio, *args: object, **kwargs: object) -> None:
        seen.append(type(self))
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(RescaledTargetVolPortfolio, "__init__", _spy_init)

    rescaled_config = replace(config, rescale_to_portfolio_vol=True)
    run_macd_benchmark(rescaled_config)

    assert seen == [RescaledTargetVolPortfolio]


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
