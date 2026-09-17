"""Runs the MACD benchmark strategy (Poh, Wood, Roberts & Zohren 2023)
against a buy-and-hold reference, both through ``backtester``'s engine.

Mirrors ``backtester.runner.run_strategy_and_benchmark``'s shape, but for
``MacdBenchmarkStrategy`` / ``TargetVolPortfolio`` instead of that module's
own strategy and named portfolios -- ``TargetVolPortfolio`` lives in this
repo, not in ``backtester.portfolio.factory``, so it cannot go through
that function directly.
"""

from __future__ import annotations

from backtester.portfolio.equal_weight import EqualWeightPortfolio
from backtester.runner import run_backtest
from backtester.strategy.buy_and_hold import BuyAndHoldStrategy
from backtester.tracker.metrics import PerformanceTracker

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.network_momentum.portfolio import (
    RescaledTargetVolPortfolio,
    TargetVolPortfolio,
)
from trading_strategies.network_momentum.strategies import MacdBenchmarkStrategy


def run_macd_benchmark(config: MacdBacktestConfig) -> dict[str, PerformanceTracker]:
    """The MACD benchmark strategy leg under ``config``, plus a fixed
    equal-weight buy-and-hold reference. Both legs pay the same trading
    costs (``config.cost_bps`` / ``config.commission_bps``) so the
    comparison is fair; the benchmark leg's own gross is fixed at 1.0
    regardless of ``config.max_gross``, matching
    ``backtester.runner.run_strategy_and_benchmark``'s treatment of its
    buy-and-hold leg as a passive reference, not a thing under test.

    ``config.rescale_to_portfolio_vol`` switches the strategy leg's
    portfolio from the paper-faithful ``TargetVolPortfolio`` (per-asset Eq.
    9 sizing only) to ``RescaledTargetVolPortfolio`` (the same sizing, plus
    a final rescale so the book's *own* realized vol tracks
    ``target_vol``).
    """
    backtest_config = config.to_backtest_config()
    portfolio_cls = (
        RescaledTargetVolPortfolio if config.rescale_to_portfolio_vol else TargetVolPortfolio
    )

    strategy_tracker = run_backtest(
        MacdBenchmarkStrategy(),
        lambda price_source: portfolio_cls(
            price_source=price_source,
            initial_cash=config.initial_cash,
            target_vol=config.target_vol,
            max_gross=config.max_gross,
        ),
        lambda portfolio: None,
        backtest_config,
    )
    benchmark_tracker = run_backtest(
        BuyAndHoldStrategy(),
        lambda price_source: EqualWeightPortfolio(
            price_source=price_source, initial_cash=config.initial_cash
        ),
        lambda portfolio: None,
        backtest_config,
    )
    return {"MACD Benchmark": strategy_tracker, "Buy & Hold": benchmark_tracker}
