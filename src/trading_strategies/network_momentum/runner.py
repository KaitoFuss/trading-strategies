"""Runs the MACD benchmark strategy (Poh, Wood, Roberts & Zohren 2023)
against a buy-and-hold reference, both through ``backtester``'s engine.

Mirrors ``backtester.runner.run_strategy_and_benchmark``'s shape, but for
``MacdBenchmarkStrategy`` / ``TargetVolPortfolio`` instead of that module's
own strategy and named portfolios -- ``TargetVolPortfolio`` lives in this
repo, not in ``backtester.portfolio.factory``, so it cannot go through
that function directly.
"""

from __future__ import annotations

from backtester.core.engine import Portfolio, PriceSource
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
from trading_strategies.portfolio.optimized import OptimizedPortfolio
from trading_strategies.risk.config import OptimizerConfig


def run_macd_benchmark(config: MacdBacktestConfig) -> dict[str, PerformanceTracker]:
    """The MACD benchmark strategy leg under ``config``, plus a fixed
    equal-weight buy-and-hold reference. Both legs pay the same trading
    costs (``config.cost_bps`` / ``config.commission_bps``) so the
    comparison is fair; the benchmark leg's own gross is fixed at 1.0
    regardless of ``config.max_gross``, matching
    ``backtester.runner.run_strategy_and_benchmark``'s treatment of its
    buy-and-hold leg as a passive reference, not a thing under test.

    ``config.portfolio`` switches the strategy leg's portfolio: the
    paper-faithful ``TargetVolPortfolio`` (per-asset Eq. 9 sizing only,
    ``"target_vol"``), ``RescaledTargetVolPortfolio`` (the same sizing, plus
    a final rescale so the book's *own* realized vol tracks ``target_vol``,
    ``"rescaled_target_vol"``), or ``OptimizedPortfolio`` (a constrained
    optimizer over the factor risk model, ``"optimized"``).
    """

    def build(price_source: PriceSource) -> Portfolio:
        if config.portfolio == "optimized":
            assert config.factor_model is not None  # enforced by MacdBacktestConfig
            return OptimizedPortfolio(
                price_source=price_source,
                initial_cash=config.initial_cash,
                target_vol=config.target_vol,
                max_gross=config.max_gross,
                model_config=config.factor_model,
                optimizer_config=config.optimizer or OptimizerConfig(),
                tickers=config.tickers,
            )
        portfolio_cls = (
            RescaledTargetVolPortfolio
            if config.portfolio == "rescaled_target_vol"
            else TargetVolPortfolio
        )
        return portfolio_cls(
            price_source=price_source,
            initial_cash=config.initial_cash,
            target_vol=config.target_vol,
            max_gross=config.max_gross,
        )

    strategy_tracker = run_backtest(
        MacdBenchmarkStrategy(),
        build,
        lambda portfolio: None,
        config,
    )
    benchmark_tracker = run_backtest(
        BuyAndHoldStrategy(),
        lambda price_source: EqualWeightPortfolio(
            price_source=price_source, initial_cash=config.initial_cash
        ),
        lambda portfolio: None,
        config,
    )
    return {"MACD Benchmark": strategy_tracker, "Buy & Hold": benchmark_tracker}
