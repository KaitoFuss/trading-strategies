"""Run NetworkMomentumStrategy + MeanVariancePortfolio against locally fetched
data, alongside an equal-weight buy-and-hold benchmark, and write a report.

Usage:
    uv run scripts/run_network_momentum.py configs/network_momentum.json -v
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backtester.portfolio.equal_weight import EqualWeightPortfolio
from backtester.runner import run_backtest, verbosity_to_level
from backtester.strategy.buy_and_hold import BuyAndHoldStrategy
from backtester.tracker.metrics import monthly_returns_table, strategy_correlation_matrix
from backtester.tracker.report import save_report

from trading_strategies.config import NetworkMomentumConfig
from trading_strategies.factor_risk import FactorRiskModel
from trading_strategies.portfolio.mean_variance import MeanVariancePortfolio
from trading_strategies.strategy.network_momentum import NetworkMomentumStrategy

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="path to a NetworkMomentumConfig JSON")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v for the trade blotter (INFO), -vv for the full numeric trail (DEBUG)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=verbosity_to_level(args.verbose),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    config = NetworkMomentumConfig.from_json(args.config)
    backtest_config = config.to_backtest_config()

    logger.info("Running network-momentum backtest on %s …", config.data)
    risk_model = FactorRiskModel(halflife=config.halflife)
    strategy_tracker = run_backtest(
        NetworkMomentumStrategy(halflife=config.halflife, risk_model=risk_model),
        lambda price_source: MeanVariancePortfolio(
            price_source=price_source,
            initial_cash=config.initial_cash,
            max_gross=config.max_gross,
            risk_aversion=config.risk_aversion,
            min_periods=config.min_periods,
            drift_band=config.drift_band,
            risk_model=risk_model,
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
    trackers = {"Strategy": strategy_tracker, "Buy & Hold": benchmark_tracker}
    histories = {label: tracker.mark_to_market_history for label, tracker in trackers.items()}

    report_path = save_report(
        output_dir=Path(config.output_dir) / "network_momentum",
        histories=histories,
        metrics={label: tracker.metrics() for label, tracker in trackers.items()},
        trade_metrics={label: tracker.trade_metrics() for label, tracker in trackers.items()},
        monthly_tables={
            label: monthly_returns_table(history) for label, history in histories.items()
        },
        correlation=strategy_correlation_matrix(histories),
        config=backtest_config,
    )
    logger.warning("Report written to %s", report_path)


if __name__ == "__main__":
    main()
