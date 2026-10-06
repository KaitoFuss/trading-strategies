"""Run the MACD benchmark strategy against a buy-and-hold benchmark and
write a PDF performance report.

Usage:
    uv run scripts/run_backtest.py configs/backtest_macd.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from backtester.tracker.metrics import monthly_returns_table, strategy_correlation_matrix
from backtester.tracker.report import ReportPage, save_report

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.network_momentum.runner import run_macd_benchmark
from trading_strategies.risk.attribution import attribute_backtest
from trading_strategies.risk.report_pages import attribution_pages

logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_backtest.py <config.json>")
    config = MacdBacktestConfig.from_json(Path(sys.argv[1]))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )

    trackers = run_macd_benchmark(config)
    histories = {label: tracker.mark_to_market_history for label, tracker in trackers.items()}
    metrics = {label: tracker.metrics() for label, tracker in trackers.items()}
    trade_metrics = {label: tracker.trade_metrics() for label, tracker in trackers.items()}
    monthly_tables = {label: monthly_returns_table(history) for label, history in histories.items()}
    correlation = strategy_correlation_matrix(histories)

    extra_pages: list[ReportPage] = []
    if config.factor_model is not None:
        results = attribute_backtest(
            Path(config.data), config.tickers, config.factor_model, trackers
        )
        for label, result in results.items():
            logger.info("%s: factor-model bias statistic %.2f", label, result.bias_statistic())
        extra_pages = attribution_pages(results)

    report_path = save_report(
        Path(config.output_dir),
        histories,
        metrics,
        trade_metrics,
        monthly_tables,
        correlation,
        extra_pages=extra_pages,
        config=config,
    )
    logger.info("Done — wrote %s", report_path)


if __name__ == "__main__":
    main()
