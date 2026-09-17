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
from backtester.tracker.report import save_report

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.network_momentum.runner import run_macd_benchmark

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

    report_path = save_report(
        Path(config.output_dir),
        histories,
        metrics,
        trade_metrics,
        monthly_tables,
        correlation,
        config=config,
    )
    logger.info("Done — wrote %s", report_path)


if __name__ == "__main__":
    main()
