"""Sweep the optimizer's shrinkage parameter and pick one by walk-forward.

Usage:
    uv run scripts/calibrate_shrinkage.py configs/backtest_optimized.json

Runs one backtest per candidate shrinkage (five by default), with trading costs
on, then slices the per-bar forecasts and realized returns into expanding folds.
Selection uses the bias statistic only; Sharpe and turnover are printed as a
guard. Takes several minutes per candidate -- this is a calibration tool, not
part of the backtest.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.network_momentum.runner import run_macd_benchmark
from trading_strategies.portfolio.calibration import (
    SweepRow,
    align_to_next_return,
    bias_statistic,
    fold_table,
    select_shrinkage,
)
from trading_strategies.portfolio.optimized_portfolio import OptimizedPortfolio
from trading_strategies.risk.config import OptimizerConfig

GRID = (0.0, 0.1, 0.2, 0.3, 0.5)
COST_BPS = 1.0
COMMISSION_BPS = 0.5

logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: calibrate_shrinkage.py <config.json>")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base = MacdBacktestConfig.from_json(Path(sys.argv[1]))
    if base.portfolio != "optimized":
        raise SystemExit("config must set portfolio='optimized'")

    per_delta: dict[float, pd.DataFrame] = {}
    rows: list[SweepRow] = []
    for shrinkage in GRID:
        optimizer = replace(base.optimizer or OptimizerConfig(), shrinkage=shrinkage)
        config = replace(
            base, optimizer=optimizer, cost_bps=COST_BPS, commission_bps=COMMISSION_BPS
        )
        trackers, portfolio = run_macd_benchmark(config)
        assert isinstance(portfolio, OptimizedPortfolio)
        tracker = trackers["MACD Benchmark"]
        metrics = tracker.metrics()
        trades = tracker.trade_metrics()

        history = tracker.mark_to_market_history
        equity = pd.Series(
            [value for _, value in history],
            index=pd.DatetimeIndex([timestamp for timestamp, _ in history]),
            dtype=float,
        )
        realized = align_to_next_return(equity)
        forecast = pd.Series(
            [entry.ex_ante_vol / np.sqrt(TRADING_DAYS_PER_YEAR) for entry in portfolio.solve_log],
            index=pd.DatetimeIndex([entry.timestamp for entry in portfolio.solve_log]),
            dtype=float,
        )
        # ``realized`` is already shifted by align_to_next_return, so a plain join on the
        # solve-log timestamp pairs each forecast with the return it is responsible for.
        frame = pd.DataFrame({"forecast": forecast}).join(realized.rename("realized"), how="inner")
        frame = frame.dropna()
        per_delta[shrinkage] = frame
        if len(frame) < 250:
            logger.warning("only %d aligned bars for shrinkage=%.2f", len(frame), shrinkage)

        gross = float(np.mean([entry.gross for entry in portfolio.solve_log]))
        rows.append(
            SweepRow(
                shrinkage=shrinkage,
                bias=bias_statistic(frame["forecast"], frame["realized"]),
                sharpe=metrics.sharpe,
                realized_vol=metrics.annualized_vol,
                turnover=trades.annual_turnover,
                gross=gross,
            )
        )
        logger.info(
            "shrinkage=%.2f bias=%.3f sharpe=%.2f vol=%.2f%% turnover=%.1fx gross=%.2f",
            shrinkage,
            rows[-1].bias,
            rows[-1].sharpe,
            100 * rows[-1].realized_vol,
            rows[-1].turnover,
            gross,
        )

    logger.info(
        "\nin-sample reference only (not the selection): shrinkage=%.2f", select_shrinkage(rows)
    )
    logger.info("\nwalk-forward folds:\n%s", fold_table(per_delta).round(3).to_string())


if __name__ == "__main__":
    main()
