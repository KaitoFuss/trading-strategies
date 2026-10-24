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

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.network_momentum.runner import run_macd_benchmark
from trading_strategies.portfolio.calibration import (
    SweepRow,
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
        realized = equity.pct_change()
        forecast = pd.Series(
            [entry.ex_ante_vol / np.sqrt(252.0) for entry in portfolio.solve_log],
            index=pd.DatetimeIndex([entry.timestamp for entry in portfolio.solve_log]),
            dtype=float,
        )
        frame = pd.DataFrame({"forecast": forecast}).join(realized.rename("realized"), how="inner")
        per_delta[shrinkage] = frame.dropna()

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

    logger.info("\nwalk-forward folds:\n%s", fold_table(per_delta).round(3).to_string())
    logger.info("\nfull-sample selection: shrinkage=%.2f", select_shrinkage(rows))


if __name__ == "__main__":
    main()
