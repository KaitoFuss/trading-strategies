"""Print the factor model's three sizing diagnostics for a backtest config.

Usage:
    uv run scripts/model_diagnostics.py configs/backtest_optimized.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.risk.diagnostics import (
    exposure_collinearity,
    loading_drift,
    replay_for_diagnostics,
    residual_correlations,
)
from trading_strategies.risk.model import MOMENTUM

logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: model_diagnostics.py <config.json>")
    config = MacdBacktestConfig.from_json(Path(sys.argv[1]))
    if config.factor_model is None:
        raise SystemExit("config has no factor_model section")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    replay = replay_for_diagnostics(Path(config.data), config.tickers, config.factor_model)

    report = residual_correlations(
        replay.final_snapshot, replay.final_empirical_cov, config.tickers
    )
    off = report.correlations.to_numpy()
    off = off[np.isfinite(off)]
    logger.info(
        "residual correlations: N=%d of %d  mean|rho|=%.3f  p95|rho|=%.3f  share>0.5=%.1f%%",
        len(report.correlations),
        len(report.share),
        float(np.mean(np.abs(off))),
        float(np.percentile(np.abs(off), 95)),
        100.0 * float(np.mean(np.abs(off) > 0.5)),
    )
    logger.info("worst pairs:\n%s", report.worst_pairs().round(2).to_string())
    logger.info(
        "masked (residual share <= 5%%): %s",
        ", ".join(f"{t} {report.share[t]:.1%}" for t in report.masked),
    )

    returns = replay.factor_returns
    if MOMENTUM in returns.columns:
        corr = returns.corr()[MOMENTUM].drop(MOMENTUM)
        logger.info(
            "momentum factor-return correlation (max |rho| = %.3f):\n%s",
            float(corr.abs().max()),
            corr.round(3).to_string(),
        )
    logger.info(
        "momentum exposure vs macro betas:\n%s",
        exposure_collinearity(replay.final_snapshot).round(3).to_string(),
    )
    logger.info("loading drift:\n%s", loading_drift(replay.loadings).round(4).to_string())


if __name__ == "__main__":
    main()
