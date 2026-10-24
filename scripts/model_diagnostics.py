"""Check the factor model before you trust it for position sizing.

It measures three things. Do residual correlations stay small, so a diagonal
``D`` is defensible. Is the momentum factor really separate from the macro
betas. Have the loadings drifted so much that the betas are no longer stable
enough to size on.

Run it by hand, not during a backtest. Run it before trusting the model for
sizing, and again after any change to the factor set, the spans, or the
universe.

Usage:
    uv run scripts/model_diagnostics.py configs/backtest_optimized.json

How to read it: residual |rho| mostly under ~0.2 is fine; a pair above 0.5
wants a factor or a universe change. Momentum's correlation with the macro
factor returns near zero justifies the block-diagonal ``F``. A |rho| above 1
is possible and just means ``D`` is too small for the observed
cross-covariance -- that is informative, not a bug. Masked assets are the
factor-defining ETFs.
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
    if off.size:
        logger.info(
            "residual correlations: N=%d of %d  mean|rho|=%.3f  p95|rho|=%.3f  share>0.5=%.1f%%",
            len(report.correlations),
            len(report.share),
            float(np.mean(np.abs(off))),
            float(np.percentile(np.abs(off), 95)),
            100.0 * float(np.mean(np.abs(off) > 0.5)),
        )
    else:
        logger.info(
            "residual correlations: N=%d of %d (no off-diagonal pairs to report)",
            len(report.correlations),
            len(report.share),
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
