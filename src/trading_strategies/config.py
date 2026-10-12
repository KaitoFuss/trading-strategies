"""Config for the MACD benchmark backtest.

``backtester.config.BacktestConfig`` has no field for ``target_vol`` --
it is specific to ``TargetVolPortfolio``, not to ``backtester``'s own
portfolios -- so this repo carries its own config dataclass. It satisfies
``backtester.config.EngineConfig`` / ``backtester.tracker.report.ReportConfig``
structurally, so it passes straight into ``run_backtest`` and
``report.save_report`` with no adapter needed.

The optional ``factor_model`` field enables factor-attribution reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from trading_strategies.risk.config import FactorModelConfig, OptimizerConfig

PortfolioKind = Literal["target_vol", "rescaled_target_vol", "optimized"]
_PORTFOLIO_KINDS: frozenset[str] = frozenset({"target_vol", "rescaled_target_vol", "optimized"})


@dataclass(frozen=True)
class MacdBacktestConfig:
    name: str
    data: str
    tickers: list[str]
    target_vol: float = 0.15
    portfolio: PortfolioKind = "target_vol"
    optimizer: OptimizerConfig | None = None
    initial_cash: float = 100_000.0
    max_gross: float = 1.0
    cost_bps: float = 0.0
    commission_bps: float = 0.0
    risk_free_rate: float = 0.0
    output_dir: str = "output"
    factor_model: FactorModelConfig | None = None

    @classmethod
    def from_json(cls, path: Path) -> Self:
        try:
            raw = json.loads(path.read_text())
            if "factor_model" in raw:
                raw["factor_model"] = FactorModelConfig.from_dict(raw["factor_model"])
            if "optimizer" in raw:
                raw["optimizer"] = OptimizerConfig.from_dict(raw["optimizer"])
            kind = raw.get("portfolio", "target_vol")
            if kind not in _PORTFOLIO_KINDS:
                raise ValueError(
                    f"portfolio must be one of {sorted(_PORTFOLIO_KINDS)}, got {kind!r}"
                )
            config = cls(**raw)
            if config.portfolio == "optimized":
                if config.factor_model is None:
                    raise ValueError("portfolio='optimized' requires a factor_model section")
                if not config.factor_model.granular:
                    raise ValueError("portfolio='optimized' requires factor_model.granular=true")
            return config
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"invalid config at {path}: {error}") from error
