"""Config for the MACD benchmark backtest.

``backtester.config.BacktestConfig`` has no field for ``target_vol`` --
it is specific to ``TargetVolPortfolio``, not to ``backtester``'s own
portfolios -- so this repo carries its own config dataclass. It satisfies
``backtester.config.EngineConfig`` / ``backtester.tracker.report.ReportConfig``
structurally, so it passes straight into ``run_backtest`` and
``report.save_report`` with no adapter needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class MacdBacktestConfig:
    name: str
    data: str
    tickers: list[str]
    target_vol: float = 0.15
    rescale_to_portfolio_vol: bool = False
    initial_cash: float = 100_000.0
    max_gross: float = 1.0
    cost_bps: float = 0.0
    commission_bps: float = 0.0
    risk_free_rate: float = 0.0
    output_dir: str = "output"

    @classmethod
    def from_json(cls, path: Path) -> Self:
        try:
            return cls(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"invalid config at {path}: {error}") from error
