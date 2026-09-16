"""Config for the MACD benchmark backtest.

``backtester.config.BacktestConfig`` has no field for ``target_vol`` --
it is specific to ``TargetVolPortfolio``, not to ``backtester``'s own
portfolios -- so this repo carries its own config dataclass and adapts
the shared subset of fields into a real ``BacktestConfig`` where
``backtester`` requires that exact type (``run_backtest``,
``report.save_report``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from backtester.config import BacktestConfig


@dataclass(frozen=True)
class MacdBacktestConfig:
    name: str
    data: str
    tickers: list[str]
    target_vol: float = 0.15
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

    def to_backtest_config(self) -> BacktestConfig:
        return BacktestConfig(
            name=self.name,
            data=self.data,
            tickers=self.tickers,
            initial_cash=self.initial_cash,
            max_gross=self.max_gross,
            cost_bps=self.cost_bps,
            commission_bps=self.commission_bps,
            risk_free_rate=self.risk_free_rate,
            output_dir=self.output_dir,
        )
