from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from backtester.config import BacktestConfig


@dataclass(frozen=True)
class NetworkMomentumConfig:
    name: str
    data: str
    tickers: list[str] | None = None
    initial_cash: float = 100_000.0
    max_gross: float = 1.0
    risk_aversion: float = 1.0
    min_periods: int = 60
    """Minimum days of return history a ticker needs before Sigma will use it."""
    drift_band: float = 0.0
    """No-trade region around each ticker's target weight; see MeanVariancePortfolio."""
    cost_bps: float = 0.0
    commission_bps: float = 0.0
    output_dir: str = "output"
    risk_free_rate: float = 0.0

    @classmethod
    def from_json(cls, path: Path) -> Self:
        return cls(**json.loads(path.read_text()))

    def to_backtest_config(self) -> BacktestConfig:
        """The subset backtester.runner.run_backtest actually needs — just
        the shared fields, none of MeanVariancePortfolio's own (those go to
        its constructor directly, since BacktestConfig has no slot for them
        and isn't ours to extend)."""
        return BacktestConfig(
            name=self.name,
            data=self.data,
            tickers=self.tickers,
            initial_cash=self.initial_cash,
            max_gross=self.max_gross,
            cost_bps=self.cost_bps,
            commission_bps=self.commission_bps,
            output_dir=self.output_dir,
            risk_free_rate=self.risk_free_rate,
        )
