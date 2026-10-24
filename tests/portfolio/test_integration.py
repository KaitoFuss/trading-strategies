"""End-to-end integration test for the optimized portfolio.

Runs a full MACD benchmark backtest with ``portfolio="optimized"`` over
synthetic data, through ``run_macd_benchmark``, and checks the returned
``OptimizedPortfolio`` solved every bar and stayed within its gross cap.
"""

from __future__ import annotations

from pathlib import Path

from tests.synthetic_data import write_synthetic_parquet
from trading_strategies.config import MacdBacktestConfig
from trading_strategies.network_momentum.runner import run_macd_benchmark
from trading_strategies.portfolio.optimized_portfolio import OptimizedPortfolio
from trading_strategies.risk.config import FactorModelConfig, OptimizerConfig

_TICKERS = ("SPY", "QQQ", "TLT")
REPO_FACTORS = Path(__file__).resolve().parents[2] / "configs" / "factors.json"


def test_optimized_config_runs_end_to_end(tmp_path: Path) -> None:
    data = tmp_path / "raw.parquet"
    write_synthetic_parquet(data, _TICKERS, 500)
    config = MacdBacktestConfig(
        name="MACD Optimized",
        data=str(data),
        tickers=list(_TICKERS),
        portfolio="optimized",
        max_gross=2.0,
        output_dir=str(tmp_path / "output"),
        factor_model=FactorModelConfig(factors_file=str(REPO_FACTORS), granular=True),
        optimizer=OptimizerConfig(),
    )

    trackers, portfolio = run_macd_benchmark(config)

    assert isinstance(portfolio, OptimizedPortfolio)
    assert portfolio.solve_log, "the optimizer never ran"
    assert all(entry.gross <= config.max_gross + 1e-6 for entry in portfolio.solve_log)
    # run_macd_benchmark keys its legs "MACD Benchmark" and "Buy & Hold"
    # regardless of config.portfolio; keep that assumption explicit here.
    assert len(trackers["MACD Benchmark"].mark_to_market_history) == 500
