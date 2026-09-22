from __future__ import annotations

from pathlib import Path

from backtester.tracker.metrics import monthly_returns_table, strategy_correlation_matrix
from backtester.tracker.report import save_report

from tests.synthetic_data import write_synthetic_parquet
from trading_strategies.config import MacdBacktestConfig
from trading_strategies.network_momentum.runner import run_macd_benchmark
from trading_strategies.risk.attribution import attribute_backtest
from trading_strategies.risk.config import FactorModelConfig
from trading_strategies.risk.report_pages import attribution_pages

REPO_FACTORS = Path(__file__).resolve().parents[2] / "configs" / "factors.json"
_TICKERS = ("SPY", "QQQ", "TLT")
_NUM_BARS = 500


def test_backtest_report_includes_factor_attribution(tmp_path: Path) -> None:
    data = tmp_path / "raw.parquet"
    write_synthetic_parquet(data, _TICKERS, _NUM_BARS)
    config = MacdBacktestConfig(
        name="MACD Benchmark",
        data=str(data),
        tickers=list(_TICKERS),
        output_dir=str(tmp_path / "output"),
        factor_model=FactorModelConfig(factors_file=str(REPO_FACTORS)),
    )
    assert config.factor_model is not None

    trackers = run_macd_benchmark(config)
    results = attribute_backtest(Path(config.data), config.tickers, config.factor_model, trackers)

    buy_and_hold = results["Buy & Hold"]
    assert not buy_and_hold.exposures.empty
    assert "Equity" in buy_and_hold.exposures.columns  # SPY-only Equity leg
    histories = {label: t.mark_to_market_history for label, t in trackers.items()}
    path = save_report(
        Path(config.output_dir),
        histories,
        {label: t.metrics() for label, t in trackers.items()},
        {label: t.trade_metrics() for label, t in trackers.items()},
        {label: monthly_returns_table(h) for label, h in histories.items()},
        strategy_correlation_matrix(histories),
        extra_pages=attribution_pages(results),
        config=config,
    )
    assert path.stat().st_size > 0
