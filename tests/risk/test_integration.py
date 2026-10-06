from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import pytest
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


def test_weights_history_earns_the_marked_return(tmp_path: Path) -> None:
    """The timing contract attribution relies on: ``weights_history[t]`` holds
    the weights that earned bar t, so r_p,t = sum_i w_i,t * (c_i,t / c_i,t-1 - 1)."""
    data = tmp_path / "raw.parquet"
    write_synthetic_parquet(data, _TICKERS, _NUM_BARS)
    config = MacdBacktestConfig(
        name="MACD Benchmark",
        data=str(data),
        tickers=list(_TICKERS),
        output_dir=str(tmp_path / "output"),
        cost_bps=0.0,
        commission_bps=0.0,
    )
    raw = pd.read_parquet(data).pivot(index="date", columns="ticker", values="close")
    closes = raw.to_dict(orient="index")  # pd.Timestamp keys hash like datetime

    trackers = run_macd_benchmark(config)

    for label, tracker in trackers.items():
        weights = dict(tracker.weights_history)
        checked = 0
        for (previous, prev_equity), (timestamp, equity) in itertools.pairwise(
            tracker.mark_to_market_history
        ):
            asset_returns = {t: closes[timestamp][t] / closes[previous][t] - 1.0 for t in _TICKERS}
            expected = sum(w * asset_returns[t] for t, w in weights[timestamp].items())
            assert equity / prev_equity - 1.0 == pytest.approx(expected, abs=1e-9), (
                f"{label} at {timestamp}"
            )
            checked += 1
        assert checked == _NUM_BARS - 1
        assert any(weights[timestamp] for timestamp in weights)
