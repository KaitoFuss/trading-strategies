from __future__ import annotations

import json
from pathlib import Path

import pytest
from backtester.config import BacktestConfig

from trading_strategies.config import MacdBacktestConfig


def test_from_json_round_trips_fields(tmp_path: Path) -> None:
    payload = {
        "name": "MACD Benchmark",
        "data": "data/raw.parquet",
        "tickers": ["SPY", "QQQ"],
        "target_vol": 0.2,
        "initial_cash": 250_000.0,
        "max_gross": 1.5,
        "cost_bps": 0.0,
        "commission_bps": 0.0,
        "risk_free_rate": 0.01,
        "output_dir": "output",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))

    config = MacdBacktestConfig.from_json(path)

    assert config.name == "MACD Benchmark"
    assert config.tickers == ["SPY", "QQQ"]
    assert config.target_vol == 0.2
    assert config.initial_cash == 250_000.0
    assert config.max_gross == 1.5
    assert config.risk_free_rate == 0.01


def test_to_backtest_config_maps_shared_fields() -> None:
    config = MacdBacktestConfig(
        name="MACD Benchmark",
        data="data/raw.parquet",
        tickers=["SPY", "QQQ"],
        target_vol=0.2,
        initial_cash=250_000.0,
        max_gross=1.5,
        cost_bps=0.1,
        commission_bps=0.05,
        risk_free_rate=0.01,
        output_dir="output",
    )

    backtest_config = config.to_backtest_config()

    assert isinstance(backtest_config, BacktestConfig)
    assert backtest_config.name == "MACD Benchmark"
    assert backtest_config.data == "data/raw.parquet"
    assert backtest_config.tickers == ["SPY", "QQQ"]
    assert backtest_config.initial_cash == 250_000.0
    assert backtest_config.max_gross == 1.5
    assert backtest_config.cost_bps == 0.1
    assert backtest_config.commission_bps == 0.05
    assert backtest_config.risk_free_rate == 0.01
    assert backtest_config.output_dir == "output"


def test_from_json_loads_shipped_config() -> None:
    config_path = Path(__file__).parent.parent / "configs" / "backtest_macd.json"
    config = MacdBacktestConfig.from_json(config_path)

    assert len(config.tickers) == 61
    assert config.target_vol == 0.15
    assert config.cost_bps == 0.0
    assert config.commission_bps == 0.0


def test_from_json_raises_value_error_on_invalid_config(tmp_path: Path) -> None:
    payload = {
        "name": "MACD Benchmark",
        "data": "data/raw.parquet",
        "tickers": ["SPY", "QQQ"],
        "not_a_real_field": 1,
    }
    path = tmp_path / "bad_config.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=str(path)):
        MacdBacktestConfig.from_json(path)
