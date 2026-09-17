from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_from_json_loads_shipped_config() -> None:
    config_path = Path(__file__).parent.parent / "configs" / "backtest_macd.json"
    config = MacdBacktestConfig.from_json(config_path)

    assert len(config.tickers) == 61
    assert config.target_vol == 0.15
    assert config.cost_bps == 0.0
    assert config.commission_bps == 0.0
    assert config.rescale_to_portfolio_vol is False


def test_from_json_loads_shipped_rescaled_config() -> None:
    config_path = Path(__file__).parent.parent / "configs" / "backtest_macd_rescaled.json"
    config = MacdBacktestConfig.from_json(config_path)

    assert len(config.tickers) == 61
    assert config.target_vol == 0.15
    assert config.rescale_to_portfolio_vol is True


def test_rescale_to_portfolio_vol_defaults_to_false() -> None:
    config = MacdBacktestConfig(name="MACD Benchmark", data="data/raw.parquet", tickers=["SPY"])
    assert config.rescale_to_portfolio_vol is False


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
