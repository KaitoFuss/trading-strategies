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
    assert config.portfolio == "target_vol"


def test_from_json_loads_shipped_rescaled_config() -> None:
    config_path = Path(__file__).parent.parent / "configs" / "backtest_macd_rescaled.json"
    config = MacdBacktestConfig.from_json(config_path)

    assert len(config.tickers) == 61
    assert config.target_vol == 0.15
    assert config.portfolio == "rescaled_target_vol"


def test_portfolio_defaults_to_target_vol() -> None:
    config = MacdBacktestConfig(name="MACD Benchmark", data="data/raw.parquet", tickers=["SPY"])
    assert config.portfolio == "target_vol"


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


def test_portfolio_field_parses_and_defaults(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        '{"name": "X", "data": "d.parquet", "tickers": ["SPY"], "portfolio": "optimized",'
        ' "factor_model": {"factors_file": "f.json", "granular": true},'
        ' "optimizer": {"information_coefficient": 0.1,'
        ' "factor_bounds": {"Equity": [-0.2, 0.2]}}}'
    )

    config = MacdBacktestConfig.from_json(path)

    assert config.portfolio == "optimized"
    assert config.optimizer is not None
    assert config.optimizer.information_coefficient == 0.1
    assert config.optimizer.factor_bounds == {"Equity": (-0.2, 0.2)}


def test_unknown_portfolio_value_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text('{"name": "X", "data": "d.parquet", "tickers": ["SPY"], "portfolio": "bogus"}')

    with pytest.raises(ValueError, match="portfolio"):
        MacdBacktestConfig.from_json(path)


def test_optimized_requires_a_factor_model(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        '{"name": "X", "data": "d.parquet", "tickers": ["SPY"], "portfolio": "optimized"}'
    )

    with pytest.raises(ValueError, match="factor_model"):
        MacdBacktestConfig.from_json(path)


def test_optimized_requires_granular_factors(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        '{"name": "X", "data": "d.parquet", "tickers": ["SPY"], "portfolio": "optimized", '
        '"factor_model": {"factors_file": "f.json"}}'
    )

    with pytest.raises(ValueError, match="granular"):
        MacdBacktestConfig.from_json(path)


def test_shipped_configs_hold_the_full_universe() -> None:
    """PFF stays in the universe: the risk model explains it almost fully,
    so the diagnostics mask it instead of reporting a noise-over-noise
    correlation. Dropping it from the universe changed Sharpe by nothing
    (0.20 either way), so no fetched ETF is excluded on discretion."""
    configs_dir = Path(__file__).parent.parent / "configs"
    expected_portfolios = {
        "backtest_macd.json": "target_vol",
        "backtest_macd_rescaled.json": "rescaled_target_vol",
        "backtest_optimized.json": "optimized",
    }
    for name, expected_portfolio in expected_portfolios.items():
        config = MacdBacktestConfig.from_json(configs_dir / name)
        assert len(config.tickers) == 61, name
        assert "PFF" in config.tickers, name
        assert config.portfolio == expected_portfolio, name
