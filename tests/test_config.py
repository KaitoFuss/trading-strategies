import json
from pathlib import Path

from trading_strategies.config import NetworkMomentumConfig


def test_from_json_round_trips_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "name": "Test",
                "data": "data/raw.parquet",
                "tickers": ["A", "B"],
                "initial_cash": 500_000.0,
                "max_gross": 1.5,
                "risk_aversion": 3.0,
                "min_periods": 30,
                "cost_bps": 1.0,
                "commission_bps": 0.5,
                "output_dir": "out",
                "risk_free_rate": 0.03,
            }
        )
    )

    config = NetworkMomentumConfig.from_json(path)

    assert config.name == "Test"
    assert config.tickers == ["A", "B"]
    assert config.risk_aversion == 3.0
    assert config.min_periods == 30


def test_to_backtest_config_carries_only_the_shared_fields() -> None:
    config = NetworkMomentumConfig(
        name="Test",
        data="data/raw.parquet",
        tickers=["A", "B"],
        initial_cash=500_000.0,
        max_gross=1.5,
        risk_aversion=3.0,
        min_periods=30,
        cost_bps=1.0,
        commission_bps=0.5,
        output_dir="out",
        risk_free_rate=0.03,
    )

    backtest_config = config.to_backtest_config()

    assert backtest_config.name == "Test"
    assert backtest_config.data == "data/raw.parquet"
    assert backtest_config.tickers == ["A", "B"]
    assert backtest_config.initial_cash == 500_000.0
    assert backtest_config.max_gross == 1.5
    assert backtest_config.cost_bps == 1.0
    assert backtest_config.commission_bps == 0.5
    assert backtest_config.output_dir == "out"
    assert backtest_config.risk_free_rate == 0.03
