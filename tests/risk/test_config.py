from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.risk.config import FactorModelConfig, MomentumConfig


def test_from_dict_applies_defaults() -> None:
    config = FactorModelConfig.from_dict({"factors_file": "configs/factors.json"})

    assert config == FactorModelConfig(factors_file="configs/factors.json")
    assert config.momentum == MomentumConfig(lookback=252, skip=21)
    assert config.residual_floor == 0.10


def test_from_dict_parses_nested_momentum() -> None:
    config = FactorModelConfig.from_dict(
        {"factors_file": "f.json", "granular": True, "momentum": {"lookback": 126, "skip": 5}}
    )

    assert config.granular is True
    assert config.momentum == MomentumConfig(lookback=126, skip=5)


def _write(path: Path, extra: dict[str, object]) -> Path:
    path.write_text(json.dumps({"name": "X", "data": "d.parquet", "tickers": ["SPY"], **extra}))
    return path


def test_backtest_config_without_factor_model_is_none(tmp_path: Path) -> None:
    config = MacdBacktestConfig.from_json(_write(tmp_path / "c.json", {}))

    assert config.factor_model is None


def test_backtest_config_parses_factor_model(tmp_path: Path) -> None:
    config = MacdBacktestConfig.from_json(
        _write(tmp_path / "c.json", {"factor_model": {"factors_file": "f.json", "corr_span": 120}})
    )

    assert config.factor_model == FactorModelConfig(factors_file="f.json", corr_span=120)


def test_backtest_config_rejects_unknown_factor_model_field(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid config"):
        MacdBacktestConfig.from_json(
            _write(tmp_path / "c.json", {"factor_model": {"factors_file": "f.json", "bogus": 1}})
        )
