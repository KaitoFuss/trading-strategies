from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_strategies.config import MacdBacktestConfig
from trading_strategies.risk.config import (
    EigenAdjustmentConfig,
    FactorModelConfig,
    MomentumConfig,
    OptimizerConfig,
)


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


def test_optimizer_objective_defaults_to_linear() -> None:
    config = OptimizerConfig.from_dict({})

    assert config.objective == "linear"
    assert config.risk_aversion is None


def test_optimizer_parses_mean_variance() -> None:
    config = OptimizerConfig.from_dict({"objective": "mean_variance", "risk_aversion": 30.0})

    assert config.objective == "mean_variance"
    assert config.risk_aversion == 30.0


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"objective": "quadratic"}, "objective"),
        ({"objective": "mean_variance"}, "risk_aversion"),
        ({"objective": "mean_variance", "risk_aversion": 0.0}, "risk_aversion"),
    ],
)
def test_invalid_mean_variance_settings_are_rejected(raw: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OptimizerConfig.from_dict(raw)


def test_idiosyncratic_risk_defaults_to_residual() -> None:
    config = FactorModelConfig.from_dict({"factors_file": "f.json"})

    assert config.idiosyncratic_risk == "residual"


def test_idiosyncratic_risk_parses_implied() -> None:
    config = FactorModelConfig.from_dict(
        {"factors_file": "f.json", "idiosyncratic_risk": "implied"}
    )

    assert config.idiosyncratic_risk == "implied"


def test_idiosyncratic_risk_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="idiosyncratic_risk"):
        FactorModelConfig(factors_file="f.json", idiosyncratic_risk="measured")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name", ["backtest_macd.json", "backtest_macd_rescaled.json", "backtest_optimized.json"]
)
def test_shipped_configs_set_idiosyncratic_risk_explicitly(name: str) -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / name
    raw = json.loads(path.read_text())

    assert raw["factor_model"]["idiosyncratic_risk"] == "residual"


def test_eigen_adjustment_defaults_to_on() -> None:
    config = FactorModelConfig.from_dict({"factors_file": "f.json"})

    assert config.eigen_adjustment == EigenAdjustmentConfig()
    assert config.eigen_adjustment.enabled is True
    assert config.eigen_adjustment.scale == 1.4


def test_eigen_adjustment_parses_nested_block() -> None:
    config = FactorModelConfig.from_dict(
        {
            "factors_file": "f.json",
            "eigen_adjustment": {"enabled": True, "simulations": 50, "window": 300},
        }
    )

    assert config.eigen_adjustment == EigenAdjustmentConfig(
        enabled=True, simulations=50, window=300
    )


@pytest.mark.parametrize(
    "fields",
    [{"scale": -0.1}, {"simulations": 0}, {"refresh_every": 0}, {"window": 1}],
)
def test_eigen_adjustment_rejects_bad_values(fields: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="eigen_adjustment"):
        EigenAdjustmentConfig(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name", ["backtest_macd.json", "backtest_macd_rescaled.json", "backtest_optimized.json"]
)
def test_shipped_configs_set_eigen_adjustment_explicitly(name: str) -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / name
    block = json.loads(path.read_text())["factor_model"]["eigen_adjustment"]

    assert set(block) == {"enabled", "scale", "simulations", "window", "refresh_every", "seed"}
    assert block["enabled"] is True
