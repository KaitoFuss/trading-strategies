"""Config for the factor risk model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self

Objective = Literal["linear", "mean_variance"]
_OBJECTIVES = ("linear", "mean_variance")


def validate_objective(objective: str, risk_aversion: float | None, vol_cap: float | None) -> None:
    """``mean_variance`` needs a positive ``risk_aversion`` (daily units) and a
    positive ``vol_cap`` (annualized); ``linear`` ignores both."""
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {list(_OBJECTIVES)}, got {objective!r}")
    if objective != "mean_variance":
        return
    if risk_aversion is None or risk_aversion <= 0:
        raise ValueError(f"objective='mean_variance' needs risk_aversion > 0, got {risk_aversion}")
    if vol_cap is None or vol_cap <= 0:
        raise ValueError(f"objective='mean_variance' needs vol_cap > 0, got {vol_cap}")


@dataclass(frozen=True)
class MomentumConfig:
    lookback: int = 252
    skip: int = 21


@dataclass(frozen=True)
class FactorModelConfig:
    factors_file: str
    granular: bool = False
    vol_span: int = 60
    corr_span: int = 250
    corr_min_periods: int = 250
    residual_floor: float = 0.10
    momentum: MomentumConfig = field(default_factory=MomentumConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        fields = dict(raw)
        momentum = MomentumConfig(**fields.pop("momentum", {}))
        return cls(momentum=momentum, **fields)


@dataclass(frozen=True)
class OptimizerConfig:
    information_coefficient: float = 0.05
    turnover_penalty_bps: float = 5.0
    factor_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    objective: Objective = "linear"
    risk_aversion: float | None = None
    vol_cap: float | None = None

    def __post_init__(self) -> None:
        validate_objective(self.objective, self.risk_aversion, self.vol_cap)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        fields = dict(raw)
        bounds = {
            name: (float(low), float(high))
            for name, (low, high) in fields.pop("factor_bounds", {}).items()
        }
        return cls(factor_bounds=bounds, **fields)
