"""Config for the factor risk model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self


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
    shrinkage: float = 0.0
    ridge: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be within [0, 1], got {self.shrinkage}")
        if self.ridge < 0.0:
            raise ValueError(f"ridge must be >= 0, got {self.ridge}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        fields = dict(raw)
        bounds = {
            name: (float(low), float(high))
            for name, (low, high) in fields.pop("factor_bounds", {}).items()
        }
        return cls(factor_bounds=bounds, **fields)
