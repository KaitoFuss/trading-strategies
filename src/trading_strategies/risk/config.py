"""Config for the factor risk model (see
docs/superpowers/specs/2026-09-22-factor-risk-model-design.md)."""

from __future__ import annotations

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
