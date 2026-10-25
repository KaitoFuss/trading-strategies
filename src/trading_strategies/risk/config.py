"""Config for the factor risk model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self, get_args

Objective = Literal["linear", "mean_variance"]
_OBJECTIVES = get_args(Objective)
IdiosyncraticRisk = Literal["implied", "residual"]
_IDIOSYNCRATIC_RISKS = get_args(IdiosyncraticRisk)


def validate_objective(objective: str, risk_aversion: float | None) -> None:
    """``mean_variance`` needs a positive ``risk_aversion`` (daily units);
    ``linear`` ignores it."""
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {list(_OBJECTIVES)}, got {objective!r}")
    if objective != "mean_variance":
        return
    if risk_aversion is None or risk_aversion <= 0:
        raise ValueError(f"objective='mean_variance' needs risk_aversion > 0, got {risk_aversion}")


@dataclass(frozen=True)
class MomentumConfig:
    lookback: int = 252
    skip: int = 21


@dataclass(frozen=True)
class EigenAdjustmentConfig:
    """Eigenfactor risk adjustment of ``F`` (``risk/eigen_adjustment.py``).
    ``window=None`` means ``3 * corr_span``. ``v`` is re-simulated every
    ``refresh_every`` bars and reused in between."""

    enabled: bool = True
    scale: float = 1.4
    simulations: int = 200
    window: int | None = None
    refresh_every: int = 21
    seed: int = 0

    def __post_init__(self) -> None:
        if self.scale < 0:
            raise ValueError(f"eigen_adjustment.scale must be >= 0, got {self.scale}")
        if self.simulations < 1:
            raise ValueError(f"eigen_adjustment.simulations must be >= 1, got {self.simulations}")
        if self.refresh_every < 1:
            raise ValueError(
                f"eigen_adjustment.refresh_every must be >= 1, got {self.refresh_every}"
            )
        if self.window is not None and self.window < 2:
            raise ValueError(f"eigen_adjustment.window must be >= 2, got {self.window}")


@dataclass(frozen=True)
class FactorModelConfig:
    factors_file: str
    granular: bool = False
    vol_span: int = 60
    corr_span: int = 250
    corr_min_periods: int = 250
    residual_floor: float = 0.10
    idiosyncratic_risk: IdiosyncraticRisk = "residual"
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    eigen_adjustment: EigenAdjustmentConfig = field(default_factory=EigenAdjustmentConfig)

    def __post_init__(self) -> None:
        if self.idiosyncratic_risk not in _IDIOSYNCRATIC_RISKS:
            raise ValueError(
                f"idiosyncratic_risk must be one of {list(_IDIOSYNCRATIC_RISKS)}, "
                f"got {self.idiosyncratic_risk!r}"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        fields = dict(raw)
        momentum = MomentumConfig(**fields.pop("momentum", {}))
        eigen_adjustment = EigenAdjustmentConfig(**fields.pop("eigen_adjustment", {}))
        return cls(momentum=momentum, eigen_adjustment=eigen_adjustment, **fields)


@dataclass(frozen=True)
class OptimizerConfig:
    information_coefficient: float = 0.05
    turnover_penalty_bps: float = 5.0
    factor_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    objective: Objective = "linear"
    risk_aversion: float | None = None

    def __post_init__(self) -> None:
        validate_objective(self.objective, self.risk_aversion)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        fields = dict(raw)
        bounds = {
            name: (float(low), float(high))
            for name, (low, high) in fields.pop("factor_bounds", {}).items()
        }
        return cls(factor_bounds=bounds, **fields)
