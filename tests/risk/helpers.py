"""Shared test helpers for the risk model tests (also used by task B7)."""

from __future__ import annotations

import json
from pathlib import Path

from trading_strategies.risk.config import FactorModelConfig, MomentumConfig


def market_factor_config(tmp_path: Path, **overrides: object) -> FactorModelConfig:
    factors_file = tmp_path / "factors.json"
    factors_file.write_text(
        json.dumps(
            {
                "factors": [
                    {"name": "Market", "level": 1, "granular": False, "weights": {"MKT": 1.0}},
                ]
            }
        )
    )
    fields: dict[str, object] = {
        "factors_file": str(factors_file),
        "vol_span": 20,
        "corr_span": 100,
        "corr_min_periods": 50,
        "momentum": MomentumConfig(lookback=20, skip=2),
    }
    fields.update(overrides)
    return FactorModelConfig(**fields)  # type: ignore[arg-type]
