from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trading_strategies.risk.factors import (
    FactorDefinition,
    factor_weight_matrix,
    load_factors,
    residualize,
)

REPO_FACTORS = Path(__file__).resolve().parents[2] / "configs" / "factors.json"


def test_repo_factor_file_has_eight_macro_and_seven_granular() -> None:
    assert len(load_factors(REPO_FACTORS, granular=False)) == 8
    assert len(load_factors(REPO_FACTORS, granular=True)) == 15


def test_factors_are_sorted_by_level_stably(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(
        json.dumps(
            {
                "factors": [
                    {"name": "B", "level": 2, "granular": False, "weights": {"X": 1.0}},
                    {"name": "A1", "level": 1, "granular": False, "weights": {"Y": 1.0}},
                    {"name": "A2", "level": 1, "granular": False, "weights": {"Z": 1.0}},
                ]
            }
        )
    )

    assert [f.name for f in load_factors(path, granular=False)] == ["A1", "A2", "B"]


@pytest.mark.parametrize(
    ("factors", "message"),
    [
        ([{"name": "A", "level": 1, "granular": False, "weights": {}}], "weights"),
        (
            [
                {"name": "A", "level": 1, "granular": False, "weights": {"X": 1.0}},
                {"name": "A", "level": 2, "granular": False, "weights": {"Y": 1.0}},
            ],
            "duplicate",
        ),
    ],
)
def test_invalid_factor_files_are_rejected(
    tmp_path: Path, factors: list[dict[str, object]], message: str
) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"factors": factors}))
    with pytest.raises(ValueError, match=message):
        load_factors(path, granular=True)


def _factor(name: str, weights: dict[str, float], level: int = 1) -> FactorDefinition:
    return FactorDefinition(name=name, level=level, granular=False, weights=weights)


def test_missing_long_ticker_renormalizes_the_leg() -> None:
    equity = _factor("Equity", {"SPY": 0.6, "EFA": 0.3, "EEM": 0.1})
    matrix = factor_weight_matrix([equity], ["SPY", "EFA"], available={"SPY", "EFA"})

    np.testing.assert_allclose(matrix[:, 0], [0.6 / 0.9, 0.3 / 0.9])


def test_spread_with_a_missing_leg_is_inactive() -> None:
    spread = _factor("US vs Intl", {"SPY": 1.0, "EFA": -1.0})
    matrix = factor_weight_matrix([spread], ["SPY", "EFA"], available={"SPY"})

    assert not matrix.any()


def test_short_only_factor_keeps_its_sign() -> None:
    fx = _factor("Foreign Currency", {"UUP": -1.0})
    matrix = factor_weight_matrix([fx], ["UUP"], available={"UUP"})

    np.testing.assert_allclose(matrix[:, 0], [-1.0])


def test_later_level_is_orthogonal_to_earlier_levels() -> None:
    covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
    weights = np.eye(2)

    residual = residualize(weights, [1, 2], covariance)

    assert residual[:, 0] @ covariance @ residual[:, 1] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(residual[:, 1], [-0.5, 1.0])


def test_same_level_factors_are_not_residualized() -> None:
    covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
    weights = np.eye(2)

    np.testing.assert_allclose(residualize(weights, [1, 1], covariance), weights)


def test_inactive_columns_are_skipped() -> None:
    covariance = np.eye(2)
    weights = np.array([[0.0, 1.0], [0.0, 0.0]])

    np.testing.assert_allclose(residualize(weights, [1, 2], covariance), weights)
