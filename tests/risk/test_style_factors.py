from __future__ import annotations

import math

import numpy as np
import pytest

from trading_strategies.risk.style_factors import MomentumSignal, momentum_exposures


def test_raw_is_none_until_lookback_plus_one_closes() -> None:
    signal = MomentumSignal(["A"], lookback=3, skip=1)
    for close in (100.0, 101.0, 102.0):
        signal.update({"A": close})
    assert signal.raw_return("A") is None

    signal.update({"A": 104.0})
    assert signal.raw_return("A") == pytest.approx(math.log(102.0 / 100.0))


def test_raw_rolls_forward_and_ignores_unknown_tickers() -> None:
    signal = MomentumSignal(["A"], lookback=2, skip=0)
    for close in (100.0, 110.0, 121.0, 133.1):
        signal.update({"A": close, "ZZZ": 1.0})

    assert signal.raw_return("A") == pytest.approx(math.log(133.1 / 110.0))


def test_exposures_are_standardized() -> None:
    raw = {f"T{i}": 0.01 * i for i in range(10)}
    sigma = dict.fromkeys(raw, 0.01)

    values = np.array(list(momentum_exposures(raw, sigma).values()))

    assert values.mean() == pytest.approx(0.0, abs=1e-12)
    assert values.std() == pytest.approx(1.0)


def test_single_outlier_is_winsorized() -> None:
    """Without the robust clip, one outlier among 101 names sits at z ~ 10."""
    rng = np.random.default_rng(0)
    raw = {f"T{i}": float(v) for i, v in enumerate(rng.normal(0, 1, 100))} | {"BIG": 1_000.0}
    sigma = dict.fromkeys(raw, 1.0)

    exposures = momentum_exposures(raw, sigma)

    assert exposures["BIG"] < 4.0
    assert exposures["BIG"] == max(exposures.values())


@pytest.mark.parametrize(
    "raw",
    [{"A": 0.1, "B": 0.2}, {"A": 0.1, "B": 0.1, "C": 0.1}],
)
def test_too_few_names_or_no_dispersion_gives_no_exposures(raw: dict[str, float]) -> None:
    assert momentum_exposures(raw, dict.fromkeys(raw, 0.01)) == {}


def test_names_without_a_positive_vol_are_dropped() -> None:
    raw = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}
    sigma = {"A": 0.01, "B": 0.01, "C": 0.01, "D": 0.0}

    assert set(momentum_exposures(raw, sigma)) == {"A", "B", "C"}
