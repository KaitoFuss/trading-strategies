from __future__ import annotations

import numpy as np
import pytest

from trading_strategies.risk.eigen_adjustment import (
    eigen_adjust,
    ewm_covariance,
    ewm_weights,
    simulated_eigen_bias,
)
from trading_strategies.utils.streaming import EwmCovariance


def test_ewm_weights_decay_from_the_newest_bar() -> None:
    weights = ewm_weights(span=9, window=5)

    assert len(weights) == 5
    assert weights[-1] == 1.0
    np.testing.assert_allclose(weights[:-1] / weights[1:], 0.8)  # 1 - 2/(9+1)


def test_small_eigenvalues_are_underestimated_large_ones_overestimated() -> None:
    """Known property of sample eigenvalues (the reason for the whole step):
    with 18 equal true eigenvalues and about 250 effective observations, the
    prototype measured v = 1.31 for the smallest and 0.80 for the largest."""
    v = simulated_eigen_bias(np.ones(18), ewm_weights(250, 750), 200, np.random.default_rng(0))

    assert v[0] > 1.15
    assert v[-1] < 0.9
    assert np.all(np.diff(v) < 0.02)  # roughly decreasing from small to large


def test_spread_spectrum_has_a_smaller_bias() -> None:
    """Prototype: v = 1.10 smallest, 0.99 largest for a 1e-3..1 spectrum."""
    v = simulated_eigen_bias(
        np.geomspace(1e-3, 1.0, 18), ewm_weights(250, 750), 200, np.random.default_rng(0)
    )

    assert 1.05 < v[0] < 1.2
    assert 0.95 < v[-1] < 1.05


def test_single_factor_bias_is_close_to_one() -> None:
    v = simulated_eigen_bias(np.ones(1), ewm_weights(250, 750), 200, np.random.default_rng(0))

    assert v.shape == (1,)
    assert v[0] == pytest.approx(1.0, abs=0.02)


def test_near_singular_spectrum_stays_finite() -> None:
    v = simulated_eigen_bias(
        np.geomspace(1e-8, 1.0, 10), ewm_weights(100, 300), 50, np.random.default_rng(0)
    )

    assert np.isfinite(v).all()


def test_simulated_bias_is_reproducible_with_a_seed() -> None:
    weights = ewm_weights(100, 300)
    first = simulated_eigen_bias(np.ones(5), weights, 20, np.random.default_rng(7))
    second = simulated_eigen_bias(np.ones(5), weights, 20, np.random.default_rng(7))

    np.testing.assert_array_equal(first, second)


def _random_covariance(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, k))
    return a @ a.T / k + 1e-3 * np.eye(k)


def test_eigen_adjust_keeps_the_eigenvectors_and_scales_the_eigenvalues() -> None:
    f = _random_covariance(6, 0)
    bias = np.linspace(1.3, 0.95, 6)  # ascending eigenvalue order
    values, vectors = np.linalg.eigh(f)

    adjusted = eigen_adjust(f, bias, scale=1.4)

    rotated = vectors.T @ adjusted @ vectors
    np.testing.assert_allclose(rotated, np.diag(np.diag(rotated)), atol=1e-12)
    gamma = 1.4 * (bias - 1.0) + 1.0
    np.testing.assert_allclose(np.diag(rotated), gamma**2 * values, rtol=1e-10)
    assert np.linalg.eigvalsh(adjusted).min() > 0


def test_eigen_adjust_without_bias_is_the_identity() -> None:
    f = _random_covariance(4, 1)

    np.testing.assert_allclose(eigen_adjust(f, np.ones(4), scale=1.4), f, rtol=1e-10)
    np.testing.assert_allclose(eigen_adjust(f, np.linspace(1.2, 0.9, 4), scale=0.0), f, rtol=1e-10)


def test_window_shorter_than_the_factor_count_is_rejected() -> None:
    """A re-estimate from fewer bars than factors is rank-deficient: its zero
    eigenvalues would make v and F~ NaN and reach the optimizer silently."""
    with pytest.raises(ValueError, match="window"):
        simulated_eigen_bias(np.ones(18), ewm_weights(250, 10), 5, np.random.default_rng(0))


def test_simulation_re_estimates_like_the_model() -> None:
    """The simulated re-estimate must be the model's own estimator
    (EwmCovariance: demeaned, bias-corrected); without the bias correction F~
    came out about 0.55% too high."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 3))
    streaming = EwmCovariance(["a", "b", "c"], span=40, min_periods=2)
    for row in x:
        streaming.update(dict(zip("abc", row.tolist(), strict=True)))

    np.testing.assert_allclose(
        ewm_covariance(x, ewm_weights(40, 120)), streaming.covariance(), rtol=1e-10
    )
