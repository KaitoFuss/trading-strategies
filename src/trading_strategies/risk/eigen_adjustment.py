"""Eigenfactor risk adjustment (Menchero, Wang & Orr, "Improving Risk
Forecasts for Optimized Portfolios", FAJ 2012; spec:
docs/superpowers/specs/2026-10-25-risk-model-calibration-design.md, step 3).

A sample covariance underestimates its smallest eigenvalues and overestimates
its largest. An optimizer loads exactly the small eigen-directions, so its
realized risk comes out above the forecast. The fix measures that bias by
simulation: draw returns from the estimated covariance itself, re-estimate
with the model's own EWM weights, and compare the "true" variance of each
simulated eigenportfolio with its estimated variance. The square root of the
mean ratio is ``v(k)``; the adjustment scales eigenvalue ``k`` by
``gamma(k)**2`` with ``gamma = a (v - 1) + 1``, where ``a`` (the paper's 1.4)
corrects for the simulation starting from an already-biased estimate."""

from __future__ import annotations

import numpy as np

from trading_strategies.utils.streaming import FloatArray


def ewm_weights(span: int, window: int) -> FloatArray:
    """EWM observation weights, oldest first, newest weight 1. Truncated at
    ``window`` bars (3 spans leave about e^-6 of the weight out)."""
    decay = 1.0 - 2.0 / (span + 1.0)
    return decay ** np.arange(window - 1, -1, -1, dtype=np.float64)


def simulated_eigen_bias(
    eigenvalues: FloatArray, weights: FloatArray, simulations: int, rng: np.random.Generator
) -> FloatArray:
    """``v(k) = sqrt(mean_m true_m(k) / estimated_m(k))`` per eigenvalue, in
    the (ascending) order of ``eigenvalues``. Simulated in the eigenbasis,
    where the true covariance is ``diag(eigenvalues)`` -- the bias does not
    depend on the eigenvectors. The re-estimate is the weighted, demeaned
    covariance, like the model's EWM estimator."""
    w = weights / weights.sum()
    ratios = np.empty((simulations, len(eigenvalues)))
    sd = np.sqrt(eigenvalues)
    for m in range(simulations):
        x = rng.standard_normal((len(w), len(eigenvalues))) * sd
        centered = x - w @ x
        estimated, vectors = np.linalg.eigh((centered * w[:, None]).T @ centered)
        true = np.einsum("ik,i,ik->k", vectors, eigenvalues, vectors)
        ratios[m] = true / estimated
    return np.sqrt(ratios.mean(axis=0))


def eigen_adjust(f: FloatArray, bias: FloatArray, scale: float) -> FloatArray:
    """``U diag(gamma**2 * lambda) U'`` with ``gamma = scale (bias - 1) + 1``.
    ``bias`` is in ascending eigenvalue order, the order ``eigh`` returns."""
    values, vectors = np.linalg.eigh((f + f.T) / 2)
    gamma = scale * (bias - 1.0) + 1.0
    return (vectors * (gamma**2 * values)) @ vectors.T
