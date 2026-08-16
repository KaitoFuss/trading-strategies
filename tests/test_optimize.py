import numpy as np
import pytest

from trading_strategies.optimize import mean_variance_weights


def test_mean_variance_weights_matches_closed_form_when_gross_cap_not_binding() -> None:
    # Sigma = I, expected_returns = [1, -1]: unconstrained-except-dollar-neutral
    # optimum is w = expected_returns / risk_aversion (Sigma^-1 = I), which
    # already sums to zero by symmetry.
    expected_returns = np.array([1.0, -1.0])
    covariance = np.eye(2)
    risk_aversion = 2.0

    weights = mean_variance_weights(
        expected_returns, covariance, risk_aversion=risk_aversion, gross_cap=10.0
    )

    assert weights == pytest.approx([0.5, -0.5], abs=1e-4)
    assert weights.sum() == pytest.approx(0.0, abs=1e-6)


def test_mean_variance_weights_respects_the_gross_cap_when_binding() -> None:
    expected_returns = np.array([1.0, -1.0])
    covariance = np.eye(2)

    weights = mean_variance_weights(expected_returns, covariance, risk_aversion=2.0, gross_cap=0.2)

    assert np.abs(weights).sum() == pytest.approx(0.2, abs=1e-4)
    assert weights.sum() == pytest.approx(0.0, abs=1e-6)
