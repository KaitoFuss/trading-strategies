import numpy as np
import numpy.typing as npt
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize


def mean_variance_weights(
    expected_returns: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64],
    risk_aversion: float,
    gross_cap: float,
) -> npt.NDArray[np.float64]:
    """max_w w'expected_returns - (risk_aversion/2) w'covariance w
    s.t. sum(w) == 0, ||w||_1 <= gross_cap.

    scipy.optimize.minimize solves the negated objective. The L1 gross cap
    is a nonsmooth constraint (abs() has no derivative at 0); SLSQP falls
    back to a numerical Jacobian for it, fine at this scale (~26 assets).
    """
    n = len(expected_returns)

    def objective(w: npt.NDArray[np.float64]) -> float:
        return float(-w @ expected_returns + (risk_aversion / 2) * w @ covariance @ w)

    def objective_grad(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return -expected_returns + risk_aversion * covariance @ w

    dollar_neutral = LinearConstraint(np.ones(n), lb=0.0, ub=0.0)
    gross_constraint = NonlinearConstraint(lambda w: np.abs(w).sum(), lb=-np.inf, ub=gross_cap)

    result = minimize(
        objective,
        x0=np.zeros(n),
        jac=objective_grad,
        method="SLSQP",
        constraints=[dollar_neutral, gross_constraint],
    )
    weights: npt.NDArray[np.float64] = result.x
    return weights
