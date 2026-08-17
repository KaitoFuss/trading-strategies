import numpy as np
from backtester.core.events import Ticker


def _halflife_alpha(halflife: float) -> float:
    """Weight on the newest observation for a halflife (in bars): after
    ``halflife`` bars an observation's weight has decayed by half."""
    return float(1.0 - 0.5 ** (1.0 / halflife))


class _EwCovariance:
    """Exponentially-weighted mean and covariance of a vector, maintained
    incrementally (West/Finch form) with a constant learning rate set by
    ``halflife``. O(k^2) state for a k-vector, no history retained. ``value``
    is None until a second observation exists, matching this repo's rule that
    a single point has no defined (co)variance.
    """

    def __init__(self, halflife: float) -> None:
        self._alpha = _halflife_alpha(halflife)
        self._mean: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._n = 0

    def update(self, x: np.ndarray) -> None:
        self._n += 1
        if self._mean is None:
            self._mean = x.astype(float)
            self._cov = np.zeros((x.size, x.size))
            return
        assert self._cov is not None
        diff = x - self._mean
        self._mean = self._mean + self._alpha * diff
        self._cov = (1 - self._alpha) * (self._cov + self._alpha * np.outer(diff, diff))

    @property
    def value(self) -> np.ndarray | None:
        return self._cov if self._n >= 2 else None


class FactorRiskModel:
    """Structural (Grinold-Kahn / BARRA) risk model shared between the
    strategy that produces the factor structure and the portfolio that
    consumes the covariance.

    The asset covariance is reconstructed from the factor structure rather
    than estimated as a raw p x p sample matrix:

        Sigma = B Sigma_f B' + D

    where B (p x k) are the current standardized factor loadings, Sigma_f
    (k x k) is the EWMA covariance of the factor-mimicking-portfolio returns,
    and D is the diagonal of per-asset idiosyncratic variances (EWMA of the
    residuals r_i - B_i . f). With k=8 factors this only ever estimates an
    8x8 covariance plus a diagonal, sidestepping the near-singular p~=26,
    n~=60 sample covariance entirely: B Sigma_f B' is rank <= k, but the
    strictly positive diagonal D lifts every eigenvalue off zero, so Sigma is
    positive-definite by construction -- the diagonal *is* the eigenvalue
    floor, with no statistical shrinkage needed.

    The strategy owns the update side (it computes B and f each bar); the
    portfolio owns the read side (``covariance``). backtester calls
    ``process_market`` before ``process_signal`` within a bar, so the model is
    always updated with this bar's data before the portfolio reads it.
    """

    def __init__(self, halflife: float = 126.0) -> None:
        self._halflife = halflife
        self._loadings: dict[Ticker, list[float]] = {}
        self._factor_cov = _EwCovariance(halflife)
        self._idiosyncratic: dict[Ticker, _EwCovariance] = {}

    def update(
        self,
        loadings: dict[Ticker, list[float]],
        prev_loadings: dict[Ticker, list[float]] | None,
        factor_returns: np.ndarray | None,
        returns: dict[Ticker, float],
    ) -> None:
        """Fold in one bar. ``loadings`` is this bar's B (latest exposures,
        used for reconstruction); ``factor_returns`` is this bar's realized
        factor return vector f (None during warm-up); residuals use
        ``prev_loadings`` (the B that f was formed from) against ``returns``."""
        self._loadings = loadings
        if factor_returns is None:
            return
        self._factor_cov.update(factor_returns)
        if prev_loadings is None:
            return
        for ticker, realized in returns.items():
            b_prev = prev_loadings.get(ticker)
            if b_prev is None:
                continue
            residual = realized - float(np.dot(b_prev, factor_returns))
            self._idiosyncratic.setdefault(ticker, _EwCovariance(self._halflife)).update(
                np.array([residual])
            )

    def factor_covariance(self) -> np.ndarray | None:
        """Sigma_f, the k x k EWMA factor covariance (None before warm-up)."""
        return self._factor_cov.value

    def idiosyncratic_variance(self, ticker: Ticker) -> float | None:
        acc = self._idiosyncratic.get(ticker)
        if acc is None or acc.value is None:
            return None
        return float(acc.value[0, 0])

    def covariance(self, tickers: list[Ticker]) -> np.ndarray:
        """Sigma = B Sigma_f B' + D for ``tickers``, in the given order. A
        ticker without its own idiosyncratic estimate yet falls back to the
        median of the known ones; every variance is floored strictly positive
        so Sigma stays positive-definite."""
        loadings = np.array([self._loadings[ticker] for ticker in tickers])  # p x k
        variances = self._idiosyncratic_diagonal(tickers)
        systematic = np.zeros((len(tickers), len(tickers)))
        sigma_f = self._factor_cov.value
        if sigma_f is not None:
            systematic = loadings @ sigma_f @ loadings.T
        return systematic + np.diag(variances)

    def _idiosyncratic_diagonal(self, tickers: list[Ticker]) -> np.ndarray:
        known = [v for t in tickers if (v := self.idiosyncratic_variance(t)) is not None and v > 0]
        fallback = float(np.median(known)) if known else 1e-12
        variances = []
        for ticker in tickers:
            v = self.idiosyncratic_variance(ticker)
            variances.append(v if v is not None and v > 0 else fallback)
        return np.array(variances)
