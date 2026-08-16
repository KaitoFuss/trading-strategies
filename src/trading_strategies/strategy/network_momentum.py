import numpy as np
from backtester.core.events import MarketEvent, SignalEvent, Ticker

from trading_strategies.features.online_momentum import OnlineTickerFeatures


def _cross_sectional_zscore(raw: dict[Ticker, list[float]]) -> dict[Ticker, list[float]]:
    """Standardize each of the 8 loadings to zero mean, unit std across
    today's tickers. Population std (ddof=0): today's universe is the whole
    cross-section, not a sample of a larger one. Empty (not degenerate) with
    fewer than 2 tickers, since a single point has no meaningful spread."""
    if len(raw) < 2:
        return {}
    matrix = np.array(list(raw.values()))
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    standardized = np.divide(matrix - mean, std, out=np.zeros_like(matrix), where=std != 0)
    return dict(zip(raw.keys(), standardized.tolist(), strict=True))


def _factor_portfolio_returns(
    loadings: dict[Ticker, list[float]], returns: dict[Ticker, float]
) -> np.ndarray | None:
    """One realized return per factor: a long-short portfolio weighted
    directly by yesterday's standardized loading, normalized to unit gross
    per factor (sum of |weight| = 1), applied to today's realized return."""
    tickers = [ticker for ticker in loadings if ticker in returns]
    if len(tickers) < 2:
        return None
    loading_matrix = np.array([loadings[ticker] for ticker in tickers])  # tickers x 8
    return_vector = np.array([returns[ticker] for ticker in tickers])
    gross = np.abs(loading_matrix).sum(axis=0)  # 8-vector
    weights = np.divide(loading_matrix, gross, out=np.zeros_like(loading_matrix), where=gross != 0)
    result: np.ndarray = weights.T @ return_vector
    return result


class NetworkMomentumStrategy:
    """Grinold-Kahn factor model, incremental: this bar's B (8 standardized
    momentum loadings per ticker) dotted with mu_f becomes each ticker's
    score. mu_f is the expanding-average realized return of each factor's
    own long-short mimicking portfolio (weighted by yesterday's standardized
    loading) — no regression: simpler than a daily cross-sectional OLS, at
    the cost of not separating the 8 factors' overlapping effects on
    returns the way a multivariate fit would.

    Runs entirely on O(1)/O(window) state built bar by bar from
    ``event.bars`` — no precomputed history, no unbounded per-ticker state.
    Each ticker's loadings come from ``OnlineTickerFeatures``, the native
    incremental rewrite of ``momentum_features.py`` (recomputing the pandas
    version over a growing series every bar was too slow at real backtest
    scale — verified empirically, not assumed).

    Causality: a factor portfolio's return for day t-1 (weighted by
    yesterday's B, realized against the return from t-1 to t) only becomes
    computable once today's return is known, so it is folded into mu_f
    within the same bar it becomes available — never a bar earlier or
    later. mu_f for today therefore includes that return (which needs
    today's close) but nothing that would need tomorrow's, matching
    backtester's existing same-close signal-and-execution convention used by
    every other strategy here.
    """

    def __init__(self) -> None:
        self._online_features: dict[Ticker, OnlineTickerFeatures] = {}
        self._last_price: dict[Ticker, float] = {}
        self._last_loadings: dict[Ticker, list[float]] | None = None
        self._factor_return_history: list[np.ndarray] = []

    def process_market(self, event: MarketEvent) -> SignalEvent:
        raw_loadings: dict[Ticker, list[float]] = {}
        returns: dict[Ticker, float] = {}
        for ticker, bar in event.bars.items():
            price = bar.close
            if ticker in self._last_price:
                returns[ticker] = price / self._last_price[ticker] - 1
            self._last_price[ticker] = price

            features = self._online_features.setdefault(ticker, OnlineTickerFeatures())
            loadings = features.update(price)
            if loadings is not None:
                raw_loadings[ticker] = loadings

        standardized = _cross_sectional_zscore(raw_loadings)

        if self._last_loadings is not None:
            factor_returns = _factor_portfolio_returns(self._last_loadings, returns)
            if factor_returns is not None:
                self._factor_return_history.append(factor_returns)

        scores = self._score(standardized)
        self._last_loadings = standardized
        return SignalEvent(timestamp=event.timestamp, scores=scores)

    def _score(self, standardized: dict[Ticker, list[float]]) -> dict[Ticker, float]:
        if not self._factor_return_history:
            return {}
        mu_f = np.mean(self._factor_return_history, axis=0)
        return {
            ticker: float(np.array(loadings) @ mu_f) for ticker, loadings in standardized.items()
        }
