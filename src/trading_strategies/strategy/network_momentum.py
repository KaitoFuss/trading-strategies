import numpy as np
from backtester.core.events import MarketEvent, SignalEvent, Ticker

from trading_strategies.features.factor_model import fit_cross_sectional_ols
from trading_strategies.features.online_momentum import OnlineTickerFeatures

N_FEATURES = 8  # 5 vol-scaled returns + 3 MACD pairs
N_PARAMS = N_FEATURES + 1  # + intercept


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


class NetworkMomentumStrategy:
    """Grinold-Kahn factor model, incremental: this bar's B (8 standardized
    momentum loadings per ticker) dotted with mu_f (Fama-MacBeth expected
    factor returns) becomes each ticker's score.

    Runs entirely on O(1)/O(window) state built bar by bar from
    ``event.bars`` — no precomputed history, no unbounded per-ticker state.
    Each ticker's loadings come from ``OnlineTickerFeatures``, the native
    incremental rewrite of ``momentum_features.py`` (recomputing the pandas
    version over a growing series every bar was too slow at real backtest
    scale — verified empirically, not assumed).

    Causality: lambda_{t-1} (the daily cross-sectional regression fit using
    yesterday's B against the return realized from t-1 to t) only becomes
    computable once today's return is known, so it is fit and folded into
    mu_f within the same bar it becomes available — never a bar earlier or
    later. mu_f for today therefore includes lambda_{t-1} (which needs
    today's close) but nothing that would need tomorrow's, matching
    backtester's existing same-close signal-and-execution convention used by
    every other strategy here.
    """

    def __init__(self) -> None:
        self._online_features: dict[Ticker, OnlineTickerFeatures] = {}
        self._last_price: dict[Ticker, float] = {}
        self._last_loadings: dict[Ticker, list[float]] | None = None
        self._lambda_history: list[np.ndarray] = []

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
            self._fit_and_accumulate_lambda(returns)

        scores = self._score(standardized)
        self._last_loadings = standardized
        return SignalEvent(timestamp=event.timestamp, scores=scores)

    def _fit_and_accumulate_lambda(self, returns: dict[Ticker, float]) -> None:
        assert self._last_loadings is not None
        tickers = [ticker for ticker in self._last_loadings if ticker in returns]
        if not tickers:
            return

        features = np.array([self._last_loadings[ticker] for ticker in tickers])
        target = np.array([returns[ticker] for ticker in tickers])
        coefficients = fit_cross_sectional_ols(features, target, n_params=N_PARAMS)
        if not np.isnan(coefficients).any():
            self._lambda_history.append(coefficients)

    def _score(self, standardized: dict[Ticker, list[float]]) -> dict[Ticker, float]:
        if not self._lambda_history:
            return {}
        mu_f = np.mean(self._lambda_history, axis=0)
        return {
            ticker: float(np.array([1.0, *loadings]) @ mu_f)
            for ticker, loadings in standardized.items()
        }
