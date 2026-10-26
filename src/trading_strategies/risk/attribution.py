"""Holdings-based factor attribution. Per bar t, with the weights that earned
bar t's return (``PerformanceTracker.weights_history``) and the model
snapshot built from data up to bar t-1:

- exposures ``x = B^T w``
- risk shares ``x_k (F x)_k / var`` and ``w^T D w / var``
- return contributions ``x_k * f_k,t``, plus ``RESIDUAL`` = portfolio return
  minus their sum (specific returns, cash, costs, intraday trading).

``edge_decomposition`` regroups the same return into five buckets that answer
"what earned it": structural beta (the full-sample mean exposure times each
macro factor's return -- what a passive holder of the average tilt earns),
factor timing (the deviation from that mean times the factor return; summed
over the sample it is ``T * cov(x_k, f_k)``), momentum, asset selection
(``w'(r - B f)``) and other (the rest of ``RESIDUAL``: log-return aggregation,
intraday trading, costs). The mean is ex post on purpose: this is a
description of the backtest, not a trading input.

``factor_regression`` is the returns-based cross-check (AQR, "Measuring
Portfolio Factor Exposures: A Practical Guide", Exhibit 2): a full-sample OLS of
the portfolio's arithmetic returns on the factor returns, with each factor's
contribution ``beta_k * mean(f_k)`` and the intercept as alpha. It needs no
holdings and no ``B``. A static beta cannot see timing, so timing P&L lands in
its alpha.

Contributions sum arithmetically over time; per bar the decomposition is
exact by construction. Holdings-based rather than a returns regression,
because a trend strategy's exposures move on purpose."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from backtester.core.events import MarketEvent, Ticker
from backtester.data.frame_market_data import FrameMarketData
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR, PerformanceTracker

from trading_strategies.risk.asset_analytics import (
    AssetRisk,
    AssetRiskAccumulator,
    ModelFitAccumulator,
)
from trading_strategies.risk.config import FactorModelConfig
from trading_strategies.risk.factors import factor_tickers, load_factors
from trading_strategies.risk.model import MOMENTUM, FactorRiskModel

RESIDUAL = "Specific + other"
STRUCTURAL = "Structural beta"
TIMING = "Factor timing"
SELECTION = "Asset selection"
OTHER = "Other"


@dataclass(frozen=True)
class FactorRegression:
    table: pd.DataFrame  # per factor: beta, t_stat, contribution (annualized)
    alpha: float  # annualized
    alpha_t: float
    r_squared: float


@dataclass(frozen=True)
class AttributionResult:
    exposures: pd.DataFrame
    risk_decomposition: pd.DataFrame
    contributions: pd.DataFrame
    portfolio_returns: pd.Series[float]
    predicted_vol: pd.Series[float]
    factor_returns: pd.DataFrame  # log, like the model; the edge buckets need these
    simple_factor_returns: pd.DataFrame  # W'(e^r - 1): arithmetic, for factor_regression
    specific_returns: pd.Series[float]  # w'(r - B f)
    asset_fit: pd.DataFrame = field(default_factory=pd.DataFrame)  # per asset, no weights
    asset_risk: AssetRisk = field(default_factory=AssetRisk.empty)

    def standardized_returns(self) -> pd.Series[float]:
        live = self.predicted_vol > 0
        return self.portfolio_returns[live] / self.predicted_vol[live]

    def bias_statistic(self) -> float:
        z = self.standardized_returns()
        return float(z.std()) if len(z) > 1 else math.nan

    def rolling_bias(self, window: int = 250) -> pd.Series[float]:
        return self.standardized_returns().rolling(window, min_periods=window // 2).std()

    def structural_returns(self) -> pd.DataFrame:
        """Per bar and factor: full-sample mean exposure times the factor return."""
        return self.factor_returns[self.exposures.columns] * self.exposures.mean()

    def timing_returns(self) -> pd.DataFrame:
        """Per bar and factor: (exposure - its full-sample mean) times the factor return."""
        exposures = self.exposures
        return (exposures - exposures.mean()) * self.factor_returns[exposures.columns]

    def edge_decomposition(self) -> pd.DataFrame:
        """Five buckets per bar that sum exactly to ``portfolio_returns``.
        Momentum stays one bucket; ``factor_summary`` shows its own split."""
        structural = self.structural_returns()
        timing = self.timing_returns()
        macro = [c for c in self.exposures.columns if c != MOMENTUM]
        momentum = (
            self.contributions[MOMENTUM]
            if MOMENTUM in self.contributions
            else pd.Series(0.0, index=self.portfolio_returns.index)
        )
        explained = self.contributions.drop(columns=RESIDUAL, errors="ignore").sum(axis=1)
        return pd.DataFrame(
            {
                STRUCTURAL: structural[macro].sum(axis=1),
                TIMING: timing[macro].sum(axis=1),
                MOMENTUM: momentum,
                SELECTION: self.specific_returns,
                OTHER: self.portfolio_returns - explained - self.specific_returns,
            },
            index=self.portfolio_returns.index,
        )

    def edge_summary(self) -> pd.DataFrame:
        """Per bucket: annualized mean contribution and its t-stat."""
        edge = self.edge_decomposition()
        return pd.DataFrame(
            {"annualized": edge.mean() * TRADING_DAYS_PER_YEAR, "t_stat": _t_stats(edge)}
        )

    def factor_summary(self) -> pd.DataFrame:
        """Per factor: annualized structural and timing contribution, and the
        t-stat of the timing part."""
        timing = self.timing_returns()
        return pd.DataFrame(
            {
                "structural": self.structural_returns().mean() * TRADING_DAYS_PER_YEAR,
                "timing": timing.mean() * TRADING_DAYS_PER_YEAR,
                "t_timing": _t_stats(timing),
            }
        )

    def factor_regression(self) -> FactorRegression:
        """OLS ``R_t = alpha + sum_k beta_k f_k,t + e_t``, arithmetic on both
        sides: log factor returns would leave about ``sigma_p^2 / 2`` in alpha.
        Contributions plus alpha equal the mean return."""
        factors = list(self.simple_factor_returns.columns)
        y = np.expm1(self.portfolio_returns.to_numpy(dtype=np.float64))
        n, k = len(y), len(factors) + 1
        if n <= k:
            nan = pd.Series(math.nan, index=factors, dtype=float)
            table = pd.DataFrame({"beta": nan, "t_stat": nan, "contribution": nan})
            return FactorRegression(table, math.nan, math.nan, math.nan)
        f = self.simple_factor_returns.to_numpy(dtype=np.float64)
        x = np.column_stack([np.ones(n), f])
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ coef
        variance = float(residual @ residual) / (n - k)
        stderr = np.sqrt(np.diag(np.linalg.pinv(x.T @ x)) * variance)
        t = np.divide(coef, stderr, out=np.zeros_like(coef), where=stderr > 0)
        total = float(((y - y.mean()) ** 2).sum())
        table = pd.DataFrame(
            {
                "beta": coef[1:],
                "t_stat": t[1:],
                "contribution": coef[1:] * f.mean(axis=0) * TRADING_DAYS_PER_YEAR,
            },
            index=factors,
        )
        r_squared = 1.0 - float(residual @ residual) / total if total > 0 else math.nan
        return FactorRegression(
            table, float(coef[0]) * TRADING_DAYS_PER_YEAR, float(t[0]), r_squared
        )


def _t_stats(frame: pd.DataFrame) -> pd.Series[float]:
    """``mean / std * sqrt(n)`` per column; 0 for a column with no variation."""
    std: pd.Series[float] = frame.std()
    t: pd.Series[float] = frame.mean() / std * math.sqrt(len(frame))
    return t.where(std > 0, 0.0).fillna(0.0)


def run_attribution(
    events: Iterable[MarketEvent],
    weights_history: Sequence[tuple[datetime, Mapping[Ticker, float]]],
    equity_history: Sequence[tuple[datetime, float]],
    model: FactorRiskModel,
    assets: Sequence[Ticker] | None = None,
) -> AttributionResult:
    """``assets`` picks the rows of ``asset_fit`` and ``asset_risk``;
    ``None`` means the model's own universe."""
    weights_at = dict(weights_history)
    # Log returns, matching FactorRiskModel's own convention (model.py's
    # update()) -- B/F/D/sigma are all fit on log returns, so every return fed
    # into factor_returns() or compared against predicted_vol needs the same
    # units. equity > 0 guards math.log against a wiped-out/negative equity
    # bar (simple-return division wouldn't raise there, log does).
    returns_at = {
        timestamp: math.log(equity / previous)
        for (_, previous), (timestamp, equity) in itertools.pairwise(equity_history)
        if previous > 0 and equity > 0
    }
    index: list[datetime] = []
    exposures: list[dict[str, float]] = []
    decomposition: list[dict[str, float]] = []
    contributions: list[dict[str, float]] = []
    portfolio: list[float] = []
    predicted: list[float] = []
    factor_returns: list[dict[str, float]] = []
    simple_factor_returns: list[dict[str, float]] = []
    specific: list[float] = []
    last_close: dict[Ticker, float] = {}
    fit = ModelFitAccumulator()
    risk = AssetRiskAccumulator()

    for event in events:
        closes = {ticker: bar.close for ticker, bar in event.bars.items()}
        snapshot = model.snapshot  # data up to the previous bar
        timestamp = event.timestamp
        if snapshot.ready:
            asset_returns = {
                t: math.log(c / last_close[t]) for t, c in closes.items() if t in last_close
            }
            f = snapshot.factor_returns(asset_returns)
            fit.observe(snapshot, asset_returns, f)  # every ready bar, weights or not
        if snapshot.ready and timestamp in weights_at and timestamp in returns_at:
            leg_weights = weights_at[timestamp]
            w = np.array([leg_weights.get(t, 0.0) for t in snapshot.tickers], dtype=np.float64)
            x = snapshot.exposures(w)
            simple_returns = {t: math.expm1(r) for t, r in asset_returns.items()}
            f_simple = snapshot.factor_returns(simple_returns)
            explained = x * f
            r = np.array([asset_returns.get(t, 0.0) for t in snapshot.tickers], dtype=np.float64)
            portfolio_return = returns_at[timestamp]
            index.append(timestamp)
            exposures.append(dict(zip(snapshot.factors, x.tolist(), strict=True)))
            decomposition.append(snapshot.risk_decomposition(w))
            contributions.append(
                dict(zip(snapshot.factors, explained.tolist(), strict=True))
                | {RESIDUAL: portfolio_return - float(explained.sum())}
            )
            portfolio.append(portfolio_return)
            predicted.append(math.sqrt(snapshot.portfolio_variance(w)))
            factor_returns.append(dict(zip(snapshot.factors, f.tolist(), strict=True)))
            simple_factor_returns.append(
                dict(zip(snapshot.factors, f_simple.tolist(), strict=True))
            )
            specific.append(float(w @ np.nan_to_num(r)) - float(explained.sum()))
            risk.observe(timestamp, snapshot, w, np.nan_to_num(r))
        last_close.update(closes)
        model.update(closes)

    rows = list(model.snapshot.tickers if assets is None else assets)
    dates = pd.DatetimeIndex(index)
    contribution_frame = pd.DataFrame(contributions, index=dates).fillna(0.0)
    if not contribution_frame.empty:
        ordered = [c for c in contribution_frame.columns if c != RESIDUAL] + [RESIDUAL]
        contribution_frame = contribution_frame[ordered]
    return AttributionResult(
        exposures=pd.DataFrame(exposures, index=dates).fillna(0.0),
        risk_decomposition=pd.DataFrame(decomposition, index=dates).fillna(0.0),
        contributions=contribution_frame,
        portfolio_returns=pd.Series(portfolio, index=dates, dtype=float),
        predicted_vol=pd.Series(predicted, index=dates, dtype=float),
        factor_returns=pd.DataFrame(factor_returns, index=dates).fillna(0.0),
        simple_factor_returns=pd.DataFrame(simple_factor_returns, index=dates).fillna(0.0),
        specific_returns=pd.Series(specific, index=dates, dtype=float),
        asset_fit=fit.result(rows),
        asset_risk=risk.result(rows),
    )


def iter_market_events(data: Path, tickers: Sequence[Ticker]) -> Iterator[MarketEvent]:
    market_data = FrameMarketData(data, tickers=list(tickers))
    while (event := market_data.get_next_bar()) is not None:
        yield event


def attribute_backtest(
    data: Path,
    tickers: Sequence[Ticker],
    model_config: FactorModelConfig,
    trackers: Mapping[str, PerformanceTracker],
) -> dict[str, AttributionResult]:
    """Replay the causal model over the backtest's price data once per leg.
    The replay gives the same snapshots a live run would.

    The model's own ticker universe is ``tickers`` (so every actual holding
    gets a slot in ``w``, see ``RiskModelSnapshot.exposures``) unioned with
    every ticker the factor definitions reference (so a factor exists
    whenever its ETFs have price data, not only when the backtest happens to
    trade them)."""
    definitions = load_factors(Path(model_config.factors_file), granular=model_config.granular)
    universe = tuple(dict.fromkeys((*tickers, *factor_tickers(definitions))))
    return {
        label: run_attribution(
            iter_market_events(data, universe),
            tracker.weights_history,
            tracker.mark_to_market_history,
            FactorRiskModel(universe, model_config),
            assets=tickers,
        )
        for label, tracker in trackers.items()
    }
