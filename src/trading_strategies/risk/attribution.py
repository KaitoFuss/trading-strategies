"""Holdings-based factor attribution. Per bar t, with the weights that earned
bar t's return (``PerformanceTracker.weights_history``) and the model
snapshot built from data up to bar t-1:

- exposures ``x = B^T w``
- risk shares ``x_k (F x)_k / var`` and ``w^T D w / var``
- return contributions ``x_k * f_k,t``, plus ``RESIDUAL`` = portfolio return
  minus their sum (specific returns, cash, costs, intraday trading).

Contributions sum arithmetically over time; per bar the decomposition is
exact by construction. Holdings-based rather than a returns regression,
because a trend strategy's exposures move on purpose."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from backtester.core.events import MarketEvent, Ticker
from backtester.data.frame_market_data import FrameMarketData
from backtester.tracker.metrics import PerformanceTracker

from trading_strategies.risk.config import FactorModelConfig
from trading_strategies.risk.factors import factor_tickers, load_factors
from trading_strategies.risk.model import FactorRiskModel

RESIDUAL = "Specific + other"


@dataclass(frozen=True)
class AttributionResult:
    exposures: pd.DataFrame
    risk_decomposition: pd.DataFrame
    contributions: pd.DataFrame
    portfolio_returns: pd.Series[float]
    predicted_vol: pd.Series[float]

    def standardized_returns(self) -> pd.Series[float]:
        live = self.predicted_vol > 0
        return self.portfolio_returns[live] / self.predicted_vol[live]

    def bias_statistic(self) -> float:
        z = self.standardized_returns()
        return float(z.std()) if len(z) > 1 else math.nan

    def rolling_bias(self, window: int = 250) -> pd.Series[float]:
        return self.standardized_returns().rolling(window, min_periods=window // 2).std()


def run_attribution(
    events: Iterable[MarketEvent],
    weights_history: Sequence[tuple[datetime, Mapping[Ticker, float]]],
    equity_history: Sequence[tuple[datetime, float]],
    model: FactorRiskModel,
) -> AttributionResult:
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
    last_close: dict[Ticker, float] = {}

    for event in events:
        closes = {ticker: bar.close for ticker, bar in event.bars.items()}
        snapshot = model.snapshot  # data up to the previous bar
        timestamp = event.timestamp
        if snapshot.ready and timestamp in weights_at and timestamp in returns_at:
            asset_returns = {
                t: math.log(c / last_close[t]) for t, c in closes.items() if t in last_close
            }
            leg_weights = weights_at[timestamp]
            w = np.array([leg_weights.get(t, 0.0) for t in snapshot.tickers], dtype=np.float64)
            x = snapshot.exposures(w)
            explained = x * snapshot.factor_returns(asset_returns)
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
        last_close.update(closes)
        model.update(closes)

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
        )
        for label, tracker in trackers.items()
    }
