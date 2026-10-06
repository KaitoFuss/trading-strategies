"""Measurements that say whether the factor model can carry position sizing.

The three questions, from the 2026-10-06 spike: does the diagonal ``D`` hide
shared residual risk, is the momentum factor really separate from the macro
block, and are the betas stable enough to size on. Attribution tolerates all
three being wrong; an optimizer does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from backtester.core.events import Ticker

from trading_strategies.risk.attribution import iter_market_events
from trading_strategies.risk.config import FactorModelConfig
from trading_strategies.risk.model import MOMENTUM, FactorRiskModel, RiskModelSnapshot
from trading_strategies.utils.streaming import EwmCovariance, FloatArray

Loadings = Sequence[tuple[tuple[Ticker, ...], tuple[str, ...], FloatArray]]


@dataclass(frozen=True)
class ResidualReport:
    correlations: pd.DataFrame
    share: pd.Series[float]
    masked: tuple[Ticker, ...]

    def worst_pairs(self, n: int = 12) -> pd.Series[float]:
        pairs: dict[str, float] = {}
        names = list(self.correlations.index)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                value = float(self.correlations.loc[a, b])
                if np.isfinite(value):
                    pairs[f"{a}/{b}"] = value
        series = pd.Series(pairs, dtype=float)
        if series.empty:
            return series
        return series.reindex(series.abs().sort_values(ascending=False).index).head(n)


def residual_correlations(
    snapshot: RiskModelSnapshot,
    empirical_cov: FloatArray,
    tickers: Sequence[Ticker],
    *,
    min_share: float = 0.05,
) -> ResidualReport:
    """Correlations of ``S - B F B'``.

    ``empirical_cov`` is the long-span covariance over ``tickers``; it is
    rescaled into the snapshot's short-span vol units the same way the model
    rescales, so the two sides are comparable. The empirical signal this
    check exists to find lives entirely in the off-diagonal ``correlations``
    block: two assets whose residuals move together after the model claims
    they are independent. ``share`` is deliberately a *model* quantity
    (``D_i / sigma_i**2``, the fraction of each asset's own variance the
    model leaves unexplained by the factor block) used only to decide which
    assets to mask -- an asset that *is* a factor has ~zero modeled residual,
    and its residual correlations are noise over noise, so it is reported in
    ``masked`` instead of ``correlations``.

    A residual correlation above 1 in magnitude is possible, and
    informative, not a bug: it means ``D`` is too small to be consistent
    with the observed cross-covariance.
    """
    index = {ticker: i for i, ticker in enumerate(tickers)}
    rows = [index[ticker] for ticker in snapshot.tickers]
    empirical = empirical_cov[np.ix_(rows, rows)]
    scale = snapshot.sigma / np.sqrt(np.diag(empirical))
    scaled = empirical * np.outer(scale, scale)

    residual = scaled - snapshot.B @ snapshot.F @ snapshot.B.T
    share = np.diag(residual) / np.diag(scaled)
    keep = np.flatnonzero(share > min_share)
    masked = tuple(t for t, s in zip(snapshot.tickers, share, strict=True) if s <= min_share)

    block = residual[np.ix_(keep, keep)]
    variance = np.clip(np.diag(block), 1e-18, None)
    correlations = block / np.sqrt(np.outer(variance, variance))
    np.fill_diagonal(correlations, np.nan)
    names = [snapshot.tickers[i] for i in keep]
    return ResidualReport(
        correlations=pd.DataFrame(correlations, index=names, columns=names),
        share=pd.Series(share, index=list(snapshot.tickers), dtype=float),
        masked=masked,
    )


def exposure_collinearity(snapshot: RiskModelSnapshot) -> pd.Series[float]:
    """Cross-sectional correlation of the momentum exposure with each macro
    beta. Near-collinear ``B`` columns make an optimizer build large offsetting
    positions, and a bound on one factor then binds on another."""
    if snapshot.momentum is None or MOMENTUM not in snapshot.factors:
        return pd.Series(dtype=float)
    macro_names = [name for name in snapshot.factors if name != MOMENTUM]
    values = {
        name: float(np.corrcoef(snapshot.momentum, snapshot.macro_betas[:, k])[0, 1])
        for k, name in enumerate(macro_names)
    }
    series = pd.Series(values, dtype=float)
    return series.reindex(series.abs().sort_values(ascending=False).index)


def loading_drift(history: Loadings) -> pd.DataFrame:
    """How much each factor's betas move between sampled bars, against their
    typical size. A factor whose betas jump turns into turnover the strategy
    never asked for."""
    changes: dict[str, list[float]] = {}
    levels: dict[str, list[float]] = {}
    previous: dict[tuple[Ticker, str], float] = {}
    for tickers, factors, matrix in history:
        for k, factor in enumerate(factors):
            for n, ticker in enumerate(tickers):
                value = float(matrix[n, k])
                key = (ticker, factor)
                if key in previous:
                    changes.setdefault(factor, []).append(abs(value - previous[key]))
                levels.setdefault(factor, []).append(abs(value))
                previous[key] = value
    frame = pd.DataFrame(
        {
            "median_abs_change": {k: float(np.median(v)) for k, v in changes.items()},
            "median_level": {k: float(np.median(v)) for k, v in levels.items()},
        }
    )
    frame["drift_ratio"] = frame["median_abs_change"] / frame["median_level"]
    return frame.sort_values("drift_ratio", ascending=False)


@dataclass(frozen=True)
class DiagnosticsReplay:
    factor_returns: pd.DataFrame
    loadings: list[tuple[tuple[Ticker, ...], tuple[str, ...], FloatArray]]
    final_snapshot: RiskModelSnapshot
    final_empirical_cov: FloatArray


def replay_for_diagnostics(
    data: Path,
    tickers: Sequence[Ticker],
    model_config: FactorModelConfig,
    *,
    sample_every: int = 5,
) -> DiagnosticsReplay:
    """One causal pass over the price data, collecting what the three
    diagnostics need. The long-span covariance is re-estimated here because the
    model does not expose its own ``S``."""
    model = FactorRiskModel(tickers, model_config)
    empirical = EwmCovariance(
        tickers, span=model_config.corr_span, min_periods=model_config.corr_min_periods
    )
    last_close: dict[Ticker, float] = {}
    factor_returns: list[dict[str, float]] = []
    loadings: list[tuple[tuple[Ticker, ...], tuple[str, ...], FloatArray]] = []

    for i, event in enumerate(iter_market_events(data, tickers)):
        closes = {ticker: bar.close for ticker, bar in event.bars.items()}
        snapshot = model.snapshot
        simple = {t: c / last_close[t] - 1.0 for t, c in closes.items() if t in last_close}
        logs = {t: float(np.log(c / last_close[t])) for t, c in closes.items() if t in last_close}
        if snapshot.ready and simple:
            returns = snapshot.factor_returns(simple)
            factor_returns.append(dict(zip(snapshot.factors, returns.tolist(), strict=True)))
            if i % sample_every == 0:
                loadings.append((snapshot.tickers, snapshot.factors, snapshot.B))
        last_close.update(closes)
        empirical.update(logs)
        model.update(closes)

    return DiagnosticsReplay(
        factor_returns=pd.DataFrame(factor_returns),
        loadings=loadings,
        final_snapshot=model.snapshot,
        final_empirical_cov=empirical.covariance(),
    )
