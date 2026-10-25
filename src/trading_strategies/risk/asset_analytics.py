"""Per-asset statistics gathered bar by bar during the attribution replay.

``ModelFitAccumulator`` scores the model on each asset and needs no weights.
``AssetRiskAccumulator`` splits one leg's exposures and vol by asset.
Running sums keep the memory flat over long backtests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from backtester.core.events import Ticker
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR

from trading_strategies.risk.model import RiskModelSnapshot
from trading_strategies.utils.streaming import FloatArray

ANNUAL = math.sqrt(TRADING_DAYS_PER_YEAR)

BIAS = "bias"
R_SQUARED = "r_squared"
SPECIFIC_SHARE = "specific_share"

# Running sums per ticker: count, z, z^2, r, r^2, eps, eps^2, specific share
_COUNT, _Z, _Z2, _R, _R2, _EPS, _EPS2, _SHARE = range(8)


def _variance(total: float, squares: float, count: float) -> float:
    return (squares - total * total / count) / (count - 1)


class ModelFitAccumulator:
    """Per asset: bias ``std(r / sqrt(Σ_ii))``, R squared of ``r`` on ``B f``,
    and the mean specific share ``D_i / Σ_ii``. Bars where the asset has no
    finite return are left out."""

    def __init__(self) -> None:
        self._sums: dict[Ticker, FloatArray] = {}

    def observe(
        self, snapshot: RiskModelSnapshot, asset_returns: Mapping[Ticker, float], f: FloatArray
    ) -> None:
        total = (snapshot.B @ snapshot.F * snapshot.B).sum(axis=1) + snapshot.D
        explained = snapshot.B @ f
        for i, ticker in enumerate(snapshot.tickers):
            r = asset_returns.get(ticker, math.nan)
            if not math.isfinite(r) or total[i] <= 0:
                continue
            z = r / math.sqrt(total[i])
            eps = r - float(explained[i])
            share = float(snapshot.D[i] / total[i])
            sums = self._sums.setdefault(ticker, np.zeros(8))
            sums += (1.0, z, z * z, r, r * r, eps, eps * eps, share)

    def result(self, assets: Sequence[Ticker]) -> pd.DataFrame:
        rows = {ticker: self._row(self._sums.get(ticker)) for ticker in assets}
        return pd.DataFrame.from_dict(
            rows, orient="index", columns=[BIAS, R_SQUARED, SPECIFIC_SHARE]
        ).reindex(list(assets))

    @staticmethod
    def _row(sums: FloatArray | None) -> tuple[float, float, float]:
        if sums is None or sums[_COUNT] < 2:
            share = math.nan if sums is None else float(sums[_SHARE] / sums[_COUNT])
            return math.nan, math.nan, share
        n = float(sums[_COUNT])
        share = float(sums[_SHARE]) / n
        var_r = _variance(sums[_R], sums[_R2], n)
        if var_r <= 0:
            return math.nan, math.nan, share
        bias = math.sqrt(max(_variance(sums[_Z], sums[_Z2], n), 0.0))
        r_squared = 1.0 - _variance(sums[_EPS], sums[_EPS2], n) / var_r
        return bias, r_squared, share


@dataclass(frozen=True)
class AssetRisk:
    """One leg's per-asset view, averaged over its attribution bars.
    ``predicted_*`` are annualized contributions to predicted vol (vol points);
    ``holding_returns`` holds ``w_i r_i`` per bar for the realized side."""

    exposure_contributions: pd.DataFrame
    predicted_factor: pd.Series[float]
    predicted_specific: pd.Series[float]
    holding_returns: pd.DataFrame

    @classmethod
    def empty(cls) -> AssetRisk:
        return cls(pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame())

    def is_empty(self) -> bool:
        return self.holding_returns.empty

    def realized(self) -> pd.Series[float]:
        """Euler split of the realized vol of ``R = sum_i w_i r_i``:
        ``cov(w_i r_i, R) / std(R)``, annualized. Sums to ``std(R)``."""
        holdings = self.holding_returns
        total = holdings.sum(axis=1)
        sd = float(total.std()) if len(total) > 1 else 0.0
        if not sd > 0:
            return pd.Series(0.0, index=holdings.columns)
        covariance = holdings.apply(lambda c: c.cov(total))
        return pd.Series(covariance / sd * ANNUAL, index=holdings.columns, dtype=float)


class AssetRiskAccumulator:
    """Per bar: ``w_i B_ik`` per factor, the factor and specific parts of
    ``w_i (Σw)_i / σ``, and ``w_i r_i``. A bar with ``σ = 0`` adds zeros but
    still counts, so the parts average to the mean predicted vol."""

    def __init__(self) -> None:
        self._tickers: tuple[Ticker, ...] = ()
        self._exposure: dict[str, FloatArray] = {}
        self._factor = np.zeros(0)
        self._specific = np.zeros(0)
        self._timestamps: list[datetime] = []
        self._holdings: list[FloatArray] = []

    def observe(
        self, timestamp: datetime, snapshot: RiskModelSnapshot, w: FloatArray, r: FloatArray
    ) -> None:
        if not self._timestamps:
            self._tickers = snapshot.tickers
            self._factor = np.zeros(len(self._tickers))
            self._specific = np.zeros(len(self._tickers))
        assert snapshot.tickers == self._tickers, "the model universe must not change"
        for k, name in enumerate(snapshot.factors):
            column = self._exposure.setdefault(name, np.zeros(len(self._tickers)))
            column += w * snapshot.B[:, k]
        common = snapshot.B @ (snapshot.F @ (snapshot.B.T @ w))
        specific = snapshot.D * w * w
        sigma = math.sqrt(max(float(w @ common + specific.sum()), 0.0))
        if sigma > 0:
            self._factor += w * common / sigma
            self._specific += specific / sigma
        self._timestamps.append(timestamp)
        self._holdings.append(w * r)

    def result(self, assets: Sequence[Ticker]) -> AssetRisk:
        if not self._timestamps:
            return AssetRisk.empty()
        n = len(self._timestamps)
        rows = list(assets)
        index = list(self._tickers)
        exposure = pd.DataFrame(
            {name: column / n for name, column in self._exposure.items()}, index=index
        )
        return AssetRisk(
            exposure_contributions=exposure.reindex(rows, fill_value=0.0),
            predicted_factor=pd.Series(self._factor / n * ANNUAL, index=index).reindex(
                rows, fill_value=0.0
            ),
            predicted_specific=pd.Series(self._specific / n * ANNUAL, index=index).reindex(
                rows, fill_value=0.0
            ),
            holding_returns=pd.DataFrame(
                np.vstack(self._holdings), index=pd.DatetimeIndex(self._timestamps), columns=index
            ).reindex(columns=rows, fill_value=0.0),
        )
