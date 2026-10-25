"""Per-asset statistics gathered bar by bar during the attribution replay.

``ModelFitAccumulator`` scores the model on each asset and needs no weights.
Running sums keep the memory flat over long backtests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from backtester.core.events import Ticker

from trading_strategies.risk.model import RiskModelSnapshot
from trading_strategies.utils.streaming import FloatArray

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
