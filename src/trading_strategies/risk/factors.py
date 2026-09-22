"""Macro factor definitions (Two Sigma Factor Lens style) and their
hierarchical residualization. A factor is a fixed portfolio of universe ETFs;
a factor at level n is residualized against every active factor at an earlier
level, so it stays a portfolio of ETFs and carries only the risk the levels
above it do not explain."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from backtester.core.events import Ticker

from trading_strategies.risk.covariance import FloatArray

FactorGroup = Literal["core", "secondary", "granular"]
_GROUPS: frozenset[str] = frozenset({"core", "secondary", "granular"})


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    level: int
    group: FactorGroup
    weights: Mapping[Ticker, float]


def load_factors(path: Path, *, granular: bool) -> list[FactorDefinition]:
    entries = json.loads(path.read_text())["factors"]
    factors: list[FactorDefinition] = []
    seen: set[str] = set()
    for entry in entries:
        name = str(entry["name"])
        if entry["group"] not in _GROUPS:
            raise ValueError(f"factor {name!r}: unknown group {entry['group']!r}")
        if not entry["weights"]:
            raise ValueError(f"factor {name!r}: empty weights")
        if name in seen:
            raise ValueError(f"duplicate factor name {name!r}")
        seen.add(name)
        factors.append(
            FactorDefinition(
                name=name,
                level=int(entry["level"]),
                group=cast(FactorGroup, entry["group"]),
                weights={str(t): float(w) for t, w in entry["weights"].items()},
            )
        )
    if not granular:
        factors = [factor for factor in factors if factor.group != "granular"]
    return sorted(factors, key=lambda factor: factor.level)


def factor_weight_matrix(
    factors: Sequence[FactorDefinition],
    tickers: Sequence[Ticker],
    available: Collection[Ticker],
) -> FloatArray:
    """N x K raw factor weights over ``tickers``. The long and short legs are
    each renormalized over their available tickers to their original gross, so
    a missing EEM does not shrink the Equity factor. A leg with no available
    ticker makes the factor inactive (a zero column): a spread missing a leg is
    not the factor any more."""
    index = {ticker: i for i, ticker in enumerate(tickers)}
    matrix = np.zeros((len(tickers), len(factors)))
    for k, factor in enumerate(factors):
        column = np.zeros(len(tickers))
        for sign in (1.0, -1.0):
            leg = {t: w for t, w in factor.weights.items() if w * sign > 0}
            if not leg:
                continue
            live = {t: w for t, w in leg.items() if t in available and t in index}
            if not live:
                break
            scale = sum(leg.values()) / sum(live.values())
            for ticker, weight in live.items():
                column[index[ticker]] = weight * scale
        else:
            matrix[:, k] = column
    return matrix


def residualize(weights: FloatArray, levels: Sequence[int], covariance: FloatArray) -> FloatArray:
    """Residualize each factor portfolio, in column order, against the already
    residualized active factors at strictly earlier levels (GLS projection
    under ``covariance``). Same-level factors stay correlated; ``F`` carries
    that. Contract: columns are ordered by non-decreasing level (as
    ``load_factors`` returns them); a column is only residualized against
    columns before it."""
    residual = weights.copy()
    active = np.any(weights != 0, axis=0)
    for k in range(weights.shape[1]):
        if not active[k]:
            continue
        earlier = [j for j in range(k) if active[j] and levels[j] < levels[k]]
        if not earlier:
            continue
        basis = residual[:, earlier]
        beta = np.linalg.lstsq(
            basis.T @ covariance @ basis, basis.T @ covariance @ weights[:, k], rcond=None
        )[0]
        residual[:, k] = weights[:, k] - basis @ beta
    return residual
