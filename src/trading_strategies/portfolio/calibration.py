"""Choosing the shrinkage parameter without fitting it to performance.

``shrinkage`` is constant within a backtest, so one run per candidate value
gives, for every bar, the forecast vol and the realized return. The folds are
slices of that output, which is why no nested backtest is needed. Selection
uses the bias statistic only: its correct value is 1.0 by definition, so it is
not a performance target that can be gamed. Sharpe travels alongside as a
guard -- if the best-calibrated value destroys it, that is a finding about
where the edge came from, not a reason to pick another value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd

# (calibration ends before, evaluation ends before); calibration starts at bar 1.
FOLDS: tuple[tuple[int, int], ...] = ((2013, 2017), (2017, 2021), (2021, 2024), (2024, 2026))
_BAND = (0.95, 1.05)


def bias_statistic(forecast: pd.Series[float], realized: pd.Series[float]) -> float:
    """``std(realized / forecast)``. 1.0 means the forecast was right."""
    live = forecast > 0
    standardized = (realized[live] / forecast[live]).dropna()
    if len(standardized) < 2:
        return float("nan")
    return float(standardized.std())


@dataclass(frozen=True)
class SweepRow:
    shrinkage: float
    bias: float
    sharpe: float
    realized_vol: float
    turnover: float
    gross: float


def select_shrinkage(rows: Sequence[SweepRow], band: tuple[float, float] = _BAND) -> float:
    """The smallest shrinkage whose bias sits inside ``band``; failing that, the
    one whose bias is closest to 1.0. Smallest, because shrinkage is a
    distortion of the model's own estimate: take as little as the calibration
    allows."""
    inside = [row for row in rows if band[0] <= row.bias <= band[1]]
    if inside:
        return min(inside, key=lambda row: row.shrinkage).shrinkage
    return min(rows, key=lambda row: abs(row.bias - 1.0)).shrinkage


def fold_table(
    per_delta: Mapping[float, pd.DataFrame], folds: tuple[tuple[int, int], ...] = FOLDS
) -> pd.DataFrame:
    """Expanding-window walk-forward over the sweep's per-bar output.

    Each frame in ``per_delta`` holds ``forecast`` and ``realized`` on a
    DatetimeIndex, with ``realized`` already aligned so that index ``t`` holds
    the return the forecast at ``t`` is responsible for. For each fold the selection sees only bars before the
    calibration boundary, and the reported evaluation bias covers the bars
    between the two boundaries.
    """
    records: dict[str, dict[str, float]] = {}
    for calibration_end, evaluation_end in folds:
        rows: list[SweepRow] = []
        for shrinkage, frame in sorted(per_delta.items()):
            years = cast("pd.DatetimeIndex", frame.index).year
            calibration = frame[years < calibration_end]
            rows.append(
                SweepRow(
                    shrinkage=shrinkage,
                    bias=bias_statistic(calibration["forecast"], calibration["realized"]),
                    sharpe=float("nan"),
                    realized_vol=float("nan"),
                    turnover=float("nan"),
                    gross=float("nan"),
                )
            )
        selected = select_shrinkage(rows)
        frame = per_delta[selected]
        years = cast("pd.DatetimeIndex", frame.index).year
        evaluation = frame[(years >= calibration_end) & (years < evaluation_end)]
        label = f"{calibration_end}-{evaluation_end - 1}"
        records[label] = {
            "selected": selected,
            "calibration_bias": next(r.bias for r in rows if r.shrinkage == selected),
            "evaluation_bias": bias_statistic(evaluation["forecast"], evaluation["realized"]),
        }
    return pd.DataFrame(records).T
