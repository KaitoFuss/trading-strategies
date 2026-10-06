"""Factor-attribution pages for ``backtester``'s PDF report, via its
``ReportPage`` / ``extra_pages`` API. One row per leg; legs without
attribution rows are skipped, and a page with no legs is not written."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR
from backtester.tracker.report import (
    GRID,
    INK,
    MUTED,
    NEGATIVE,
    SURFACE,
    ReportPage,
    new_page,
    save_page,
    style_table,
)
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.textpath import TextPath
from matplotlib.ticker import PercentFormatter

from trading_strategies.risk.attribution import AttributionResult
from trading_strategies.risk.model import IDIOSYNCRATIC

_ANNUAL = math.sqrt(TRADING_DAYS_PER_YEAR)
_BIAS_BAND = (0.8, 1.2)

_Draw = Callable[[Axes, Axes, str, AttributionResult], None]


def _style(ax: Axes, title: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=6)


def _colors(names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return {name: cmap(i % 20) for i, name in enumerate(names)}


def _lines(ax: Axes, frame: pd.DataFrame, title: str) -> None:
    names = [str(c) for c in frame.columns]
    colors = _colors(names)
    for name in names:
        series = frame[name]
        ax.plot(series.index, series.to_numpy(), color=colors[name], linewidth=1.0, label=name)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    if names:
        ax.legend(fontsize=6.5, ncol=3, frameon=False, loc="upper left")
    _style(ax, title)


def _col_widths(rows: list[list[str]], headers: list[str]) -> list[float]:
    """Column widths proportional to their longest cell, so a long name
    (e.g. "Cyclical vs Defensive") doesn't get squished into the same width
    as a short one ("Equity")."""
    widths = [
        max(len(header), *(len(row[col]) for row in rows)) if rows else len(header)
        for col, header in enumerate(headers)
    ]
    total = sum(widths)
    return [w / total for w in widths]


def _left_margin(
    names: set[str], fig_width: float, *, pad: float = 0.12, max_margin: float = 0.22
) -> float:
    """Gridspec left margin, wide enough for the longest of ``names`` rendered
    as an 8pt y-axis tick label (matching ``_style``'s ``labelsize=8``) plus a
    little padding, instead of a fixed margin that clips a long factor name
    like "Foreign Currency". ``TextPath`` gives the label's rendered width
    without needing a draw/renderer -- checked against actual rendered tick
    labels (barh, labelsize=8): within ~3% for "Equity"/"Foreign
    Currency"/"Cyclical vs Defensive"."""
    if not names:
        return 0.06
    longest = max(names, key=len)
    width_inches = TextPath((0, 0), longest, size=8).get_extents().width / 72
    return min(max_margin, (width_inches + pad) / fig_width)


def _render_legs(
    pdf: PdfPages,
    title: str,
    results: Mapping[str, AttributionResult],
    draw: _Draw,
    *,
    left_labels: set[str] | None = None,
) -> None:
    legs = {label: r for label, r in results.items() if not r.exposures.empty}
    if not legs:
        return
    fig = new_page(title)
    left = _left_margin(left_labels, fig.get_size_inches()[0]) if left_labels else 0.06
    grid = fig.add_gridspec(
        len(legs),
        2,
        width_ratios=[1.6, 1],
        left=left,
        right=0.975,
        top=0.86,
        bottom=0.07,
        hspace=0.45,
        wspace=0.18,
    )
    for row, (label, result) in enumerate(legs.items()):
        draw(fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1]), label, result)
    save_page(pdf, fig)


def _draw_exposures(left: Axes, right: Axes, label: str, result: AttributionResult) -> None:
    exposures = result.exposures
    _lines(left, exposures, label)
    headers = ["Factor", "Mean", "Min", "Max"]
    rows = [
        [str(name), f"{col.mean():+.2f}", f"{col.min():+.2f}", f"{col.max():+.2f}"]
        for name, col in exposures.items()
    ]
    style_table(right, rows, headers, cellLoc="center", colWidths=_col_widths(rows, headers))


def _draw_risk(left: Axes, right: Axes, label: str, result: AttributionResult) -> None:
    decomposition = result.risk_decomposition.mean()
    names = [str(n) for n in decomposition.index]
    left.barh(
        names, decomposition.to_numpy(), color=[MUTED if n == IDIOSYNCRATIC else INK for n in names]
    )
    left.invert_yaxis()
    left.xaxis.set_major_formatter(PercentFormatter(1.0))
    _style(left, f"{label} - mean share of predicted variance")

    predicted = result.predicted_vol * _ANNUAL
    realized = result.portfolio_returns.rolling(60, min_periods=20).std() * _ANNUAL
    right.plot(predicted.index, predicted.to_numpy(), color=INK, linewidth=1.0, label="Predicted")
    right.plot(
        realized.index, realized.to_numpy(), color=NEGATIVE, linewidth=1.0, label="Realized 60d"
    )
    right.yaxis.set_major_formatter(PercentFormatter(1.0))
    right.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))  # type: ignore[no-untyped-call]
    right.legend(fontsize=7, frameon=False, loc="upper left")
    _style(right, "Annualized vol")


def _draw_returns(left: Axes, right: Axes, label: str, result: AttributionResult) -> None:
    _lines(left, result.contributions.cumsum(), f"{label} - cumulative contribution")
    left.yaxis.set_major_formatter(PercentFormatter(1.0))
    annual = result.contributions.mean() * TRADING_DAYS_PER_YEAR
    headers = ["Source", "Annualized"]
    rows = [[str(name), f"{value:+.2%}"] for name, value in annual.items()]
    style_table(right, rows, headers, cellLoc="center", colWidths=_col_widths(rows, headers))


def _annualized_vol(returns: pd.Series[float]) -> float:
    """Same convention as ``PerformanceTracker.metrics``'s ``annualized_vol``:
    annualize off elapsed calendar time (the observed return frequency), not
    a fixed trading-days-per-year constant, so the two don't disagree just
    because this series has gaps or a shorter span."""
    if len(returns) < 2:
        return 0.0
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    periods_per_year = len(returns) / years
    return float(returns.std() * periods_per_year**0.5)


def _draw_health(left: Axes, right: Axes, label: str, result: AttributionResult) -> None:
    rolling = result.rolling_bias()
    left.axhspan(*_BIAS_BAND, color=GRID, alpha=0.6)
    left.axhline(1.0, color=MUTED, linewidth=0.8)
    left.plot(rolling.index, rolling.to_numpy(), color=INK, linewidth=1.0)
    _style(left, f"{label} - rolling bias statistic (target 1.0)")

    rows = [
        ["Bias statistic", f"{result.bias_statistic():.2f}"],
        ["Bars", f"{len(result.portfolio_returns):,}"],
        ["Mean predicted vol", f"{result.predicted_vol.mean() * _ANNUAL:.2%}"],
        ["Realized vol", f"{_annualized_vol(result.portfolio_returns):.2%}"],
    ]
    style_table(right, rows, ["Metric", "Value"], cellLoc="center")


@dataclass(frozen=True)
class FactorExposurePage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        _render_legs(pdf, "Factor Exposures in Betas", self.results, _draw_exposures)


@dataclass(frozen=True)
class RiskDecompositionPage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        names = {str(n) for r in self.results.values() for n in r.risk_decomposition.columns}
        _render_legs(pdf, "Risk Decomposition", self.results, _draw_risk, left_labels=names)


@dataclass(frozen=True)
class ReturnAttributionPage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        _render_legs(pdf, "Return Attribution", self.results, _draw_returns)


@dataclass(frozen=True)
class ModelHealthPage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        _render_legs(pdf, "Factor Model Health", self.results, _draw_health)


def attribution_pages(results: Mapping[str, AttributionResult]) -> list[ReportPage]:
    return [
        FactorExposurePage(results),
        RiskDecompositionPage(results),
        ReturnAttributionPage(results),
        ModelHealthPage(results),
    ]
