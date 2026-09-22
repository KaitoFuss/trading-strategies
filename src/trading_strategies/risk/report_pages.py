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
from matplotlib.ticker import PercentFormatter

from trading_strategies.risk.attribution import AttributionResult
from trading_strategies.risk.model import SPECIFIC

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


def _render_legs(
    pdf: PdfPages, title: str, results: Mapping[str, AttributionResult], draw: _Draw
) -> None:
    legs = {label: r for label, r in results.items() if not r.exposures.empty}
    if not legs:
        return
    fig = new_page(title)
    grid = fig.add_gridspec(
        len(legs),
        2,
        width_ratios=[1.6, 1],
        left=0.06,
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
    rows = [
        [str(name), f"{col.mean():+.2f}", f"{col.min():+.2f}", f"{col.max():+.2f}"]
        for name, col in exposures.items()
    ]
    style_table(right, rows, ["Factor", "Mean", "Min", "Max"], cellLoc="center")


def _draw_risk(left: Axes, right: Axes, label: str, result: AttributionResult) -> None:
    shares = result.risk_shares.mean()
    names = [str(n) for n in shares.index]
    left.barh(names, shares.to_numpy(), color=[MUTED if n == SPECIFIC else INK for n in names])
    left.invert_yaxis()
    left.xaxis.set_major_formatter(PercentFormatter(1.0))
    _style(left, f"{label} - mean share of ex-ante variance")

    ex_ante = result.predicted_vol * _ANNUAL
    realized = result.portfolio_returns.rolling(60, min_periods=20).std() * _ANNUAL
    right.plot(ex_ante.index, ex_ante.to_numpy(), color=INK, linewidth=1.0, label="Ex-ante")
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
    rows = [[str(name), f"{value:+.2%}"] for name, value in annual.items()]
    style_table(right, rows, ["Source", "Annualized"], cellLoc="center")


def _draw_health(left: Axes, right: Axes, label: str, result: AttributionResult) -> None:
    rolling = result.rolling_bias()
    left.axhspan(*_BIAS_BAND, color=GRID, alpha=0.6)
    left.axhline(1.0, color=MUTED, linewidth=0.8)
    left.plot(rolling.index, rolling.to_numpy(), color=INK, linewidth=1.0)
    _style(left, f"{label} - rolling bias statistic (target 1.0)")

    rows = [
        ["Bias statistic", f"{result.bias_statistic():.2f}"],
        ["Bars", f"{len(result.portfolio_returns):,}"],
        ["Mean ex-ante vol", f"{result.predicted_vol.mean() * _ANNUAL:.2%}"],
        ["Realized vol", f"{result.portfolio_returns.std() * _ANNUAL:.2%}"],
    ]
    style_table(right, rows, ["Metric", "Value"], cellLoc="center")


@dataclass(frozen=True)
class FactorExposurePage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        _render_legs(pdf, "Factor Exposures", self.results, _draw_exposures)


@dataclass(frozen=True)
class RiskDecompositionPage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        _render_legs(pdf, "Risk Decomposition", self.results, _draw_risk)


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
