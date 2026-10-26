"""Per-asset report pages: model fit (one table for the model), and per leg
the weight x beta heatmap and the predicted vs realized vol contributions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from backtester.tracker.report import (
    INK,
    MUTED,
    NEGATIVE,
    SURFACE,
    ReportPage,
    new_page,
    save_page,
    save_table_pages,
)
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from trading_strategies.risk.asset_analytics import BIAS, R_SQUARED, SPECIFIC_SHARE
from trading_strategies.risk.attribution import AttributionResult

_POSITIVE = "#2f6fb0"
_HEATMAP = LinearSegmentedColormap.from_list("exposure", [NEGATIVE, SURFACE, _POSITIVE])
_FIT_SUBTITLE = (
    "Bias: std(return / predicted vol), 0.8-1.2 is good. "
    "R squared: variance explained by factors. "
    "Specific share: mean D / total variance. Same for all legs."
)
_EXPOSURE_SUBTITLE = (
    "Time-average of weight x beta. "
    "Each column sums to the portfolio's mean exposure to that factor."
)


def _subtitle(fig: Figure, text: str) -> None:
    fig.text(0.04, 0.905, text, color=MUTED, fontsize=9)


def _plain_axes(fig: Figure, rect: tuple[float, float, float, float]) -> Axes:
    ax = fig.add_axes(rect)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7)
    return ax


def _number(value: float, spec: str) -> str:
    return "n/a" if math.isnan(value) else format(value, spec)


@dataclass(frozen=True)
class AssetFitPage:
    results: Mapping[str, AttributionResult]

    def render(self, pdf: PdfPages) -> None:
        fit = next((r.asset_fit for r in self.results.values() if not r.asset_fit.empty), None)
        if fit is None:
            return
        ordered = fit.sort_values(BIAS, ascending=False, na_position="last")
        rows = [
            [
                str(asset),
                _number(row[BIAS], ".2f"),
                _number(row[R_SQUARED], ".2f"),
                _number(row[SPECIFIC_SHARE], ".0%"),
            ]
            for asset, row in ordered.iterrows()
        ]
        save_table_pages(
            pdf,
            "Per-Asset Model Fit",
            rows,
            ["Asset", "Bias", "R squared", "Specific share"],
            col_widths=[0.25, 0.25, 0.25, 0.25],
            wrap_widths=[None, None, None, None],
            subtitle=_FIT_SUBTITLE,
        )


@dataclass(frozen=True)
class ExposureContributionPage:
    label: str
    result: AttributionResult

    def render(self, pdf: PdfPages) -> None:
        frame = self.result.asset_risk.exposure_contributions
        values = frame.to_numpy(dtype=float)
        largest = float(np.abs(values).max()) if values.size else 0.0
        limit = largest if largest > 0 else 1.0
        fig = new_page(f"Exposure Contributions - {self.label}")
        _subtitle(fig, _EXPOSURE_SUBTITLE)
        ax = fig.add_axes((0.08, 0.03, 0.88, 0.76))
        ax.imshow(values, aspect="auto", cmap=_HEATMAP, norm=TwoSlopeNorm(0.0, -limit, limit))
        ax.set_yticks(range(len(frame.index)), [str(a) for a in frame.index], fontsize=5)
        ax.set_xticks(
            range(len(frame.columns)),
            [str(c) for c in frame.columns],
            fontsize=6.5,
            rotation=40,
            ha="left",
        )
        ax.xaxis.tick_top()
        ax.tick_params(colors=INK, length=0)
        for (i, j), value in np.ndenumerate(values):
            text = f"{value:+.2f}"
            if text not in ("+0.00", "-0.00"):
                ax.text(j, i, text, ha="center", va="center", fontsize=3.8, color=INK)
        save_page(pdf, fig)


@dataclass(frozen=True)
class RiskContributionPage:
    label: str
    result: AttributionResult

    def render(self, pdf: PdfPages) -> None:
        risk = self.result.asset_risk
        factor, specific, realized = (
            risk.predicted_factor,
            risk.predicted_specific,
            risk.realized(),
        )
        order = (factor + specific).sort_values(ascending=False).index
        fig = new_page(f"Per-Asset Risk Contribution - {self.label}")
        _subtitle(
            fig,
            "Contribution to annualized portfolio vol, in vol points. "
            f"Bars sum to predicted vol {(factor.sum() + specific.sum()):.1%}. "
            f"Dots sum to realized vol {realized.sum():.1%}.",
        )
        ax = _plain_axes(fig, (0.08, 0.07, 0.88, 0.80))
        y = np.arange(len(order))
        f, s = factor[order].to_numpy(), specific[order].to_numpy()
        ax.barh(y, f, color=INK, label="Predicted: factor")
        # stack each part away from zero on its own side, so a hedge reads left of zero
        base = np.where(np.sign(s) == np.sign(f), f, 0.0)
        ax.barh(y, s, left=base, color=MUTED, label="Predicted: specific")
        ax.scatter(realized[order].to_numpy(), y, color=NEGATIVE, s=14, zorder=3, label="Realized")
        ax.axvline(0.0, color=MUTED, linewidth=0.8)
        ax.set_yticks(y, [str(a) for a in order], fontsize=5)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        ax.set_xlabel("Contribution to annualized vol (vol points)", color=INK, fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        save_page(pdf, fig)


def asset_pages(results: Mapping[str, AttributionResult]) -> list[ReportPage]:
    legs = {label: r for label, r in results.items() if not r.asset_risk.is_empty()}
    return [
        AssetFitPage(results),
        *(ExposureContributionPage(label, r) for label, r in legs.items()),
        *(RiskContributionPage(label, r) for label, r in legs.items()),
    ]
