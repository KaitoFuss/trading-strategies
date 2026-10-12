"""The optimizer's own page in the backtest report: did the solver hit the
limits it was given, and what did that cost in turnover."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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

from trading_strategies.portfolio.optimized import SolveLogEntry


def _style(ax: Axes, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=6)


def _frame(entries: Sequence[SolveLogEntry]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ex_ante_vol": [e.ex_ante_vol for e in entries],
            "gross": [e.gross for e in entries],
            "turnover": [e.turnover for e in entries],
            "binding": [e.binding for e in entries],
            "status": [e.status for e in entries],
        },
        index=pd.DatetimeIndex([e.timestamp for e in entries]),
    )


@dataclass(frozen=True)
class OptimizerDiagnosticsPage:
    entries: Sequence[SolveLogEntry]
    target_vol: float
    max_gross: float

    def render(self, pdf: PdfPages) -> None:
        if not self.entries:
            return
        frame = _frame(self.entries)
        fig = new_page("Optimizer Diagnostics")
        grid = fig.add_gridspec(
            2, 2, left=0.06, right=0.975, top=0.86, bottom=0.08, hspace=0.45, wspace=0.18
        )

        ax_vol = fig.add_subplot(grid[0, 0])
        ax_vol.plot(frame.index, frame["ex_ante_vol"], color=INK, linewidth=1.0)
        ax_vol.axhline(self.target_vol, color=NEGATIVE, linewidth=1.0, linestyle="--")
        ax_vol.yaxis.set_major_formatter(PercentFormatter(1.0))
        _style(ax_vol, "Ex-ante vol vs target (annualized)")

        ax_gross = fig.add_subplot(grid[0, 1])
        ax_gross.plot(frame.index, frame["gross"], color=INK, linewidth=1.0)
        ax_gross.axhline(self.max_gross, color=NEGATIVE, linewidth=1.0, linestyle="--")
        _style(ax_gross, "Gross leverage vs cap")

        ax_turnover = fig.add_subplot(grid[1, 0])
        ax_turnover.plot(frame.index, frame["turnover"], color=INK, linewidth=0.8)
        ax_turnover.yaxis.set_major_formatter(PercentFormatter(1.0))
        _style(ax_turnover, "One-way turnover per bar")

        shares = frame["binding"].value_counts(normalize=True)
        rows = [[f"Binding: {name}", f"{share:.1%}"] for name, share in shares.items()]
        rows += [
            ["Solver failures", f"{int((frame['status'] != 'optimal').sum()):,}"],
            ["Mean ex-ante vol", f"{frame['ex_ante_vol'].mean():.2%}"],
            ["Max gross", f"{frame['gross'].max():.2f}x"],
            [
                "Annual turnover",
                f"{frame['turnover'].mean() * TRADING_DAYS_PER_YEAR:.1f}x",
            ],
        ]
        style_table(fig.add_subplot(grid[1, 1]), rows, ["Metric", "Value"], cellLoc="center")
        save_page(pdf, fig)


def optimizer_pages(
    entries: Sequence[SolveLogEntry], *, target_vol: float, max_gross: float
) -> list[ReportPage]:
    if not entries:
        return []
    return [OptimizerDiagnosticsPage(entries, target_vol, max_gross)]
