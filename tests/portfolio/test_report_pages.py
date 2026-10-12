from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from trading_strategies.portfolio.optimized import SolveLogEntry
from trading_strategies.portfolio.report_pages import optimizer_pages


class _CountingPdf:
    def __init__(self) -> None:
        self.pages = 0

    def savefig(self, figure: Figure, **_: Any) -> None:
        self.pages += 1
        plt.close(figure)


def _entries(n: int = 300) -> list[SolveLogEntry]:
    start = datetime(2024, 1, 1)
    return [
        SolveLogEntry(
            timestamp=start + timedelta(days=i),
            status="optimal" if i % 50 else "infeasible",
            binding="vol" if i % 3 else "leverage",
            ex_ante_vol=0.15 + 0.01 * (i % 5),
            gross=1.8,
            turnover=0.05,
        )
        for i in range(n)
    ]


def test_one_page_renders_for_a_solve_log() -> None:
    pdf = _CountingPdf()

    for page in optimizer_pages(_entries(), target_vol=0.15, max_gross=2.0):
        page.render(cast(PdfPages, pdf))

    assert pdf.pages == 1


def test_no_page_without_entries() -> None:
    pdf = _CountingPdf()

    pages = optimizer_pages([], target_vol=0.15, max_gross=2.0)
    for page in pages:
        page.render(cast(PdfPages, pdf))

    assert pdf.pages == 0


def test_page_renders_into_a_real_pdf(tmp_path: Any) -> None:
    path = tmp_path / "optimizer.pdf"
    with PdfPages(path) as pdf:
        for page in optimizer_pages(_entries(), target_vol=0.15, max_gross=2.0):
            page.render(pdf)

    assert path.stat().st_size > 0
