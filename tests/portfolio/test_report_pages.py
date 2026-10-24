from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from trading_strategies.portfolio.optimized_portfolio import SolveLogEntry
from trading_strategies.portfolio.report_pages import optimizer_pages, rolling_bias


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


def _returns(n: int = 300, scale: float = 1.0) -> pd.Series[float]:
    """Realized daily returns whose std is `scale` times the logged ex-ante vol."""
    start = datetime(2024, 1, 1)
    index = pd.DatetimeIndex([start + timedelta(days=i) for i in range(n)])
    rng = np.random.default_rng(1)
    daily = 0.15 / np.sqrt(TRADING_DAYS_PER_YEAR)
    return pd.Series(rng.normal(0, daily * scale, n), index=index, dtype=float)


def test_rolling_bias_is_near_one_when_the_forecast_is_right() -> None:
    entries = _entries(300)

    series = rolling_bias(entries, _returns(300, scale=1.0), window=100)

    assert series.dropna().median() == pytest.approx(1.0, abs=0.2)


def test_rolling_bias_rises_when_realized_risk_exceeds_the_forecast() -> None:
    entries = _entries(300)

    tight = rolling_bias(entries, _returns(300, scale=1.0), window=100).dropna().median()
    loose = rolling_bias(entries, _returns(300, scale=1.5), window=100).dropna().median()

    assert loose > tight * 1.3


def test_the_page_gains_a_bias_panel_when_returns_are_supplied() -> None:
    pdf = _CountingPdf()

    for page in optimizer_pages(
        _entries(), target_vol=0.15, max_gross=2.0, realized_returns=_returns()
    ):
        page.render(cast(PdfPages, pdf))

    assert pdf.pages == 1


def test_the_page_still_renders_without_returns() -> None:
    pdf = _CountingPdf()

    for page in optimizer_pages(_entries(), target_vol=0.15, max_gross=2.0):
        page.render(cast(PdfPages, pdf))

    assert pdf.pages == 1
