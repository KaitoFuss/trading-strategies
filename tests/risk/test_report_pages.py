from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from trading_strategies.risk.attribution import RESIDUAL, AttributionResult
from trading_strategies.risk.model import SPECIFIC
from trading_strategies.risk.report_pages import attribution_pages


class _CountingPdf:
    def __init__(self) -> None:
        self.pages = 0

    def savefig(self, figure: Figure, **_: Any) -> None:
        self.pages += 1
        plt.close(figure)


def _result(n: int = 300) -> AttributionResult:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, n), index=dates)
    return AttributionResult(
        exposures=pd.DataFrame(
            {"Equity": rng.normal(0.5, 0.1, n), "Momentum": rng.normal(1.0, 0.2, n)}, index=dates
        ),
        risk_shares=pd.DataFrame({"Equity": 0.5, "Momentum": 0.3, SPECIFIC: 0.2}, index=dates),
        contributions=pd.DataFrame(
            {"Equity": returns * 0.5, "Momentum": returns * 0.3, RESIDUAL: returns * 0.2},
            index=dates,
        ),
        portfolio_returns=returns,
        predicted_vol=pd.Series(0.01, index=dates),
    )


def _empty() -> AttributionResult:
    dates = pd.DatetimeIndex([])
    return AttributionResult(
        exposures=pd.DataFrame(index=dates),
        risk_shares=pd.DataFrame(index=dates),
        contributions=pd.DataFrame(index=dates),
        portfolio_returns=pd.Series(dtype=float, index=dates),
        predicted_vol=pd.Series(dtype=float, index=dates),
    )


def test_four_pages_render_for_two_legs() -> None:
    pdf = _CountingPdf()

    for page in attribution_pages({"Strategy": _result(), "Buy & Hold": _result()}):
        page.render(cast(PdfPages, pdf))

    assert pdf.pages == 4


def test_empty_legs_are_skipped_and_all_empty_writes_nothing() -> None:
    pdf = _CountingPdf()

    for page in attribution_pages({"Strategy": _empty()}):
        page.render(cast(PdfPages, pdf))

    assert pdf.pages == 0


def test_pages_render_into_a_real_pdf(tmp_path: Any) -> None:
    path = tmp_path / "pages.pdf"
    with PdfPages(path) as pdf:
        for page in attribution_pages({"Strategy": _result(), "Empty": _empty()}):
            page.render(pdf)

    assert path.stat().st_size > 0
