from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from trading_strategies.risk import report_pages
from trading_strategies.risk.attribution import RESIDUAL, AttributionResult
from trading_strategies.risk.model import IDIOSYNCRATIC
from trading_strategies.risk.report_pages import (
    FactorDefinitionsPage,
    FactorRegressionPage,
    attribution_pages,
)


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
        risk_decomposition=pd.DataFrame(
            {"Equity": 0.5, "Momentum": 0.3, IDIOSYNCRATIC: 0.2}, index=dates
        ),
        contributions=pd.DataFrame(
            {"Equity": returns * 0.5, "Momentum": returns * 0.3, RESIDUAL: returns * 0.2},
            index=dates,
        ),
        portfolio_returns=returns,
        predicted_vol=pd.Series(0.01, index=dates),
        factor_returns=pd.DataFrame({"Equity": returns, "Momentum": returns * 0.3}, index=dates),
        simple_factor_returns=pd.DataFrame(
            {"Equity": returns, "Momentum": returns * 0.3}, index=dates
        ),
        specific_returns=returns * 0.15,
    )


def _empty() -> AttributionResult:
    dates = pd.DatetimeIndex([])
    return AttributionResult(
        exposures=pd.DataFrame(index=dates),
        risk_decomposition=pd.DataFrame(index=dates),
        contributions=pd.DataFrame(index=dates),
        portfolio_returns=pd.Series(dtype=float, index=dates),
        predicted_vol=pd.Series(dtype=float, index=dates),
        factor_returns=pd.DataFrame(index=dates),
        simple_factor_returns=pd.DataFrame(index=dates),
        specific_returns=pd.Series(dtype=float, index=dates),
    )


def test_seven_pages_render_for_two_legs() -> None:
    pdf = _CountingPdf()

    for page in attribution_pages({"Strategy": _result(), "Buy & Hold": _result()}):
        page.render(cast(PdfPages, pdf))

    assert pdf.pages == 7


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


def test_factor_regression_page_renders_one_page() -> None:
    pdf = _CountingPdf()

    FactorRegressionPage({"Strategy": _result(), "Empty": _empty()}).render(cast(PdfPages, pdf))

    assert pdf.pages == 1


def _factors_file(tmp_path: Path) -> Path:
    path = tmp_path / "factors.json"
    path.write_text(
        json.dumps(
            {
                "factors": [
                    {"name": "Tech", "level": 4, "granular": True, "weights": {"QQQ": 1.0}},
                    {
                        "name": "Equity",
                        "level": 1,
                        "granular": False,
                        "weights": {"SPY": 0.6, "EFA": -0.4},
                    },
                ]
            }
        )
    )
    return path


@pytest.mark.parametrize(("granular", "tech_active"), [(True, "yes"), (False, "no")])
def test_factor_definitions_page_lists_every_factor_and_marks_inactive_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, granular: bool, tech_active: str
) -> None:
    captured: list[Sequence[Sequence[str]]] = []
    monkeypatch.setattr(
        report_pages, "save_table_pages", lambda _pdf, _title, rows, *_, **__: captured.append(rows)
    )

    FactorDefinitionsPage(_factors_file(tmp_path), granular=granular).render(
        cast(PdfPages, _CountingPdf())
    )

    assert captured == [
        [
            ["Equity", "1", "no", "yes", "SPY +0.60, EFA -0.40"],
            ["Tech", "4", "yes", tech_active, "QQQ +1.00"],
        ]
    ]


def test_factor_definitions_page_renders_the_repo_factors_on_one_page() -> None:
    pdf = _CountingPdf()

    FactorDefinitionsPage(Path("configs/factors.json"), granular=True).render(cast(PdfPages, pdf))

    assert pdf.pages == 1
