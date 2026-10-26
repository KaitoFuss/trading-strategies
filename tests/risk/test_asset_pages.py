from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from trading_strategies.risk import asset_pages
from trading_strategies.risk.asset_analytics import (
    BIAS,
    R_SQUARED,
    SPECIFIC_SHARE,
    AssetRisk,
)
from trading_strategies.risk.asset_pages import asset_pages as build_pages
from trading_strategies.risk.attribution import AttributionResult


class _RecordingPdf:
    def __init__(self) -> None:
        self.texts: list[list[str]] = []

    @property
    def pages(self) -> int:
        return len(self.texts)

    def savefig(self, figure: Figure, **_: Any) -> None:
        texts = [t.get_text() for t in figure.texts]
        texts += [t.get_text() for ax in figure.axes for t in (ax.title, ax.xaxis.label)]
        self.texts.append(texts)
        plt.close(figure)


_ASSETS = ["X", "Y", "Z"]


def _result(n: int = 100) -> AttributionResult:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    empty = pd.DataFrame(index=dates)
    series = pd.Series(0.0, index=dates)
    return AttributionResult(
        exposures=pd.DataFrame({"Equity": 0.5, "Rates": -0.2}, index=dates),
        risk_decomposition=empty,
        contributions=empty,
        portfolio_returns=series,
        predicted_vol=series + 0.01,
        factor_returns=empty,
        simple_factor_returns=empty,
        specific_returns=series,
        asset_fit=pd.DataFrame(
            {BIAS: [1.1, math.nan, 0.9], R_SQUARED: [0.5, math.nan, 0.7], SPECIFIC_SHARE: 0.3},
            index=_ASSETS,
        ),
        asset_risk=AssetRisk(
            exposure_contributions=pd.DataFrame(
                {"Equity": [0.3, 0.2, 0.0], "Rates": [-0.1, 0.0, -0.1]}, index=_ASSETS
            ),
            predicted_factor=pd.Series([0.06, -0.01, 0.0], index=_ASSETS),
            predicted_specific=pd.Series([0.02, 0.01, 0.0], index=_ASSETS),
            holding_returns=pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=dates, columns=_ASSETS),
        ),
    )


def _empty() -> AttributionResult:
    dates = pd.DatetimeIndex([])
    frame = pd.DataFrame(index=dates)
    series = pd.Series(dtype=float, index=dates)
    return AttributionResult(frame, frame, frame, series, series, frame, frame, series)


def _render(results: dict[str, AttributionResult]) -> _RecordingPdf:
    pdf = _RecordingPdf()
    for page in build_pages(results):
        page.render(cast(PdfPages, pdf))
    return pdf


def test_pages_for_two_legs() -> None:
    pdf = _render({"Strategy": _result(), "Buy & Hold": _result()})

    # 1 model-fit table page + 2 exposure pages + 2 risk pages
    assert pdf.pages == 5


def test_empty_legs_render_nothing() -> None:
    assert _render({"Strategy": _empty(), "Buy & Hold": _empty()}).pages == 0


def test_an_empty_leg_is_skipped() -> None:
    assert _render({"Strategy": _result(), "Buy & Hold": _empty()}).pages == 3


def test_fit_rows_sort_nan_last_and_show_na(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Sequence[Sequence[str]]] = []

    def fake_save(
        _pdf: object, _title: str, rows: Sequence[Sequence[str]], *_: Any, **__: Any
    ) -> None:
        captured.append(rows)

    monkeypatch.setattr(asset_pages, "save_table_pages", fake_save)
    _render({"Strategy": _result()})

    rows = captured[0]
    assert [row[0] for row in rows] == ["X", "Z", "Y"]
    assert rows[0][1:] == ["1.10", "0.50", "30%"]
    assert rows[-1][1] == "n/a"
    assert rows[-1][2] == "n/a"


def test_risk_page_states_vol_units() -> None:
    pdf = _render({"Strategy": _result()})

    text = " ".join(pdf.texts[-1])
    assert "Per-Asset Risk Contribution - Strategy" in text
    assert "in vol points" in text
    assert "Bars sum to predicted vol 8.0%" in text
    assert "Dots sum to realized vol" in text
    assert "Contribution to annualized vol (vol points)" in text


def test_exposure_page_explains_the_column_sums() -> None:
    pdf = _render({"Strategy": _result()})

    text = " ".join(pdf.texts[1])
    assert "Exposure Contributions - Strategy" in text
    assert "Each column sums to the portfolio's mean exposure" in text
