from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from backtester.core.events import Bar, MarketEvent

from tests.risk.helpers import market_factor_config
from trading_strategies.risk.attribution import RESIDUAL, AttributionResult, run_attribution
from trading_strategies.risk.model import FactorRiskModel

_START = datetime(2020, 1, 1)
_WEIGHTS = {"MKT": 0.5, "A": 0.5}


def _scenario(
    n_bars: int = 500, seed: int = 0
) -> tuple[
    list[MarketEvent], list[tuple[datetime, dict[str, float]]], list[tuple[datetime, float]]
]:
    rng = np.random.default_rng(seed)
    closes = {"MKT": 100.0, "A": 100.0, "B": 100.0}
    events: list[MarketEvent] = []
    weights: list[tuple[datetime, dict[str, float]]] = []
    equity: list[tuple[datetime, float]] = []
    value = 1_000_000.0
    for i in range(n_bars):
        ts = _START + timedelta(days=i)
        if i > 0:
            market = rng.normal(0, 0.01)
            previous = dict(closes)
            closes["MKT"] *= math.exp(market)
            closes["A"] *= math.exp(0.5 * market + rng.normal(0, 0.005))
            closes["B"] *= math.exp(-market + rng.normal(0, 0.005))
            value *= 1 + sum(w * (closes[t] / previous[t] - 1) for t, w in _WEIGHTS.items())
        events.append(MarketEvent(timestamp=ts, bars={t: Bar(close=c) for t, c in closes.items()}))
        weights.append((ts, {} if i == 0 else dict(_WEIGHTS)))
        equity.append((ts, value))
    return events, weights, equity


def _result(tmp_path: Path) -> AttributionResult:
    events, weights, equity = _scenario()
    model = FactorRiskModel(["MKT", "A", "B"], market_factor_config(tmp_path))
    return run_attribution(events, weights, equity, model)


def test_contributions_add_up_to_the_portfolio_return(tmp_path: Path) -> None:
    result = _result(tmp_path)

    np.testing.assert_allclose(
        result.contributions.sum(axis=1).to_numpy(), result.portfolio_returns.to_numpy()
    )
    assert result.contributions.columns[-1] == RESIDUAL


def test_rows_start_only_after_the_model_is_ready(tmp_path: Path) -> None:
    result = _result(tmp_path)

    # corr_min_periods=50 returns -> first ready snapshot after bar 50,
    # used for bar 51.
    assert result.exposures.index[0] == _START + timedelta(days=51)


def test_market_exposure_matches_the_blended_beta(tmp_path: Path) -> None:
    result = _result(tmp_path)

    # 0.5 * beta(MKT)=1 + 0.5 * beta(A)=0.5
    assert result.exposures["Market"].iloc[-100:].mean() == pytest.approx(0.75, abs=0.15)


def test_bias_statistic_is_close_to_one(tmp_path: Path) -> None:
    result = _result(tmp_path)

    assert 0.7 < result.bias_statistic() < 1.3
    assert result.rolling_bias(window=100).dropna().size > 0


def test_empty_history_gives_empty_frames(tmp_path: Path) -> None:
    events, _, _ = _scenario(n_bars=30)
    model = FactorRiskModel(["MKT", "A", "B"], market_factor_config(tmp_path))

    result = run_attribution(events, [], [], model)

    assert result.exposures.empty
    assert result.contributions.empty
    assert math.isnan(result.bias_statistic())
