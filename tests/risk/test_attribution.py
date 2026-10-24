from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from backtester.core.events import Bar, MarketEvent
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR

from tests.risk.helpers import market_factor_config
from trading_strategies.risk.attribution import (
    OTHER,
    RESIDUAL,
    SELECTION,
    STRUCTURAL,
    TIMING,
    AttributionResult,
    run_attribution,
)
from trading_strategies.risk.model import MOMENTUM, FactorRiskModel

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
    assert result.edge_decomposition().empty
    assert math.isnan(result.bias_statistic())


def test_edge_buckets_add_up_to_the_portfolio_return(tmp_path: Path) -> None:
    result = _result(tmp_path)

    edge = result.edge_decomposition()

    assert list(edge.columns) == [STRUCTURAL, TIMING, MOMENTUM, SELECTION, OTHER]
    np.testing.assert_allclose(
        edge.sum(axis=1).to_numpy(), result.portfolio_returns.to_numpy(), atol=1e-15
    )


def test_structural_plus_timing_equals_each_factor_contribution(tmp_path: Path) -> None:
    result = _result(tmp_path)
    factors = list(result.exposures.columns)

    total = result.structural_returns() + result.timing_returns()

    pd.testing.assert_frame_equal(total, result.contributions[factors], atol=1e-15)


def test_selection_plus_other_equals_the_old_residual(tmp_path: Path) -> None:
    result = _result(tmp_path)

    edge = result.edge_decomposition()

    np.testing.assert_allclose(
        (edge[SELECTION] + edge[OTHER]).to_numpy(),
        result.contributions[RESIDUAL].to_numpy(),
        atol=1e-15,
    )


def test_selection_is_the_holdings_return_net_of_factors(tmp_path: Path) -> None:
    # Weights match the simple-return equity curve, so "other" is only the
    # log-of-sum vs sum-of-logs gap: tiny next to the specific returns.
    result = _result(tmp_path)

    edge = result.edge_decomposition()

    assert edge[OTHER].abs().max() < 1e-3
    assert edge[SELECTION].abs().mean() > 10 * edge[OTHER].abs().mean()


def _hand_built(exposure: list[float], factor_return: list[float]) -> AttributionResult:
    dates = pd.date_range("2024-01-01", periods=len(exposure), freq="D")
    x = pd.DataFrame({"Equity": exposure}, index=dates)
    f = pd.DataFrame({"Equity": factor_return}, index=dates)
    contributions = (x * f).assign(**{RESIDUAL: 0.0})
    returns = pd.Series((x * f)["Equity"].to_numpy(), index=dates, dtype=float)
    return AttributionResult(
        exposures=x,
        risk_decomposition=pd.DataFrame(index=dates),
        contributions=contributions,
        portfolio_returns=returns,
        predicted_vol=pd.Series(0.01, index=dates),
        factor_returns=f,
        specific_returns=pd.Series(0.0, index=dates, dtype=float),
    )


def test_constant_exposure_has_no_timing() -> None:
    result = _hand_built([0.8, 0.8, 0.8, 0.8], [0.01, -0.02, 0.03, 0.005])

    np.testing.assert_allclose(result.timing_returns()["Equity"].to_numpy(), 0.0, atol=1e-18)
    np.testing.assert_allclose(
        result.structural_returns()["Equity"].to_numpy(), [0.008, -0.016, 0.024, 0.004]
    )


def test_timing_sums_to_n_times_the_exposure_return_covariance() -> None:
    x = [1.0, -1.0, 1.0, -1.0]
    f = [0.02, -0.01, 0.03, -0.02]
    result = _hand_built(x, f)

    covariance = float(np.cov(x, f, bias=True)[0, 1])

    assert result.timing_returns()["Equity"].sum() == pytest.approx(4 * covariance)
    assert result.structural_returns()["Equity"].sum() == pytest.approx(0.0, abs=1e-18)


def test_edge_summary_reports_annualized_mean_and_t_stat() -> None:
    result = _hand_built([1.0, 0.5, 1.5, 1.0, 0.0], [0.01, -0.02, 0.015, 0.0, 0.03])

    summary = result.edge_summary()

    timing = result.edge_decomposition()[TIMING]
    assert summary.loc[TIMING, "annualized"] == pytest.approx(timing.mean() * TRADING_DAYS_PER_YEAR)
    assert summary.loc[TIMING, "t_stat"] == pytest.approx(
        timing.mean() / timing.std() * math.sqrt(len(timing))
    )
    assert summary.loc[OTHER, "t_stat"] == 0.0  # zero series: no t-stat, not NaN


def test_factor_summary_splits_each_factor() -> None:
    result = _hand_built([0.8, 0.8, 0.8, 0.8], [0.01, -0.02, 0.03, 0.005])

    summary = result.factor_summary()

    assert list(summary.columns) == ["structural", "timing", "t_timing"]
    assert summary.loc["Equity", "structural"] == pytest.approx(0.005 * TRADING_DAYS_PER_YEAR)
    assert summary.loc["Equity", "timing"] == pytest.approx(0.0, abs=1e-15)


def _regression_result(n: int = 2000, seed: int = 1) -> AttributionResult:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    f = pd.DataFrame(
        {"Equity": rng.normal(0.0004, 0.01, n), "Rates": rng.normal(0.0001, 0.005, n)},
        index=dates,
    )
    simple = 0.0002 + 0.8 * f["Equity"] - 0.5 * f["Rates"] + rng.normal(0, 0.002, n)
    returns = pd.Series(np.log1p(simple.to_numpy()), index=dates, dtype=float)
    return AttributionResult(
        exposures=pd.DataFrame({"Equity": 0.8, "Rates": -0.5}, index=dates),
        risk_decomposition=pd.DataFrame(index=dates),
        contributions=pd.DataFrame(index=dates),
        portfolio_returns=returns,
        predicted_vol=pd.Series(0.01, index=dates),
        factor_returns=f,
        specific_returns=pd.Series(0.0, index=dates, dtype=float),
    )


def test_factor_regression_recovers_known_betas_and_alpha() -> None:
    regression = _regression_result().factor_regression()

    assert regression.table.loc["Equity", "beta"] == pytest.approx(0.8, abs=0.01)
    assert regression.table.loc["Rates", "beta"] == pytest.approx(-0.5, abs=0.02)
    assert regression.alpha == pytest.approx(0.0002 * TRADING_DAYS_PER_YEAR, abs=0.01)
    assert regression.table["t_stat"]["Equity"] > 50
    assert regression.r_squared > 0.9


def test_factor_regression_contributions_and_alpha_add_up_to_the_mean_return() -> None:
    result = _regression_result()

    regression = result.factor_regression()

    mean_simple = float(np.expm1(result.portfolio_returns).mean()) * TRADING_DAYS_PER_YEAR
    assert regression.alpha + regression.table["contribution"].sum() == pytest.approx(mean_simple)


def test_factor_regression_on_too_few_bars_is_nan() -> None:
    regression = _hand_built([1.0, 0.5], [0.01, -0.02]).factor_regression()

    assert math.isnan(regression.alpha)
    assert math.isnan(regression.r_squared)
    assert regression.table["beta"].isna().all()
