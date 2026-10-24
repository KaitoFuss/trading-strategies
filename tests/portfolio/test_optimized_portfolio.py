from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from backtester.core.events import Position, SignalEvent

from trading_strategies.portfolio import optimized_portfolio as optimized_module
from trading_strategies.portfolio.optimized_portfolio import OptimizedPortfolio
from trading_strategies.portfolio.solver import SolveResult
from trading_strategies.risk.config import FactorModelConfig, MomentumConfig, OptimizerConfig
from trading_strategies.utils.streaming import FloatArray

_TICKERS = ("MKT", "A", "B")


class _FakePriceSource:
    def __init__(self) -> None:
        self.prices: dict[str, float] = {}

    def get_price(self, ticker: str) -> float | None:
        return self.prices.get(ticker)


def _factor_file(tmp_path: Path) -> str:
    path = tmp_path / "factors.json"
    path.write_text(
        '{"factors": [{"name": "Market", "level": 1, "granular": false, "weights": {"MKT": 1.0}}]}'
    )
    return str(path)


def _portfolio(tmp_path: Path, **optimizer: object) -> tuple[OptimizedPortfolio, _FakePriceSource]:
    source = _FakePriceSource()
    portfolio = OptimizedPortfolio(
        price_source=source,
        initial_cash=1_000_000.0,
        target_vol=0.15,
        max_gross=2.0,
        model_config=FactorModelConfig(
            factors_file=_factor_file(tmp_path),
            vol_span=20,
            corr_span=100,
            corr_min_periods=50,
            momentum=MomentumConfig(lookback=20, skip=2),
        ),
        optimizer_config=OptimizerConfig(**optimizer),  # type: ignore[arg-type]
        tickers=_TICKERS,
    )
    return portfolio, source


def _drive(portfolio: OptimizedPortfolio, source: _FakePriceSource, bars: int) -> list[object]:
    rng = np.random.default_rng(0)
    closes = dict.fromkeys(_TICKERS, 100.0)
    orders: list[object] = []
    start = datetime(2020, 1, 1)
    for i in range(bars):
        market = rng.normal(0, 0.01)
        closes = {t: c * float(np.exp(market + rng.normal(0, 0.005))) for t, c in closes.items()}
        source.prices.update(closes)
        orders = list(
            portfolio.process_signal(
                SignalEvent(timestamp=start + timedelta(days=i), scores={"A": 1.0, "B": -1.0})
            )
        )
    return orders


def test_no_orders_before_the_model_is_ready(tmp_path: Path) -> None:
    portfolio, source = _portfolio(tmp_path)

    orders = _drive(portfolio, source, 20)

    assert orders == []
    assert portfolio.solve_log == []


def test_trades_once_the_model_is_ready(tmp_path: Path) -> None:
    portfolio, source = _portfolio(tmp_path)

    _drive(portfolio, source, 120)

    assert portfolio.solve_log
    last = portfolio.solve_log[-1]
    assert last.status == "optimal"
    assert last.gross <= 2.0 + 1e-6
    assert last.binding in {"vol", "leverage", "both", "turnover"}


def test_a_factor_bound_is_passed_through_to_the_solver(tmp_path: Path) -> None:
    portfolio, source = _portfolio(tmp_path, factor_bounds={"Market": (-0.05, 0.05)})

    _drive(portfolio, source, 120)

    assert portfolio.solve_log
    assert portfolio.solve_log[-1].status == "optimal"


def test_current_weight_reflects_a_held_position(tmp_path: Path) -> None:
    portfolio, source = _portfolio(tmp_path)
    source.prices["A"] = 10.0
    portfolio._positions["A"] = Position(
        ticker="A", quantity=100, entry_price=10.0, entry_date=datetime(2020, 1, 1)
    )

    weight = portfolio._current_weight("A", equity=10_000.0)

    assert weight == pytest.approx(0.1)


def test_orders_from_targets_skips_a_ticker_with_no_price(tmp_path: Path) -> None:
    portfolio, _source = _portfolio(tmp_path)

    orders = portfolio._orders_from_targets(
        {"Z": 0.1}, equity=1_000.0, timestamp=datetime(2020, 1, 1)
    )

    assert orders == []


def test_keeps_previous_weights_when_the_solver_status_is_not_optimal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    portfolio, source = _portfolio(tmp_path)
    _drive(portfolio, source, 120)
    assert portfolio.solve_log[-1].status == "optimal"

    def _failing_solve(
        scores: object, snapshot: object, w_prev: FloatArray, params: object
    ) -> SolveResult:
        return SolveResult(
            weights=w_prev, status="infeasible", binding="none", ex_ante_vol=0.0, gross=0.0
        )

    monkeypatch.setattr(optimized_module, "solve_weights", _failing_solve)

    with caplog.at_level("WARNING"):
        orders = list(
            portfolio.process_signal(
                SignalEvent(timestamp=datetime(2020, 1, 1), scores={"A": 1.0, "B": -1.0})
            )
        )

    assert orders == []
    assert portfolio.solve_log[-1].status == "infeasible"
    assert any("infeasible" in record.message for record in caplog.records)


def test_mean_variance_settings_reach_the_solver(tmp_path: Path) -> None:
    portfolio, source = _portfolio(tmp_path, objective="mean_variance", risk_aversion=1000.0)

    _drive(portfolio, source, 80)

    assert portfolio.solve_log
    assert {entry.binding for entry in portfolio.solve_log} == {"interior"}
