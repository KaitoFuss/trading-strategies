from datetime import datetime, timedelta

import numpy as np
import pytest
from backtester.core.events import FillEvent, OrderEvent, Position, SignalEvent

from trading_strategies.factor_risk import FactorRiskModel
from trading_strategies.portfolio.mean_variance import MeanVariancePortfolio


class _StubPriceSource:
    def __init__(self) -> None:
        self.prices: dict[str, float] = {}

    def get_price(self, ticker: str) -> float | None:
        return self.prices.get(ticker)


def _signal(scores: dict[str, float], day: int) -> SignalEvent:
    return SignalEvent(timestamp=datetime(2020, 1, 1) + timedelta(days=day), scores=scores)


def _risk_model_for(tickers: list[str], *, halflife: float = 20.0, n: int = 60) -> FactorRiskModel:
    """A FactorRiskModel warmed up so ``covariance`` is well-defined and
    positive-definite for ``tickers`` -- the portfolio's only Sigma source."""
    model = FactorRiskModel(halflife=halflife)
    rng = np.random.default_rng(0)
    loadings = {t: [float(x) for x in rng.standard_normal(2)] for t in tickers}
    prev = None
    for _ in range(n):
        f = rng.standard_normal(2) * 0.01
        returns = {
            t: float(np.dot(loadings[t], f)) + rng.standard_normal() * 0.005 for t in tickers
        }
        model.update(loadings=loadings, prev_loadings=prev, factor_returns=f, returns=returns)
        prev = loadings
    return model


def test_no_orders_before_min_periods_of_price_history() -> None:
    price_source = _StubPriceSource()
    portfolio = MeanVariancePortfolio(
        price_source=price_source,
        initial_cash=1_000_000.0,
        min_periods=10,
        risk_model=_risk_model_for(["A", "B"]),
    )

    orders: list[OrderEvent] = []
    for day in range(5):
        for ticker, price in [("A", 100.0 + day), ("B", 50.0 - day * 0.1)]:
            price_source.prices[ticker] = price
        orders = list(portfolio.process_signal(_signal({"A": 1.0, "B": -1.0}, day)))

    assert orders == []


def test_orders_appear_once_min_periods_reached_and_are_roughly_dollar_neutral() -> None:
    price_source = _StubPriceSource()
    portfolio = MeanVariancePortfolio(
        price_source=price_source,
        initial_cash=1_000_000.0,
        max_gross=1.0,
        risk_aversion=1.0,
        min_periods=10,
        risk_model=_risk_model_for(["A", "B"]),
    )

    orders: list[OrderEvent] = []
    for day in range(15):
        price_source.prices["A"] = 100.0 + (day % 3)
        price_source.prices["B"] = 50.0 - (day % 5) * 0.2
        orders = list(portfolio.process_signal(_signal({"A": 1.0, "B": -1.0}, day)))

    assert orders != []
    signed_notional = sum(
        (o.quantity if o.direction == "BUY" else -o.quantity) * price_source.prices[o.ticker]
        for o in orders
    )
    equity = portfolio.mark_to_market()
    assert signed_notional == pytest.approx(0.0, abs=0.05 * equity)


def test_held_but_unscored_ticker_is_left_untouched_and_reserves_gross() -> None:
    price_source = _StubPriceSource()
    price_source.prices["C"] = 100.0
    portfolio = MeanVariancePortfolio(
        price_source=price_source,
        initial_cash=1_000_000.0,
        max_gross=1.0,
        min_periods=10,
        risk_model=_risk_model_for(["A", "B"]),
    )
    # Simulate an existing held position in C via a direct fill, bypassing
    # the optimizer -- this test is about C being left alone afterward, not
    # about how it got there.
    portfolio.process_fill(
        FillEvent(
            timestamp=datetime(2020, 1, 1),
            ticker="C",
            quantity=1000,
            direction="BUY",
            fill_price=100.0,
        )
    )
    assert portfolio.get_position("C") == Position(
        ticker="C", quantity=1000, entry_price=100.0, entry_date=datetime(2020, 1, 1)
    )

    for day in range(15):
        price_source.prices["A"] = 100.0 + (day % 3)
        price_source.prices["B"] = 50.0 - (day % 5) * 0.2
        price_source.prices["C"] = 100.0
        portfolio.process_signal(_signal({"A": 1.0, "B": -1.0}, day))

    position = portfolio.get_position("C")
    assert position is not None
    assert position.quantity == 1000


def test_factor_covariance_needs_no_shared_return_dates() -> None:
    # A trades bars 0-3, B trades bars 3-6: their return dates are disjoint, so
    # any sample covariance of asset returns would be undefined. The factor
    # covariance B*Sigma_f*B' + D depends only on the loadings and factor
    # structure, so the portfolio still sizes both names and trades.
    price_source = _StubPriceSource()
    portfolio = MeanVariancePortfolio(
        price_source=price_source, min_periods=2, risk_model=_risk_model_for(["A", "B"])
    )

    orders: list[OrderEvent] = []
    for day in range(7):
        prices: dict[str, float] = {}
        if day <= 3:
            prices["A"] = 100.0 + day
        if day >= 3:
            prices["B"] = 200.0 + day
        price_source.prices = prices
        orders = list(portfolio.process_signal(_signal({"A": 1.0, "B": -1.0}, day)))

    assert orders != []


def test_drift_band_suppresses_retrading_within_the_band() -> None:
    price_source = _StubPriceSource()
    portfolio = MeanVariancePortfolio(
        price_source=price_source,
        initial_cash=1_000_000.0,
        max_gross=1.0,
        risk_aversion=1.0,
        min_periods=10,
        drift_band=0.1,  # wider than the day-to-day wobble but narrower than an opening move
        risk_model=_risk_model_for(["A", "B"]),
    )

    all_orders: list[list[OrderEvent]] = []
    for day in range(20):
        price_source.prices["A"] = 100.0 + (day % 3)
        price_source.prices["B"] = 50.0 - (day % 5) * 0.2
        orders = list(portfolio.process_signal(_signal({"A": 1.0, "B": -1.0}, day)))
        # Simulate the fills the real engine would produce, at the signal
        # price with no cost -- without this, positions never update and
        # current_weight stays 0 forever, making the drift check meaningless.
        for order in orders:
            portfolio.process_fill(
                FillEvent(
                    timestamp=order.timestamp,
                    ticker=order.ticker,
                    quantity=order.quantity,
                    direction=order.direction,
                    fill_price=price_source.prices[order.ticker],
                )
            )
        all_orders.append(orders)

    first_nonempty = next(i for i, orders in enumerate(all_orders) if orders)
    assert all(orders == [] for orders in all_orders[first_nonempty + 1 :])
