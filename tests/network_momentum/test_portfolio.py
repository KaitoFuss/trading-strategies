import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from backtester.core.engine import Portfolio
from backtester.core.events import FillEvent, OrderEvent, SignalEvent
from backtester.tracker.metrics import TRADING_DAYS_PER_YEAR

from trading_strategies.network_momentum.portfolio import (
    RescaledTargetVolPortfolio,
    TargetVolPortfolio,
)


class _FakePriceSource:
    def __init__(self) -> None:
        self._prices: dict[str, float] = {}

    def set_price(self, ticker: str, price: float) -> None:
        self._prices[ticker] = price

    def get_price(self, ticker: str) -> float | None:
        return self._prices.get(ticker)


# Type-check-only: pins that `TargetVolPortfolio` actually conforms to the
# `Portfolio` protocol under `mypy --strict`. Never used at runtime.
_portfolio_conforms: Portfolio = TargetVolPortfolio(price_source=_FakePriceSource())


def _signal(timestamp: datetime, scores: dict[str, float]) -> SignalEvent:
    return SignalEvent(timestamp=timestamp, scores=scores)


def _fill_order(
    portfolio: RescaledTargetVolPortfolio, order: OrderEvent, timestamp: datetime
) -> None:
    """Simulates an ``IdealExecutionHandler`` fill at the order's own last
    known price, so a test can put real (non-empty) positions on the book
    without running the whole engine."""
    price = portfolio._price_source.get_price(order.ticker)
    assert price is not None
    portfolio.process_fill(
        FillEvent(
            timestamp=timestamp,
            ticker=order.ticker,
            quantity=order.quantity,
            direction=order.direction,
            fill_price=price,
        )
    )


def test_target_vol_portfolio_skips_tickers_without_warmed_up_vol() -> None:
    prices = _FakePriceSource()
    prices.set_price("SPY", 100.0)
    portfolio = TargetVolPortfolio(price_source=prices, initial_cash=100_000.0)

    orders = portfolio.process_signal(_signal(datetime(2020, 1, 1), {"SPY": 0.5}))
    assert orders == []  # no return history yet -> vol not ready


def test_target_vol_portfolio_sizes_by_eq9_formula_once_warm() -> None:
    prices = _FakePriceSource()
    portfolio = TargetVolPortfolio(price_source=prices, initial_cash=100_000.0, target_vol=0.15)

    start = datetime(2020, 1, 1)
    price = 100.0
    rng = np.random.default_rng(0)
    for i in range(65):
        price *= 1 + rng.normal(0, 0.01)
        prices.set_price("SPY", price)
        # SPY must appear in `scores` every bar for `_update_vol` to track it
        # at all -- a ticker absent from every signal is invisible to this
        # portfolio, same as `InverseVolPortfolio`. Score of 0.0 here is a
        # placeholder just to keep SPY visible while vol warms up.
        portfolio.process_signal(_signal(start + timedelta(days=i), {"SPY": 0.0}))

    # one more bar, now meaningfully scored: check the order matches Eq. 9
    price *= 1.01
    prices.set_price("SPY", price)
    equity_before = portfolio.mark_to_market()
    orders = portfolio.process_signal(_signal(start + timedelta(days=65), {"SPY": 0.5}))

    assert len(orders) == 1
    order = orders[0]
    # white-box check of internal state; ruff has no private-access rule enabled here
    vol = portfolio._annualized_vol("SPY")
    assert vol is not None
    expected_weight = (0.5 / 1) * (0.15 / vol)
    expected_qty = round(expected_weight * equity_before / price)
    assert order.quantity == abs(expected_qty)
    assert order.direction == ("BUY" if expected_qty > 0 else "SELL")


def test_target_vol_portfolio_annualized_vol_matches_independent_pandas_calc() -> None:
    """`_annualized_vol` is the annualization step itself, not just the Eq. 9
    formula around it -- recompute it independently from the same raw price
    series with pandas' ewm, instead of round-tripping through the
    production method under test."""
    prices = _FakePriceSource()
    portfolio = TargetVolPortfolio(price_source=prices, initial_cash=100_000.0, target_vol=0.15)

    start = datetime(2020, 1, 1)
    price = 100.0
    rng = np.random.default_rng(0)
    recorded_prices: list[float] = []
    for i in range(65):
        price *= 1 + rng.normal(0, 0.01)
        prices.set_price("SPY", price)
        recorded_prices.append(price)
        portfolio.process_signal(_signal(start + timedelta(days=i), {"SPY": 0.0}))

    price *= 1.01
    prices.set_price("SPY", price)
    recorded_prices.append(price)
    portfolio.process_signal(_signal(start + timedelta(days=65), {"SPY": 0.5}))

    log_returns = pd.Series(recorded_prices).pct_change().apply(lambda r: math.log(1 + r))
    expected_std = log_returns.ewm(span=60, min_periods=60).std().iloc[-1]
    expected_vol = expected_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    vol = portfolio._annualized_vol("SPY")
    assert vol is not None
    assert vol == pytest.approx(expected_vol, rel=1e-9)


def test_target_vol_portfolio_excludes_unwarmed_ticker_from_n_t() -> None:
    """A bug that divides by `len(event.scores)` instead of `len(ready)`
    (the score+warm-vol subset) would only show up with 2+ tickers where one
    isn't warm yet -- all other tests here use a single ticker, so N=1 is a
    no-op that can't distinguish the two."""
    prices = _FakePriceSource()
    portfolio = TargetVolPortfolio(price_source=prices, initial_cash=100_000.0, target_vol=0.15)

    start = datetime(2020, 1, 1)
    price = 100.0
    rng = np.random.default_rng(0)
    for i in range(65):
        price *= 1 + rng.normal(0, 0.01)
        prices.set_price("SPY", price)
        portfolio.process_signal(_signal(start + timedelta(days=i), {"SPY": 0.0}))

    # final bar: SPY is warm, QQQ shows up for the very first time here and
    # is therefore not warm -- it must not count towards N_t.
    price *= 1.01
    prices.set_price("SPY", price)
    prices.set_price("QQQ", 50.0)
    equity_before = portfolio.mark_to_market()
    orders = portfolio.process_signal(_signal(start + timedelta(days=65), {"SPY": 0.5, "QQQ": 0.3}))

    # QQQ has no vol history yet, so it produces no order at all.
    assert len(orders) == 1
    order = orders[0]
    assert order.ticker == "SPY"

    vol = portfolio._annualized_vol("SPY")
    assert vol is not None
    expected_weight = (0.5 / 1) * (0.15 / vol)  # n=1: QQQ excluded from N_t
    expected_qty = round(expected_weight * equity_before / price)
    assert order.quantity == abs(expected_qty)
    assert order.direction == ("BUY" if expected_qty > 0 else "SELL")


def test_target_vol_portfolio_gross_is_uncapped() -> None:
    """A very-low-vol ticker with a strong score should be able to size
    past max_gross=1.0 -- Eq. 9 has no cross-sectional renormalization."""
    prices = _FakePriceSource()
    portfolio = TargetVolPortfolio(
        price_source=prices, initial_cash=100_000.0, target_vol=0.15, max_gross=1.0
    )

    start = datetime(2020, 1, 1)
    price = 100.0
    for i in range(65):
        # tiny, near-constant wiggle -> very low realized vol
        price *= 1 + (0.0001 if i % 2 == 0 else -0.0001)
        prices.set_price("SPY", price)
        # SPY must be present in `scores` every bar to stay visible to
        # `_update_vol` (see the previous test) -- 0.0 is a placeholder.
        portfolio.process_signal(_signal(start + timedelta(days=i), {"SPY": 0.0}))

    price *= 1.0001
    prices.set_price("SPY", price)
    equity = portfolio.mark_to_market()
    orders = portfolio.process_signal(_signal(start + timedelta(days=65), {"SPY": 1.0}))

    assert len(orders) == 1
    notional = orders[0].quantity * price
    assert notional / equity > 1.0  # gross > 100% of equity, i.e. uncapped


def test_rescaled_target_vol_portfolio_ignores_pre_trade_bars_for_portfolio_vol_warmup() -> None:
    """Bars before the first position opens produce a mechanically-zero
    equity return (nothing is held to earn a return) -- these must not
    count towards the portfolio-vol EWM's warm-up, or the first live
    rescale would divide by a near-zero vol built from manufactured zeros."""
    prices = _FakePriceSource()
    portfolio = RescaledTargetVolPortfolio(
        price_source=prices, initial_cash=100_000.0, target_vol=0.15
    )

    start = datetime(2020, 1, 1)
    price = 100.0
    rng = np.random.default_rng(0)
    # far more than the 60-bar warm-up window, but score stays 0.0 so no
    # position is ever opened and the book stays flat in cash throughout
    for i in range(120):
        price *= 1 + rng.normal(0, 0.01)
        prices.set_price("SPY", price)
        orders = portfolio.process_signal(_signal(start + timedelta(days=i), {"SPY": 0.0}))
        assert orders == []

    assert portfolio._annualized_portfolio_vol() is None


def test_rescaled_target_vol_portfolio_matches_unscaled_weights_before_portfolio_vol_warms() -> (
    None
):
    """Before the portfolio's own realized-return EWM has 60 samples, there
    is no valid portfolio-vol estimate to rescale by, so weights must fall
    back to the plain (unscaled) Eq. 9 formula -- identical to
    ``TargetVolPortfolio``."""
    prices = _FakePriceSource()
    portfolio = RescaledTargetVolPortfolio(
        price_source=prices, initial_cash=100_000.0, target_vol=0.15
    )

    start = datetime(2020, 1, 1)
    price = 100.0
    rng = np.random.default_rng(0)
    day = 0
    # warm SPY's own vol estimate first, score 0.0 so no position opens yet
    for _ in range(65):
        price *= 1 + rng.normal(0, 0.01)
        prices.set_price("SPY", price)
        portfolio.process_signal(_signal(start + timedelta(days=day), {"SPY": 0.0}))
        day += 1

    # open the first real position and keep trading, but fewer than 60
    # bars held -- the portfolio-vol EWM must still be unwarmed throughout.
    # Orders are a *delta* from the currently-held quantity, not the raw
    # target, so track the running position to predict them correctly.
    held_qty = 0
    for _ in range(30):
        price *= 1 + rng.normal(0, 0.01)
        prices.set_price("SPY", price)
        equity_before = portfolio.mark_to_market()
        orders = portfolio.process_signal(_signal(start + timedelta(days=day), {"SPY": 0.5}))

        vol = portfolio._annualized_vol("SPY")
        assert vol is not None
        expected_weight = (0.5 / 1) * (0.15 / vol)
        expected_target_qty = round(expected_weight * equity_before / price)
        expected_delta = expected_target_qty - held_qty
        assert portfolio._annualized_portfolio_vol() is None

        if expected_delta == 0:
            assert orders == []
        else:
            assert len(orders) == 1
            order = orders[0]
            assert order.quantity == abs(expected_delta)
            assert order.direction == ("BUY" if expected_delta > 0 else "SELL")
            _fill_order(portfolio, order, start + timedelta(days=day))
            held_qty = expected_target_qty
        day += 1


def test_rescaled_target_vol_portfolio_rescales_by_portfolio_vol_once_warm() -> None:
    """Once the portfolio-vol EWM is warm, every weight must be scaled by
    ``target_vol / portfolio_vol`` on top of the unscaled Eq. 9 weight."""
    prices = _FakePriceSource()
    portfolio = RescaledTargetVolPortfolio(
        price_source=prices, initial_cash=100_000.0, target_vol=0.15
    )

    start = datetime(2020, 1, 1)
    price = 100.0
    rng = np.random.default_rng(0)
    day = 0
    for _ in range(65):
        price *= 1 + rng.normal(0, 0.01)
        prices.set_price("SPY", price)
        portfolio.process_signal(_signal(start + timedelta(days=day), {"SPY": 0.0}))
        day += 1

    # hold a real position for 60+ bars so the portfolio-vol EWM warms up
    held_qty = 0
    for _ in range(65):
        price *= 1 + rng.normal(0, 0.01)
        prices.set_price("SPY", price)
        orders = portfolio.process_signal(_signal(start + timedelta(days=day), {"SPY": 0.5}))
        for order in orders:
            _fill_order(portfolio, order, start + timedelta(days=day))
            held_qty += order.quantity if order.direction == "BUY" else -order.quantity
        day += 1

    price *= 1 + rng.normal(0, 0.01)
    prices.set_price("SPY", price)
    equity_before = portfolio.mark_to_market()
    orders = portfolio.process_signal(_signal(start + timedelta(days=day), {"SPY": 0.5}))

    assert len(orders) == 1
    order = orders[0]
    vol = portfolio._annualized_vol("SPY")
    portfolio_vol = portfolio._annualized_portfolio_vol()
    assert vol is not None
    assert portfolio_vol is not None

    unscaled_weight = (0.5 / 1) * (0.15 / vol)
    expected_weight = unscaled_weight * (0.15 / portfolio_vol)
    expected_target_qty = round(expected_weight * equity_before / price)
    expected_delta = expected_target_qty - held_qty
    assert order.quantity == abs(expected_delta)
    assert order.direction == ("BUY" if expected_delta > 0 else "SELL")
