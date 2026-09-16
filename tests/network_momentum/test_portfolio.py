from datetime import datetime, timedelta

import numpy as np
from backtester.core.events import SignalEvent

from trading_strategies.network_momentum.portfolio import TargetVolPortfolio


class _FakePriceSource:
    def __init__(self) -> None:
        self._prices: dict[str, float] = {}

    def set_price(self, ticker: str, price: float) -> None:
        self._prices[ticker] = price

    def get_price(self, ticker: str) -> float | None:
        return self._prices.get(ticker)


def _signal(timestamp: datetime, scores: dict[str, float]) -> SignalEvent:
    return SignalEvent(timestamp=timestamp, scores=scores)


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
