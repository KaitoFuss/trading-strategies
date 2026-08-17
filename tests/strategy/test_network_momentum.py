from datetime import datetime, timedelta

import numpy as np
import pytest
from backtester.core.events import Bar, MarketEvent

from trading_strategies.strategy.network_momentum import (
    NetworkMomentumStrategy,
    _FactorReturnEwma,
)

TICKERS = [f"T{i}" for i in range(12)]  # >= n_params (9) so daily OLS isn't underdetermined
N_DAYS = 400


def _synthetic_bars(day: int) -> dict[str, Bar]:
    return {
        ticker: Bar(close=100.0 + (day + offset) % 17 - 8 + offset * 0.01)
        for offset, ticker in enumerate(TICKERS)
    }


def _run(n_days: int) -> list[dict[str, float]]:
    strategy = NetworkMomentumStrategy()
    start = datetime(2020, 1, 1)
    scores_by_day = []
    for day in range(n_days):
        event = MarketEvent(timestamp=start + timedelta(days=day), bars=_synthetic_bars(day))
        signal = strategy.process_market(event)
        scores_by_day.append(dict(signal.scores))
    return scores_by_day


def test_scores_are_empty_during_warmup() -> None:
    scores_by_day = _run(n_days=50)

    assert all(scores == {} for scores in scores_by_day)


def test_scores_are_populated_once_features_and_lambda_history_exist() -> None:
    scores_by_day = _run(n_days=N_DAYS)

    last_day_scores = scores_by_day[-1]
    assert last_day_scores != {}
    assert set(last_day_scores) <= set(TICKERS)


def test_factor_return_ewma_tracks_the_recent_regime_not_the_full_history() -> None:
    # A full-sample expanding mean of an equal run of +1s then -1s is ~0 and
    # can't reprice the factor after the flip. A halflife-weighted mean, fed
    # a long recent run of -1, must sit near -1 regardless of the +1 history.
    ewma = _FactorReturnEwma(halflife=10.0)
    for _ in range(200):
        ewma.update(np.array([1.0, 1.0]))
    for _ in range(200):
        value = ewma.update(np.array([-1.0, -1.0]))

    assert value is not None
    assert value == pytest.approx(np.array([-1.0, -1.0]), abs=1e-3)


def test_factor_return_ewma_is_none_before_the_first_observation() -> None:
    assert _FactorReturnEwma(halflife=10.0).value is None


def test_strategy_feeds_the_shared_risk_model() -> None:
    from trading_strategies.factor_risk import FactorRiskModel

    model = FactorRiskModel(halflife=50.0)
    strategy = NetworkMomentumStrategy(risk_model=model)
    start = datetime(2020, 1, 1)
    last_scores: dict[str, float] = {}
    for day in range(N_DAYS):
        event = MarketEvent(timestamp=start + timedelta(days=day), bars=_synthetic_bars(day))
        last_scores = dict(strategy.process_market(event).scores)

    tickers = list(last_scores)
    assert len(tickers) >= 2
    assert model.factor_covariance() is not None  # fed with factor returns
    # The structure the strategy fed must yield a usable, positive-definite Sigma.
    assert np.linalg.eigvalsh(model.covariance(tickers)).min() > 0


def test_scores_at_day_t_are_unaffected_by_data_after_day_t() -> None:
    # The gold-standard causality check: run once stopping at day T, run again
    # continuing past T, and confirm day T's score is bit-identical either way.
    short_run = _run(n_days=300)
    long_run = _run(n_days=350)

    assert short_run[299] == long_run[299]
