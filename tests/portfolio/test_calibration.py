from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_strategies.portfolio.calibration import (
    SweepRow,
    bias_statistic,
    fold_table,
    select_shrinkage,
)


def _series(values: list[float], start: str = "2010-01-01") -> pd.Series[float]:
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


def test_bias_statistic_is_one_for_a_perfect_forecast() -> None:
    rng = np.random.default_rng(0)
    realized = _series(list(rng.normal(0, 0.01, 2000)))
    forecast = _series([0.01] * 2000)

    assert bias_statistic(forecast, realized) == pytest.approx(1.0, abs=0.05)


def test_bias_statistic_doubles_when_realized_risk_doubles() -> None:
    rng = np.random.default_rng(0)
    realized = _series(list(rng.normal(0, 0.02, 2000)))
    forecast = _series([0.01] * 2000)

    assert bias_statistic(forecast, realized) == pytest.approx(2.0, abs=0.1)


def test_bias_statistic_ignores_bars_without_a_forecast() -> None:
    realized = _series([0.01, -0.01, 0.02, -0.02])
    forecast = _series([0.0, 0.0, 0.01, 0.01])

    assert not np.isnan(bias_statistic(forecast, realized))


def test_bias_statistic_is_nan_with_too_few_observations() -> None:
    assert np.isnan(bias_statistic(_series([0.01]), _series([0.005])))


def test_bias_statistic_needs_the_next_bars_return() -> None:
    """A forecast at ``t`` is responsible for the return earned at ``t + 1``.

    The forecast jumps on one bar and the matching large move lands on the bar
    after it. Pairing both at the same timestamp leaves a huge outlier in the
    standardized series, so the statistic drifts far from 1.0. The test fails if
    the ``.shift(-1)`` below is removed.
    """
    size = 120
    jump = 50
    realized_values = [0.01 * (-1) ** i for i in range(size)]
    realized_values[jump + 1] = 0.20 * (-1) ** (jump + 1)
    forecast_values = [0.01] * size
    forecast_values[jump] = 0.20
    realized = _series(realized_values)
    forecast = _series(forecast_values)

    aligned = bias_statistic(forecast, realized.shift(-1))
    same_timestamp = bias_statistic(forecast, realized)

    assert aligned == pytest.approx(1.0, abs=0.05)
    assert same_timestamp > 1.8
    assert abs(aligned - 1.0) < abs(same_timestamp - 1.0)


def test_select_shrinkage_prefers_the_smallest_delta_inside_the_band() -> None:
    rows = [
        SweepRow(0.0, 1.14, 0.20, 0.164, 12.0, 1.92),
        SweepRow(0.1, 1.04, 0.19, 0.152, 11.5, 1.90),
        SweepRow(0.2, 0.99, 0.19, 0.149, 11.0, 1.88),
    ]

    assert select_shrinkage(rows) == 0.1


def test_select_shrinkage_falls_back_to_the_closest_bias() -> None:
    rows = [SweepRow(0.0, 1.30, 0.20, 0.19, 12.0, 1.9), SweepRow(0.5, 1.12, 0.17, 0.17, 10.0, 1.8)]

    assert select_shrinkage(rows) == 0.5


def test_fold_table_reports_one_row_per_fold() -> None:
    index = pd.date_range("2008-01-01", "2025-12-31", freq="B")
    rng = np.random.default_rng(0)
    per_delta = {
        0.0: pd.DataFrame(
            {"forecast": 0.01, "realized": rng.normal(0, 0.012, len(index))}, index=index
        ),
        0.2: pd.DataFrame(
            {"forecast": 0.01, "realized": rng.normal(0, 0.010, len(index))}, index=index
        ),
    }

    table = fold_table(per_delta)

    assert list(table.index) == ["2013-2016", "2017-2020", "2021-2023", "2024-2025"]
    assert set(table.columns) == {"selected", "calibration_bias", "evaluation_bias"}
    # delta 0.2 forecasts better, so it should win every fold
    assert (table["selected"] == 0.2).all()
