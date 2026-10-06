"""Shared synthetic OHLC-parquet generator for backtest integration tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def write_synthetic_parquet(
    path: Path, tickers: Sequence[str], num_bars: int, seed: int = 7
) -> None:
    rng = np.random.default_rng(seed)
    start = datetime(2020, 1, 1)
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        closes = 100 + np.cumsum(rng.normal(0, 1, num_bars))
        for i, close in enumerate(closes):
            rows.append(
                {
                    "date": start + timedelta(days=i),
                    "ticker": ticker,
                    "close": float(close),
                }
            )
    pd.DataFrame(rows).to_parquet(path)
