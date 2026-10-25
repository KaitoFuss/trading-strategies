"""Fetch total-return (dividend-adjusted) prices into one tidy Parquet file.

backtester's ``fetch_to_parquet`` asks yfinance for ``auto_adjust=False``.
That ``Close`` is split-adjusted but not dividend-adjusted, so every
distribution shows up as a fake overnight loss on its ex-date. Bond ETFs pay
monthly, so price-only returns hide their carry and add a common shock on the
first trading day of each month. The backtester has no dividend handling, so
the fix has to be in the prices themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from backtester.data.fetch_yfinance import fetch_to_parquet


def fetch_total_return_to_parquet(
    tickers: list[str],
    start: str,
    end: str,
    out_path: Path,
    downloader: Callable[..., pd.DataFrame] = yf.download,
) -> None:
    """``fetch_to_parquet`` with OHLC back-adjusted for splits and dividends.

    The wrapper overrides ``auto_adjust`` to True; yfinance then returns the
    same Open/High/Low/Close/Volume columns, so backtester's column handling
    is unchanged. Back-adjusted price levels move on every refetch (each new
    distribution rescales history), but daily returns are total returns with
    no lookahead.
    """

    def adjusted(*args: Any, **kwargs: Any) -> pd.DataFrame:
        kwargs["auto_adjust"] = True
        return downloader(*args, **kwargs)

    fetch_to_parquet(tickers, start, end, out_path, downloader=adjusted)
