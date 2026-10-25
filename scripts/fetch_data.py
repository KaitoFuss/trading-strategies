"""Fetch daily OHLCV data from yfinance and write one tidy Parquet file.

Usage:
    uv run scripts/fetch_data.py configs/fetch_data.json

Prices are total-return (split- and dividend-adjusted); see
``trading_strategies.data``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from backtester.config import FetchDataConfig

from trading_strategies.data import fetch_total_return_to_parquet

logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: fetch_data.py <config.json>")
    config = FetchDataConfig.from_json(Path(sys.argv[1]))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )

    out_path = Path(config.out)
    fetch_total_return_to_parquet(config.tickers, config.start, config.end, out_path)
    logger.info("Done — wrote %s", out_path)


if __name__ == "__main__":
    main()
