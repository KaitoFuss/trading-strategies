from pathlib import Path

import pandas as pd

from trading_strategies.features.macd import macd_features
from trading_strategies.features.momentum import vol_scaled_return_features

FEATURE_COLUMNS = [
    "vol_scaled_return_1",
    "vol_scaled_return_21",
    "vol_scaled_return_63",
    "vol_scaled_return_126",
    "vol_scaled_return_252",
    "macd_8_24",
    "macd_16_48",
    "macd_32_96",
]


def _features_for_ticker(prices: pd.Series) -> pd.DataFrame:
    return pd.concat([vol_scaled_return_features(prices), macd_features(prices)], axis=1)


def build_feature_panel(raw_parquet: Path) -> pd.DataFrame:
    """The 8-feature V_t panel for every ticker in ``raw_parquet``.

    One row per (date, ticker), columns = FEATURE_COLUMNS. Rows before a
    ticker's longest warm-up (252 + 60 days for the slowest vol-scaled
    return, on top of the MACD windows) are NaN, not dropped here — the
    caller decides how to handle warm-up (drop, or let the regression step
    ignore incomplete rows).
    """
    raw = pd.read_parquet(raw_parquet)
    panels = []
    for ticker, group in raw.sort_values("date").groupby("ticker"):
        group = group.set_index("date")
        features = _features_for_ticker(group["close"])
        features["ticker"] = ticker
        panels.append(features)
    panel = pd.concat(panels).reset_index(names="date")
    return panel[["date", "ticker", *FEATURE_COLUMNS]]
