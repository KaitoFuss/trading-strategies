from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trading_strategies.data import fetch_total_return_to_parquet


def test_fetch_forces_auto_adjust_and_writes_adjusted_close(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs)
        columns = pd.MultiIndex.from_product([["HYG"], ["Open", "High", "Low", "Close", "Volume"]])
        index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
        return pd.DataFrame(
            [[1.0, 2.0, 0.5, 1.5, 100.0], [1.5, 2.5, 1.0, 2.0, 200.0]], index, columns
        )

    out_path = tmp_path / "raw.parquet"
    fetch_total_return_to_parquet(["HYG"], "2024-01-01", "2024-01-04", out_path, fake_download)

    assert len(calls) == 1
    assert calls[0]["auto_adjust"] is True
    frame = pd.read_parquet(out_path)
    assert frame["ticker"].tolist() == ["HYG", "HYG"]
    assert frame["close"].tolist() == [1.5, 2.0]
