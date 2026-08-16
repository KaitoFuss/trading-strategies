from collections.abc import Sequence

import pandas as pd


def cross_sectional_zscore(panel: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Standardize each column to zero mean, unit std within each date.

    Population std (ddof=0): the cross-section on a given date is the whole
    universe for that day, not a sample of a larger one.
    """
    result = panel.copy()

    def zscore(group: pd.DataFrame) -> pd.DataFrame:
        return (group - group.mean()) / group.std(ddof=0)

    result[columns] = panel.groupby("date")[list(columns)].transform(zscore)
    return result
