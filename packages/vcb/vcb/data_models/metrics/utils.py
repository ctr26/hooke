from typing import Any, Iterator

import polars as pl
from tqdm import tqdm


def manual_group_by(df: pl.DataFrame, cols: set[str], description: str = "") -> Iterator[dict[str, Any]]:
    """
    We don't use the actual groups that are returned by the group_by method in Polars,
    but rather derive Polars-compatible filter expressions for each group. This is a utitlity method to do so.

    Since we group across both the prediction and ground truth tasks simultaneously,
    using the actual group_by method in Polars would also have been very verbose since we need to do each group by twice.
    """
    cols = sorted(list(cols))
    groups = df[cols].drop_nulls().unique()
    for value in tqdm(groups.iter_rows(), total=len(groups), leave=False, desc=description):
        yield {cols[idx]: value[idx] for idx in range(len(cols))}
