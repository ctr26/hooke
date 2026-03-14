from typing import Iterator

import polars as pl
from tqdm import tqdm

try:
    import txam
except ImportError:
    txam = None


def predicate_group_by(
    df: pl.DataFrame, cols: set[str], description: str = ""
) -> Iterator[tuple[list[pl.Expr], pl.DataFrame]]:
    """
    We don't use the actual groups that are returned by the group_by method in Polars,
    but rather derive Polars-compatible filter expressions for each group. This is a utitlity method to get that predicate as well.
    """
    cols = sorted(list(cols))
    groups = df[cols].drop_nulls().unique()
    for value in tqdm(groups.iter_rows(), total=len(groups), leave=False, desc=description):
        predicate = [pl.col(col) == value[idx] for idx, col in enumerate(cols)]
        group = df.filter(*predicate)
        label = {cols[idx]: value[idx] for idx in range(len(cols))}
        yield label, group, predicate


def is_txam_installed() -> bool:
    return txam is not None


def filter_negative_controls_and_duplicates(obs: pl.DataFrame) -> pl.DataFrame:
    """
    Filter out the negative controls from perturbations and save to
    a new column called "clean_perturbations".
    """
    obs = obs.with_columns(
        clean_perturbations=pl.col("perturbations").list.eval(
            pl.element()
            .filter(pl.element().struct.field("usage_class") != "negative_control")
            # CAUTION: the `unique` below will collapse identical perturbations (same ids, same concentration same time)
            # this is a lesser evil for rxrx datasets where "duplicates" like this are supposed to be
            # erroneous double logging, more often than anything that happened in lab. That said, this `unique` is a
            # sorta implicit and ugly way to handle this that could back fire if we ever really had
            # highly similar but supposed to be distinct perturbations.
            .unique(maintain_order=True)
        )
    )
    return obs
