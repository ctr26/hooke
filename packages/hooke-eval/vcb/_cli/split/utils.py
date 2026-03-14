from loguru import logger
import polars as pl


def log_step(step: str, count: int) -> None:
    """
    Log the summary of the filtering of a split with aligned formatting.
    """
    # Format with fixed width for alignment (adjust width as needed)
    logger.info(f"{step:<35} {count:>8,}")


POS_CONTROL_FILTER = (
    ~pl.col("perturbations")
    .list.eval(pl.element().struct.field("usage_class").eq("positive_control"))
    .list.any()
)
