import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator


class SynchronizedDataset(BaseModel):
    """
    In the context of the rescue screen data, a dataset is metadata (or obs) and features (or X).
    When we initialize the script, these are paired such that row 1 in the obs is paired with row 1 in the X.
    We use this class to treat a dataset as a single unit and to keep obs and X in sync.

    NOTE (cwognum): This leads several copies of the data in memory. That may need to be optimized in the future.
        We'll cross that bridge when we get to it.
    """

    obs: pl.DataFrame
    X: np.ndarray

    _index_column: str = "original_index"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_same_length(self) -> "SynchronizedDataset":
        """
        We assume obs and X are paired such that row 1 in the obs is paired with row 1 in the X.
        We therefore only check if the length is equal.
        """

        if len(self.obs) != self.X.shape[0]:
            raise ValueError(f"obs and X length mismatch: {len(self.obs)} != {self.X.shape[0]}")

        if self._index_column in self.obs.columns:
            self.obs = self.obs.drop(self._index_column)
        self.obs = self.obs.with_row_index(self._index_column)

        return self

    def filter(self, predicate: pl.Expr) -> "SynchronizedDataset":
        """
        Filter both the metadata and features, such that both stay in sync.
        """
        obs = self.obs.filter(predicate)
        indices = obs[self._index_column].to_list()
        X = self.X[indices]

        return SynchronizedDataset(obs=obs, X=X)

    def join(self, other: "SynchronizedDataset") -> "SynchronizedDataset":
        """
        Join two datasets together by concatenating their metadata and features.
        """
        obs = pl.concat([self.obs, other.obs])
        X = np.vstack([self.X, other.X])
        return SynchronizedDataset(obs=obs, X=X)

    def group_by(self, groupby_columns: list[str]):
        """
        Group by the metadata and return a generator of SynchronizedDatasets.
        """
        for name, group in self.obs.group_by(groupby_columns, maintain_order=True):
            indices = group[self._index_column].to_list()
            X = self.X[indices]
            yield name, SynchronizedDataset(obs=group, X=X)

    def __len__(self) -> int:
        return len(self.obs)


def stack(datasets: list[SynchronizedDataset]) -> SynchronizedDataset:
    """Stack a list of datasets together by concatenating their metadata and features."""
    if len(datasets) == 1:
        return datasets[0]
    data = datasets[0]
    for dataset in datasets[1:]:
        data = data.join(dataset)
    return data
