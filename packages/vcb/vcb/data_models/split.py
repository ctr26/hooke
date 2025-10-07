from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator

from vcb.data_models.misc import IndexSet


class Fold(BaseModel):
    """A single fold in a repeated cross-validation split."""

    outer_fold: int
    inner_fold: int

    finetune: IndexSet
    test: IndexSet
    validation: IndexSet = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_no_overlap_between_indices(self) -> "Fold":
        """
        Assert there is no overlap between the indices.
        """
        if len(set(self.all_indices)) != len(self):
            raise ValueError("The different index sets in the fold are not disjoint")
        return self

    @property
    def all_indices(self) -> tuple[int, int, int]:
        return self.finetune + self.validation + self.test

    @property
    def sizes(self) -> tuple[int, int, int]:
        return len(self.finetune), len(self.validation), len(self.test)

    @property
    def ratios(self) -> tuple[float, float, float]:
        return tuple([n / len(self) for n in self.sizes])

    def __len__(self) -> int:
        return sum(self.sizes)

    def __str__(self) -> str:
        return f"Fold ({self.outer_fold}, {self.inner_fold}): Total size: {len(self)}, Ratio: ({self.ratios[0]:.2f}, {self.ratios[1]:.2f}, {self.ratios[2]:.2f})"


class Split(BaseModel):
    """A split specification for a dataset."""

    dataset_id: str
    version: int

    splitting_level: str
    splitting_strategy: str

    folds: list[Fold]
    controls: IndexSet
    base_states: IndexSet

    @model_validator(mode="after")
    def validate_no_overlap_between_folds(self) -> "Split":
        """
        Assert there is no overlap between the folds.
        """
        for fold in self.folds:
            index_set = set(fold.all_indices)
            if len(index_set.intersection(self.controls)) > 0:
                raise ValueError(f"Fold {fold.outer_fold}, {fold.inner_fold} contains controls")
            if len(index_set.intersection(self.base_states)) > 0:
                raise ValueError(f"Fold {fold.outer_fold}, {fold.inner_fold} contains base states")
        return self

    @classmethod
    def from_json(cls, json_path: Path) -> "Split":
        with open(json_path, "r") as fd:
            split = cls.model_validate_json(fd.read())
        return split

    def __str__(self) -> str:
        all_ratios = [list(fold.ratios) for fold in self.folds]
        mean_ratios = np.mean(all_ratios, axis=0)
        std_ratios = np.std(all_ratios, axis=0)

        ratios_str = [f"{mu:.2f} ± {std:.2f}" for mu, std in zip(mean_ratios, std_ratios)]

        all_sizes = [len(fold) for fold in self.folds]
        s = [
            f"Split ({self.dataset_id}, version {self.version}):",
            f"  Controls:      {len(self.controls)}",
            f"  Base states:   {len(self.base_states)}",
            f"  Folds:         {len(self.folds)}",
            "    • Ratio:",
            f"      ◦ Train:   {ratios_str[0]}",
            f"      ◦ Val:     {ratios_str[1]}",
            f"      ◦ Test:    {ratios_str[2]}",
            f"    • Size:      {np.mean(all_sizes):.2f} ± {np.std(all_sizes):.2f}",
        ]
        return "\n".join(s)
