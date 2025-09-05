from pathlib import Path

from pydantic import BaseModel

from vcb.models.misc import IndexSet


class Fold(BaseModel):
    """A single fold in a repeated cross-validation split."""

    outer_fold: int
    inner_fold: int
    finetune: IndexSet
    test: IndexSet


class Split(BaseModel):
    """A split specification for a dataset."""

    folds: list[Fold]

    dataset_id: str
    version: int

    # In hindsight, we probably don't need the controls and base states here.
    # We can do the matching on the fly in the dataloader.
    controls: IndexSet
    base_states: IndexSet

    @classmethod
    def from_json(cls, json_path: Path) -> "Split":
        with open(json_path, "r") as fd:
            split = cls.model_validate_json(fd.read())
        return split
