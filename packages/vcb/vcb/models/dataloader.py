from typing import Literal, TypeVar

from pydantic import BaseModel, field_validator

from vcb.models.misc import IndexSet

BiologicalContext: TypeVar = dict[str, str]


class Perturbation(BaseModel):
    """
    Metadata information for a given perturbation.
    Currently pretty minimal
    """

    type: Literal["genetic", "compound"]

    # Molecular structure
    smiles: str | None

    # Natural IDs for genes and compounds
    inchikey: str | None
    ensembl_gene_id: str | None
    genetic_id: str | None

    # Dosage information
    concentration: float
    concentration_units: Literal["nM"]

    # Other metadata
    usage_class: Literal["query", "positive_control", "negative_control"]

    @property
    def id(self) -> str:
        return self.genetic_id if self.type == "genetic" else self.inchikey


class PerturbationGroup(BaseModel):
    """
    A group of indices for a perturbation.
    """

    controls: IndexSet
    base_states: IndexSet
    perturbed_states: IndexSet

    biological_context: BiologicalContext
    perturbations: list[Perturbation]

    @field_validator("perturbations")
    def validate_perturbations(cls, v: list[Perturbation]) -> list[Perturbation]:
        if any(p.usage_class == "positive_control" for p in v):
            raise ValueError("We currently do not support positive controls")
        return v

    @field_validator("perturbed_states", "base_states", "controls")
    def validate_indices_not_empty(cls, v: IndexSet) -> IndexSet:
        if len(v) == 0:
            raise ValueError("No indices found")
        return v
