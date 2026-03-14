from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict


class Map(BaseModel):
    """Light wrapper, primarily to co-locate IO logic"""

    similarity_matrix: np.ndarray
    embeddings: np.ndarray
    perturbations: list[tuple[str, ...]]
    cell_type: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def load(cls, path: Path) -> "Map":
        cache = np.load(path)
        return cls(
            similarity_matrix=cache["similarity_matrix"],
            embeddings=cache["embeddings"],
            perturbations=cache["perturbations"].tolist(),
            cell_type=str(cache["cell_type"]),
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            similarity_matrix=self.similarity_matrix,
            embeddings=self.embeddings,
            perturbations=np.array(self.perturbations),
            cell_type=self.cell_type,
        )
