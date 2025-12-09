from typing import Literal, Self

from loguru import logger

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.preprocessing.steps.base import PreprocessingStep


def get_gene_labels(a: AnnotatedDataMatrix, gene_id_column: str) -> list[str]:
    """
    Get the gene labels from an annotated data matrix.
    """
    return a.var[gene_id_column].to_list()


class MatchGenesStep(PreprocessingStep):
    """
    A step that will match the gene space of two or more annotated data matrices.
    """

    kind: Literal["match_genes"] = "match_genes"

    gene_subset: list[str] | None = None
    ground_truth_gene_id_column: str = "ensembl_gene_id"
    predictions_gene_id_column: str = "ensembl_gene_id"

    @property
    def fitted(self) -> bool:
        return self.gene_subset is not None

    def fit(self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix) -> Self:
        """
        Find the intersection of the gene labels across all data matrices.
        This is the gene subset that we will match all data matrices to.
        """
        intersection = set(get_gene_labels(ground_truth, self.ground_truth_gene_id_column))
        intersection &= set(get_gene_labels(predictions, self.predictions_gene_id_column))
        intersection.discard(None)
        self.gene_subset = sorted(list(intersection))
        return self

    def transform_single(self, data: AnnotatedDataMatrix, gene_id_column: str):
        """
        Transform a single data matrix to the gene subset.
        """
        labels = get_gene_labels(data, gene_id_column)
        indices = [labels.index(label) for label in self.gene_subset]
        data.filter(var_indices=indices)
        logger.info(f"Matched gene space: From {len(labels)} genes to {len(indices)} genes")

    def transform(self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix):
        """
        Match the gene space of the data matrices to the gene subset.
        """
        if not self.fitted:
            raise RuntimeError("The gene subset is not fitted. Please call fit() first.")

        self.transform_single(ground_truth, self.ground_truth_gene_id_column)
        self.transform_single(predictions, self.predictions_gene_id_column)

        if len(ground_truth.var) != len(predictions.var):
            raise ValueError("The ground truth and predictions have different numbers of genes.")
