from typing import Literal, Self

from loguru import logger

from vcb.data_models.dataset.anndata import TxAnnotatedDataMatrix
from vcb.preprocessing.steps.base import PreprocessingStep


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

    def fit(self, ground_truth: TxAnnotatedDataMatrix, predictions: TxAnnotatedDataMatrix) -> Self:
        """
        Find the intersection of the gene labels across all data matrices.
        This is the gene subset that we will match all data matrices to.
        """
        intersection = set(ground_truth.gene_ids)
        intersection &= set(predictions.gene_ids)
        intersection.discard(None)
        self.gene_subset = sorted(list(intersection))
        return self

    def transform_single(self, data: TxAnnotatedDataMatrix):
        """
        Transform a single data matrix to the gene subset.
        """
        labels = data.gene_ids
        indices = [labels.index(label) for label in self.gene_subset]
        data.filter(var_indices=indices)
        logger.info(f"Matched gene space: From {len(labels)} genes to {len(indices)} genes")

    def transform(self, ground_truth: TxAnnotatedDataMatrix, predictions: TxAnnotatedDataMatrix):
        """
        Match the gene space of the data matrices to the gene subset.
        """
        if not self.fitted:
            raise RuntimeError("The gene subset is not fitted. Please call fit() first.")

        self.transform_single(ground_truth)
        self.transform_single(predictions)

        if len(ground_truth.var) != len(predictions.var):
            raise ValueError("The ground truth and predictions have different numbers of genes.")
