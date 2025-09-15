import numpy as np
from loguru import logger

from vcb.models.anndata import AnnotatedDataMatrix


def get_gene_labels(a: AnnotatedDataMatrix, gene_id_column: str) -> list[str]:
    """
    Get the gene labels from an annotated data matrix.
    """
    return a.var[gene_id_column].to_list()


def match_gene_space(
    a: AnnotatedDataMatrix,
    b: AnnotatedDataMatrix,
    gene_id_column_a: str = "ensembl_gene_id",
    gene_id_column_b: str = "ensembl_gene_id",
) -> tuple[AnnotatedDataMatrix, AnnotatedDataMatrix]:
    """
    Match genes between two annotated data matrices.
    """
    labels_a = get_gene_labels(a, gene_id_column_a)
    labels_b = get_gene_labels(b, gene_id_column_b)

    intersection = list(set(labels_a) & set(labels_b))

    mask_a = np.isin(labels_a, np.array(intersection))
    mask_b = np.isin(labels_b, np.array(intersection))

    # Since data copies are expensive with large arrays,
    # we set a mask instead of directly modifying the features.
    # This way it's also not a destructive operation.
    a.set_var_mask(mask_a)
    b.set_var_mask(mask_b)

    logger.info(
        f"Matched gene space: {len(labels_a)} -> {len(intersection)}, {len(labels_b)} -> {len(intersection)}"
    )
    return a, b
