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

    # We need consistently reorder the labels
    indices_a = [labels_a.index(label) for label in intersection]
    indices_b = [labels_b.index(label) for label in intersection]

    # Since data copies are expensive with large arrays,
    # we set the indices instead of directly modifying the features.
    # This way it's also not a destructive operation.
    a.set_var_indices(indices_a)
    b.set_var_indices(indices_b)

    logger.info(
        f"Matched gene space: {len(labels_a)} -> {len(intersection)}, {len(labels_b)} -> {len(intersection)}"
    )
    return a, b
