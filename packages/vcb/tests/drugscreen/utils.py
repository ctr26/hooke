def mock_drugscreen_genetic_perturbation():
    # Intentionally incorrectly ordered
    return [
        {
            "type": "genetic",
            "ensembl_gene_id": "ENSG000000000000000000",
            "hours_post_reference": 0,
        },
    ]


def mock_drugscreen_compound_perturbation(pert: str):
    # Intentionally incorrectly ordered
    return [
        {
            "type": "compound",
            "inchikey": pert,
            "concentration": 1.0,
            "hours_post_reference": 1,
        }
    ] + mock_drugscreen_genetic_perturbation()
