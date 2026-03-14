import scanpy as sc
import json
from functools import partial
import datasets
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from pathlib import Path
from loguru import logger
import gc

from hooke_tx.data.chem_utils import standardize_smiles
from hooke_tx.data.constants import DATA_SOURCES, GENE_NAME, MOL_PERT_NAME, MOL_PERT, DOSE, GENE_PERT, CELL_TYPE, BATCH, CONTROL, CONTROL_LABEL, BASE, BASE_LABEL, EMPTY_LABEL, NEG_CONTROL_LABEL, UNKNOWN_LABEL
from hooke_tx.data.constants import TAHOE_2K_FULL

SKIP_LIST = [CONTROL_LABEL, BASE_LABEL, NEG_CONTROL_LABEL, UNKNOWN_LABEL, EMPTY_LABEL, ""]


def data_sources(data_sources) -> list[str]:
    if data_sources == ["Tahoe_2k_full"]:
        data_sources = TAHOE_2K_FULL
    else:
        data_sources = data_sources

    return data_sources


def read_filter_single_cell_type(
    src: str,
    cfg: dict
) -> sc.AnnData | None:
    """Load one cell-type AnnData and apply quality control filters."""
    ad = sc.read_h5ad(DATA_SOURCES[src], backed="r")

    if src == "K562" or src == "K562_plus":
        ad = _K562_formatter(ad)
    elif src.startswith("TREK_VCB_"):
        ad = _TREK_VCB_formatter(ad)
    elif "Tahoe_" in src and cfg.task_config["split_type"] == "explicit_fewshot_state":
        ad = _Tahoe_STATE_formatter(ad, src)
    elif src == "TREK_B_filtered":
        ad =_TREK_B_filtered_formatter(ad)  # TrekSeq data uses `np.nan` instead of "nan"
    elif src == "TREK_B_filtered_by_batch":
        ad = _TREK_B_filtered_formatter(ad)
    elif src == "TREK_B_unfiltered_by_batch":
        ad = _TREK_B_unfiltered_formatter(ad)
    else:
        raise ValueError(f"Unsupported data source: {src}")

    ad.obs = ad.obs[[MOL_PERT_NAME, MOL_PERT, DOSE, GENE_PERT, CELL_TYPE, BATCH, BASE, CONTROL]]

    ad.obs = ad.obs.astype("category")
    ad.obs_names_make_unique()

    keep_cond = None

    return (
        ad,
        keep_cond
    )


def _TREK_B_filtered_formatter(ad: sc.AnnData) -> sc.AnnData:
    """
    Temporary formatting for TREKseq (subset B) data until unifying data specs.
    """
    ad.obs[MOL_PERT] = ad.obs["SMILES"].astype(str)
    ad.obs[GENE_PERT] = ad.obs["disease_gene_x"].astype(str)
    ad.obs[CELL_TYPE] = "HUVEC"

    ad.obs[BATCH] = "TREK_B_filtered_" + ad.obs[BATCH].astype(str)

    ad.obs.loc[ad.obs["is_control"], MOL_PERT] = BASE_LABEL
    ad.obs[MOL_PERT] = ad.obs[MOL_PERT].astype("category")
    ad.obs[BASE] = ad.obs["is_control"].astype(int).astype("category")
    ad.obs[CONTROL] = 0
    ad.obs[CONTROL] = ad.obs[CONTROL].astype(int).astype("category")
    ad.obs[BATCH] = ad.obs[BATCH].astype("category")

    ad.obs[MOL_PERT_NAME] = ad.obs[MOL_PERT]

    ad.var[GENE_NAME] = list(ad.var_names)
    ad.var_names = list(ad.var[GENE_NAME])
    ad.var_names_make_unique()

    # Reservations
    with open("/rxrx/data/valence/internal_benchmarking/vcds1/v1_reservation.json", "r") as f:
        reservation = json.load(f)

    with open("/rxrx/data/valence/internal_benchmarking/vcds1/precautionary_exclusion.json", "r") as f:
        exclusion1 = json.load(f)

    with open("/rxrx/data/valence/internal_benchmarking/vcds1/precautionary_exclusion2.json", "r") as f:
        exclusion2 = json.load(f)

    indicator_rec_id = ad.obs["rec_id"].isin(reservation["rec_ids"])
    indicator_rsv = ad.obs["experiment_label"].isin(reservation["plated_rxrx_labels"])
    indicator_exclusion1 = ad.obs["experiment_label"].isin(exclusion1)
    indicator_exclusion2 = ad.obs["experiment_label"].isin(exclusion2)

    # Additional filter
    indicator_center_guide = ad.obs["center_guide"].astype(bool)

    filtered_indicator = indicator_rec_id | indicator_rsv | indicator_exclusion1 | indicator_exclusion2 | indicator_center_guide

    filtered_ad = ad[~filtered_indicator]
    
    # Filter for frequent covariates
    frequent_conditions = filtered_ad.obs[MOL_PERT].value_counts().index[filtered_ad.obs[MOL_PERT].value_counts().values > 50].tolist()
    frequent_contexts = filtered_ad.obs[GENE_PERT].value_counts().index[filtered_ad.obs[GENE_PERT].value_counts().values > 100].tolist()

    frequent_conditions.remove("")

    indicator_reduce_conditions = ~ad.obs[MOL_PERT].isin(frequent_conditions)
    indicator_reduce_contexts = ~ad.obs[GENE_PERT].isin(frequent_contexts)
    
    filtered_and_reduced_indicator = indicator_rec_id | indicator_rsv | indicator_exclusion1 | indicator_exclusion2 | indicator_center_guide | indicator_reduce_conditions | indicator_reduce_contexts
    ad = ad[~filtered_and_reduced_indicator].to_memory()

    # Standardize smiles
    ad.obs[MOL_PERT] = ad.obs[MOL_PERT].apply(partial(standardize_smiles, skip_list=SKIP_LIST))

    return ad


def _TREK_B_unfiltered_formatter(ad: sc.AnnData) -> sc.AnnData:
    """
    Temporary formatting for TREKseq (subset B) data until unifying data specs.
    """
    ad.obs[MOL_PERT] = ad.obs["SMILES"].astype(str)
    ad.obs[GENE_PERT] = ad.obs["disease_gene_x"].astype(str)
    ad.obs[CELL_TYPE] = "HUVEC"

    ad.obs[BATCH] = "TREK_B_unfiltered_" + ad.obs[BATCH].astype(str)

    ad.obs.loc[ad.obs["is_control"], MOL_PERT] = BASE_LABEL
    ad.obs[MOL_PERT] = ad.obs[MOL_PERT].astype("category")
    ad.obs[BASE] = ad.obs["is_control"].astype(int).astype("category")
    ad.obs[CONTROL] = 0
    ad.obs[CONTROL] = ad.obs[CONTROL].astype(int).astype("category")
    ad.obs[BATCH] = ad.obs[BATCH].astype("category")

    ad.obs[MOL_PERT_NAME] = ad.obs[MOL_PERT]

    ad.var[GENE_NAME] = list(ad.var_names)
    ad.var_names = list(ad.var[GENE_NAME])
    ad.var_names_make_unique()

    # Reservations
    with open("/rxrx/data/valence/internal_benchmarking/vcds1/v1_reservation.json", "r") as f:
        reservation = json.load(f)

    with open("/rxrx/data/valence/internal_benchmarking/vcds1/precautionary_exclusion.json", "r") as f:
        exclusion1 = json.load(f)

    with open("/rxrx/data/valence/internal_benchmarking/vcds1/precautionary_exclusion2.json", "r") as f:
        exclusion2 = json.load(f)

    indicator_rec_id = ad.obs["rec_id"].isin(reservation["rec_ids"])
    indicator_rsv = ad.obs["experiment_label"].isin(reservation["plated_rxrx_labels"])
    indicator_exclusion1 = ad.obs["experiment_label"].isin(exclusion1)
    indicator_exclusion2 = ad.obs["experiment_label"].isin(exclusion2)

    # Additional filter
    indicator_center_guide = ad.obs["center_guide"].astype(bool)

    filtered_indicator = indicator_rec_id | indicator_rsv | indicator_exclusion1 | indicator_exclusion2 | indicator_center_guide

    filtered_ad = ad[~filtered_indicator]
    
    # Filter for frequent covariates
    all_conditions = filtered_ad.obs[MOL_PERT].unique().tolist()
    all_conditions.remove("")

    indicator_reduce_conditions = ~ad.obs[MOL_PERT].isin(all_conditions)
    
    filtered_and_reduced_indicator = indicator_rec_id | indicator_rsv | indicator_exclusion1 | indicator_exclusion2 | indicator_center_guide | indicator_reduce_conditions
    ad = ad[~filtered_and_reduced_indicator].to_memory()

    # Standardize smiles
    ad.obs[MOL_PERT] = ad.obs[MOL_PERT].apply(partial(standardize_smiles, skip_list=SKIP_LIST))

    return ad


def _TREK_VCB_formatter(ad: sc.AnnData) -> sc.AnnData:
    """
    Formatting for TREK_VCB datasets.
    """
    ad.obs[BATCH] = "TREK_VCB_" + ad.obs[BATCH].astype(str)

    ad.obs[MOL_PERT] = ad.obs[MOL_PERT].astype(str)
    ad.obs.loc[ad.obs["is_control"], MOL_PERT] = BASE_LABEL
    ad.obs[MOL_PERT] = ad.obs[MOL_PERT].astype("category")
    ad.obs[CELL_TYPE] = ad.obs["context"].astype("category")
    ad.obs[GENE_PERT] = "EMPTY"
    ad.obs[BASE] = ad.obs["is_control"].astype(int).astype("category")
    ad.obs[CONTROL] = 0
    ad.obs[CONTROL] = ad.obs[CONTROL].astype(int).astype("category")
    ad.obs[BATCH] = ad.obs[BATCH].astype("category")

    ad.obs[MOL_PERT_NAME] = ad.obs[MOL_PERT]

    ad.var_names_make_unique()

    # Standardize smiles
    ad.obs[MOL_PERT] = ad.obs[MOL_PERT].apply(partial(standardize_smiles, skip_list=SKIP_LIST))

    return ad


def _K562_formatter(ad: sc.AnnData) -> sc.AnnData:
    """
    Temporary formatting for TREKseq (subset B) data until unifying data specs.
    """
    ad = ad.to_memory()
    
    ad.obs[GENE_PERT] = ad.obs[GENE_NAME].astype(str)
    ad.obs[CELL_TYPE] = "K562"
    ad.obs[MOL_PERT] = EMPTY_LABEL
    ad.obs[DOSE] = EMPTY_LABEL

    ad.obs[BATCH] = ad.obs[BATCH].astype(str)
    ad.obs[BATCH] = "K562_" + ad.obs[BATCH]

    ad.obs.loc[ad.obs[CONTROL], MOL_PERT] = CONTROL_LABEL
    ad.obs[MOL_PERT] = ad.obs[MOL_PERT].astype("category")
    ad.obs[CELL_TYPE] = ad.obs[CELL_TYPE].astype("category")
    ad.obs[BATCH] = ad.obs[BATCH].astype("category")
    ad.obs[CONTROL] = ad.obs[CONTROL].astype(int).astype("category")
    ad.obs[BASE] = 0
    ad.obs[BASE] = ad.obs[BASE].astype(int).astype("category")

    ad.obs[MOL_PERT_NAME] = ad.obs[MOL_PERT]

    ad.var_names = list(ad.var[GENE_NAME])
    ad.var_names_make_unique()

    return ad


def _Tahoe_STATE_formatter(ad: sc.AnnData, src: str) -> sc.AnnData:
    """
    Temporary formatting for Tahoe data until unifying data specs.
    """
    plate = int(src.split("_")[-1])
    
    condition_names, mols, doses, base_states = [], [], [], []

    for p_d in ad.obs["drugname_drugconc"].values.tolist():
        p_d = eval(p_d)[0]
        p, d, _ = p_d

        if p.endswith(" "):
            p = p[:-1]

        condition_names.append(p_d)
        mols.append(p)
        doses.append(d * 1_000)
        base_states.append(True if p == "DMSO_TF" else False)
    
    ad.obs[MOL_PERT_NAME] = condition_names
    ad.obs[MOL_PERT] = mols
    ad.obs[DOSE] = doses
    ad.obs[BATCH] = f"{src}_{str(plate)}"
    ad.obs[BASE] = base_states

    ad.obs[MOL_PERT_NAME] = ad.obs[MOL_PERT_NAME].astype(str)
    ad.obs.loc[ad.obs[BASE], MOL_PERT_NAME] = BASE_LABEL
    ad.obs[MOL_PERT_NAME] = ad.obs[MOL_PERT_NAME].astype("category")

    ad.obs[CELL_TYPE] = ad.obs["cell_name"].astype("category")
    ad.obs[BASE] = ad.obs[BASE].astype(int).astype("category")
    ad.obs[CONTROL] = 0
    ad.obs[CONTROL] = ad.obs[CONTROL].astype(int).astype("category")
    ad.obs[BATCH] = ad.obs[BATCH].astype("category")

    ad.var_names_make_unique()

    # Convert to smiles
    drug_metadata = datasets.load_dataset("vevotx/Tahoe-100M","drug_metadata", split="train").to_pandas()
    drug2smiles = {drug: smiles for drug, smiles in zip(drug_metadata["drug"].tolist(), drug_metadata["canonical_smiles"].tolist())}
    drug2smiles["CTRL"] = "CTRL"
    drug2smiles["BASE"] = "BASE"
    drug2smiles["Sacubitril/Valsartan"] = "Sacubitril/Valsartan"
    drug2smiles["Verteporfin"] = "Verteporfin"
    drug2smiles["Fumaric Acid"] = "Fumaric acid"
    ad.obs[MOL_PERT] = ad.obs[MOL_PERT].map(drug2smiles)

    # Standardize smiles
    unique_mols = ad.obs[MOL_PERT].unique()
    std_smiles_map = {mol: standardize_smiles(mol, skip_list= SKIP_LIST + ["Sacubitril/Valsartan", "Verteporfin", "Fumaric Acid"]) for mol in unique_mols}
    ad.obs[MOL_PERT] = ad.obs[MOL_PERT].map(std_smiles_map)

    return ad


def save_adata_to_h5ad(ad: sc.AnnData, cache_file: str | Path, obsm_key: str = None) -> None:
    """
    This function saves the AnnData object as uncompressed dense h5ad file

    Args:
        ad: AnnData object
        cache_file: Path to the h5ad file
    """
    logger.info(f"Converting to dense format and saving to {cache_file}")

    # Convert X to dense float32 in one operation
    if obsm_key is not None:
        if sparse.issparse(ad.obsm[obsm_key]):
            X_dense = ad.obsm[obsm_key].toarray().astype(np.float32)
        else:
            X_dense = ad.obsm[obsm_key].astype(np.float32)

        n_features = X_dense.shape[1]
        var_df = pd.DataFrame(index=[f"feature_{i}" for i in range(n_features)])
    
    else:
        if sparse.issparse(ad.X):
            X_dense = ad.X.toarray().astype(np.float32)
        else:
            X_dense = ad.X.astype(np.float32)

        var_df = ad.var.copy()
    
    # Create new AnnData with converted X
    ad_new = sc.AnnData(
        X=X_dense,
        obs=ad.obs.copy(),
        var=var_df,
        uns=ad.uns.copy() if ad.uns else {},
        obsm={},
        varm=ad.varm.copy() if ad.varm else {},
        obsp=ad.obsp.copy() if ad.obsp else {},
        varp=ad.varp.copy() if ad.varp else {}
    )
    
    del X_dense
    gc.collect()
    
    # Write uncompressed h5ad file
    ad_new.write_h5ad(cache_file, compression=None)
    
    del ad_new
    gc.collect()
    
    logger.info(f"Saved dense h5ad file: {cache_file}")