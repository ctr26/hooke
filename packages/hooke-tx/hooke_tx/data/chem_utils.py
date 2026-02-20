import json
import torch
import pandas as pd
import numpy as np
import datamol as dm
from tqdm import tqdm
from loguru import logger

from rdkit.Chem import MolFromSmiles, MolToInchiKey, SaltRemover, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_mol(
    mol,
    disconnect_metals: bool = False,
    normalize: bool = True,
    reionize: bool = True,
    strip_salts: bool = True,
):
    """
    This function returns a standardized version the given molecule. It relies on the
    RDKit rdMolStandardize module which is largely inspired from MolVS.
    """
    if isinstance(mol, str):
        mol = MolFromSmiles(mol)

    if disconnect_metals:
        md = rdMolStandardize.MetalDisconnector()
        mol = md.Disconnect(mol)

    if normalize:
        mol = rdMolStandardize.Normalize(mol)

    if reionize:
        reionizer = rdMolStandardize.Reionizer()
        mol = reionizer.reionize(mol)

    if strip_salts:
        remover = SaltRemover.SaltRemover(defnData=None)
        mol = remover.StripMol(mol, sanitize=True)
    return mol


def standardize_smiles(smiles: str, skip_list: list[str] = [], kekulize: bool = True, ordered: bool = True) -> str | None:
    """Standardize SMILES using datamol with intelligent salt removal."""
    if smiles in skip_list:
        return smiles

    if not smiles:
        return None
    try:
        mol = dm.to_mol(smiles)
        mol = standardize_mol(mol)
        return dm.to_smiles(mol, kekulize=kekulize, ordered=ordered) if mol else None
    except Exception as e:
        import logging

        logging.debug(f"Failed to standardize SMILES '{smiles}': {e}")
        return None
    
    
def compute_inchikey(smiles: str, standardize: bool = True) -> str:
    """Compute InChIKey from SMILES.
    Args:
        smiles: SMILES string
        standardize: Whether to standardize the molecule
    Returns:
        InChIKey string
    """
    mol = (
        standardize_mol(smiles)
        if standardize
        else MolFromSmiles(smiles)
    )
    return MolToInchiKey(mol)


def compute_ecfp_embeddings(condition_list: list[str], embedding_dim: int = 1024, radius: int = 2) -> torch.Tensor:
    """
    Compute ECFP fingerprints from SMILES strings and create embedding matrix.
    
    Args:
        condition_list: List of SMILES strings
        embedding_dim: Dimension of the output embeddings (will be used as fingerprint size)
        radius: ECFP radius (default: 2 for ECFP4)
    
    Returns:
        torch.Tensor: Embedding matrix of shape (len(condition_list), embedding_dim)
    """
    # Veryfy conditions are passed as SMILES strings
    mol_list = [
        MolFromSmiles(p) for p in condition_list
    ]

    assert None not in mol_list, "Invalid SMILES strings in condition list"
    
    smiles_list = condition_list
    
    conditions, fingerprints = [], []
    for smiles in smiles_list:
        try:
            mol = MolFromSmiles(smiles)
            if mol is None:
                # Handle invalid SMILES by creating zero vector
                raise ValueError(f"Invalid SMILES: {smiles}")
            else:
                # Compute ECFP fingerprint as bit vector using MorganGenerator
                mgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=embedding_dim)
                fp = mgen.GetFingerprint(mol)
                fp = np.array(fp, dtype=np.float32)
            fingerprints.append(fp)
            conditions.append(smiles)
        except:
            logger.warning(f"Invalid smiles {smiles} removed from conditions")

    for p in ["CTRL", "NEG_CTRL", "UNKOWN"]:
        conditions.append(p)
        fingerprints.append(np.zeros(embedding_dim, dtype=np.float32))
    
    return conditions, embedding_dim, torch.tensor(np.array(fingerprints), dtype=torch.float32)


def retrieve_molgps_embeddings(condition_list: list[str], emb_names: list[str], cache_dir: str = "/rxrx/data/valence/pef/molgps") -> torch.Tensor:
    """
    Retrieve ECFP fingerprints from SMILES strings and create embedding matrix.
    
    Args:
        condition_list: List of condition smiles strings
        embedding_dim: Dimension of the output embeddings (will be used as fingerprint size)
        radius: ECFP radius (default: 2 for ECFP4)
    
    Returns:
        torch.Tensor: Embedding matrix of shape (len(condition_list), embedding_dim)
    """
    cached_embeddings = torch.load(f"{cache_dir}/embeddings.pt")
    embedding_dims = json.load(open(f"{cache_dir}/shapes.json"))
    index_map = pd.read_parquet(f"{cache_dir}/index_map.parquet")

    conditions = []
    embeddings_dict = {emb_name: [] for emb_name in emb_names}
    dim_dict = {emb_name: embedding_dims[emb_name] for emb_name in emb_names}

    conditions = []
    for smiles in tqdm(condition_list, desc="Retrieving molgps embeddings"):
        try:
            inchikey = compute_inchikey(smiles)
            idx = index_map["inchikey"].values.tolist().index(inchikey)

            for emb_name in emb_names:
                embeddings_dict[emb_name].append(cached_embeddings[emb_name][idx])

            conditions.append(smiles)
        except:
            logger.warning(f"Invalid smiles {smiles} removed from conditions")
            # Add smiles to missing_smiles.txt
            with open("/mnt/ps/home/CORP/frederik.wenkel/projects/pef/missing_smiles.txt", "a") as f:
                f.write(smiles + "\n")

    for emb_name in emb_names:
        embeddings_dict[emb_name] = torch.tensor(np.array(embeddings_dict[emb_name]), dtype=torch.float32)

    embeddings = torch.cat(list(embeddings_dict.values()), dim=-1)

    for p in ["CTRL", "NEG_CTRL", "UNKOWN"]:
        conditions.append(p)

    embeddings = torch.cat([embeddings, torch.zeros(3, sum(dim_dict.values()), dtype=torch.float32)], dim=0)
        
    return conditions, sum(dim_dict.values()), embeddings


def retrieve_px_pretrained_embeddings(condition_name_list: list[str], cache_dir: str = "/rxrx/data/valence/pef/px_pretrained") -> torch.Tensor:
    pretrained_model = torch.load(f"{cache_dir}/model.ckpt", map_location=torch.device('cpu'))

    name2rec = json.load(open(f"{cache_dir}/name2rec.json"))
    rec2idx = pretrained_model['tokenizer']["rec_id"]["token_to_id"]

    rec_embeddings = pretrained_model["ema"]["module._orig_mod.meta_adaptor.rec_id_embedder.embedding_table.weight"]

    embeddings = []

    for p_name in tqdm(condition_name_list, desc="Retrieving px_pretrained embeddings"):
        rec_id = name2rec[p_name]
        idx = rec2idx[rec_id]

        embeddings.append(rec_embeddings[idx])

    for p in ["CTRL", "NEG_CTRL", "UNKOWN"]:
        condition_name_list.append(p)
        embeddings.append(np.zeros(rec_embeddings.shape[-1], dtype=np.float32))
        
    embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)

    return condition_name_list, rec_embeddings.shape[-1], embeddings