import numpy as np
import datamol as dm

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


def compute_ecfp(smiles: str, embedding_dim: int = 1024, radius: int = 2) -> np.ndarray | None:
    """Compute ECFP fingerprint for a single SMILES. Returns None if invalid."""
    try:
        mol = MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        mgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=embedding_dim)
        fp = mgen.GetFingerprint(mol)
        
        return np.array(fp, dtype=np.float32)
    
    except Exception:
        return None