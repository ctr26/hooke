from typing import Any

from tqdm import tqdm
from loguru import logger

import zarr
import polars as pl
import numpy as np
import random
from torch.utils.data import Dataset

from hooke_tx.data.constants import PERTURBATIONS, GENE_PERT, MOL_PERT, CONTEXT, CELL_TYPE, EXPERIMENT, WELL, EMPTY, NEG_CONTROL, POS_CONTROL, GENE_ONLY, MOL_ONLY, MULTI_PERT


class TaskDataset(Dataset):
    """
    Base PyTorch Dataset for a specific single-cell task.

    Args:
        adatas: A dictionary of AnnData objects.
        data_axis_indices: A dictionary of data axis indices.
        gene_indices: A dictionary of gene indices (per data source).
        data_path: The path to the h5ad files.
        select_strategy: Strategy for gene selection ('hvg', 'top', 'all', or 'pca').
        sample_base_states: How to sample the base states.
        target_sum: The target sum for normalization.
        log1p: Whether to apply log1p transformation.
    """
    def __init__(
        self,
        metadata_dict: dict[str, pl.DataFrame],
        var_dict: dict[str, list[int]],
        X_path_dict: dict[str, str],
    ):
        self.X_path_dict = X_path_dict
        self.var_dict = var_dict

        # Transform metadata into list[dict] for faster access
        self.metadata_dict = metadata_dict
        self.metadata_map = {
            src: metadata.to_pylist() for src, metadata in metadata_dict.items()
        }

        self.base_state_indices_cache = {
            CONTEXT: {},
            EMPTY: {},
            GENE_ONLY: {},
            MOL_ONLY: {},
        }
        self.base_state_cache = {
            CONTEXT: {},
            EMPTY: {},
            GENE_ONLY: {},
            MOL_ONLY: {},
        }
        self.indices_cache = {src: {} for src in metadata_dict}
        self._gene_pert_indices = {}
        self._mol_pert_indices = {}
        self._build_pert_indices()

        self.cache_base_state_indices()

    def __len__(self):
        return len(self.source_index_map)
        
    def __getitem__(self, idx: int) -> dict:
        """
        Returns a sample from the dataset at the given index.
        """
        src, src_idx = self.source_index_map[idx]
        
        metadata = self.metadata_map[src][src_idx]
        target_expression = self.X_dict[src][src_idx]

        # Subset genes
        target_expression = target_expression[self.gene_index_map[src]]
        
        # Transform expression
        target_expression = self.transform(target_expression)
        
        # Embed expression
        target_embedding = self.embed(target_expression) if self.embedder is not None else np.nan
        
        base_expression, base_embedding, routing = self.retrieve_base_state(src, metadata)

        return {
            "base_expression": base_expression,
            "base_embedding": base_embedding,
            "target_expression": target_expression,
            "target_embedding": target_embedding,
            "metadata": metadata,
            "routing": routing,
        }

    def retrieve_base_state(self, src: str, metadata: dict[str, Any]) -> np.ndarray:
        """
        Matches a base state to a given metadata row.
        """
        target_state_id = "_".join([p[0] for p in metadata[PERTURBATIONS]])
        
        if self.sample_base_states == "noise":
            return np.nan, np.nan, f"noise:{target_state_id}"

        # Sample routing
        base_state_type, base_state_hash, base_state_id = self.sample_routing(src, metadata)
        routing = f"{base_state_id}:{target_state_id}"

        if base_state_hash in self.base_state_cache[base_state_type].keys():
            selected_indices = self.base_state_cache[base_state_type][base_state_hash]
        else:
            c, e = metadata[CELL_TYPE], metadata[EXPERIMENT]
            
            # Find indices of cells that share cell type, experiment and well
            if (c, e) in self.indices_cache[src]:
                shared_indices = self.indices_cache[src][(c, e)]
            else:
                context_mask = (np.atleast_1d(self.metadata_dict[src][CELL_TYPE]) == c) & (np.atleast_1d(self.metadata_dict[src][EXPERIMENT]) == e)
                shared_indices = np.flatnonzero(context_mask)
                self.indices_cache[src][(c, e)] = shared_indices

            if base_state_type == EMPTY:
                empty_indices = np.flatnonzero(np.atleast_1d(self.metadata_dict[src][EMPTY]))
                selected_indices = np.intersect1d(shared_indices, empty_indices)
            elif base_state_type == GENE_ONLY:
                gene_s = random.choice(metadata[GENE_PERT])
                gene_key = (gene_s[1], gene_s[2])
                gene_indices = self._gene_pert_indices[src].get(gene_key, np.array([], dtype=np.intp))
                selected_indices = np.intersect1d(shared_indices, gene_indices)
            elif base_state_type == MOL_ONLY:
                mol_s = random.choice(metadata[MOL_PERT])
                mol_key = (mol_s[1], mol_s[2])
                mol_indices = self._mol_pert_indices[src].get(mol_key, np.array([], dtype=np.intp))
                selected_indices = np.intersect1d(shared_indices, mol_indices)

            self.base_state_cache[base_state_type][base_state_hash] = selected_indices

        if self.sample_base_states == "random":
            idx = random.choice(selected_indices)
            base_expression = self.X_dict[src][idx]
            
            # Subset genes
            base_expression = base_expression[self.gene_index_map[src]]

            # Transform expression
            base_expression = self.transform(base_expression)

            # Embed expression
            base_embedding = self.embed(base_expression) if self.embedder is not None else np.nan

            return base_expression, base_embedding, routing
        else:
            raise ValueError(f"Unsupported way of sampling base states: {self.sample_base_states}")

    def sample_routing(self, src: str, metadata: dict[str, Any]) -> str:
        """
        Samples how to reteive the base state.
        """
        base_state_hash = (src, metadata[CELL_TYPE], metadata[EXPERIMENT], metadata[WELL])
        base_state_id = EMPTY

        if metadata[NEG_CONTROL] or metadata[POS_CONTROL] or metadata[GENE_ONLY] or metadata[MOL_ONLY]:
            base_state_type = EMPTY
            
            return base_state_type, base_state_hash, base_state_id
        elif metadata[MULTI_PERT]:
            if metadata[GENE_ONLY]:
                p_from_empty, p_from_gene = self.routing_args["gene_gene"][EMPTY], self.routing_args["gene_gene"][GENE_ONLY]
                
                base_state_type = np.random.choice([EMPTY, GENE_ONLY], p=[p_from_empty, p_from_gene])

                if base_state_type == GENE_ONLY:
                    base_state_gene = random.choice(metadata[GENE_PERT])
                    base_state_hash = base_state_hash + (base_state_gene[1], base_state_gene[2])
                    base_state_id = "genetic"
            elif metadata[MOL_ONLY]:
                p_from_empty, p_from_mol = self.routing_args["mol_mol"][EMPTY], self.routing_args["mol_mol"][MOL_ONLY]
                
                base_state_type = np.random.choice([EMPTY, MOL_ONLY], p=[p_from_empty, p_from_mol])

                if base_state_type == MOL_ONLY:
                    base_state_mol = random.choice(metadata[MOL_PERT])
                    base_state_hash = base_state_hash + (base_state_mol[1], base_state_mol[2],)
                    base_state_id = "compound"
            else:
                p_from_empty, p_from_gene = self.routing_args["gene_mol"][EMPTY], self.routing_args["gene_mol"][GENE_ONLY]

                base_state_type = np.random.choice([EMPTY, GENE_ONLY], p=[p_from_empty, p_from_gene])
                
                if base_state_type == GENE_ONLY:
                    base_state_gene = random.choice(metadata[GENE_PERT])
                    base_state_hash = base_state_hash + (base_state_gene[1], base_state_gene[2])
                    base_state_id = "genetic"

            return base_state_type, base_state_hash, base_state_id
        else:
            raise ValueError(f"Unexpected metadata: {metadata}")

    def transform(self, x: np.ndarray) -> np.ndarray:
        """s
        Applies target sum and log1p transformation.
        """
        if x.sum() == 0:
            return x
        if self.target_sum is not None:
            x = (x / x.sum()) * np.float32(self.target_sum)
        if self.log1p:
            x = np.log1p(x)
        
        return x

    def embed(self, x: np.ndarray) -> np.ndarray:
        # TODO...
        raise NotImplementedError

    def _pert_keys(self, val: Any, key_indices: tuple[int, ...]) -> list[tuple]:
        """Return list of hashable keys for a GENE_PERT or MOL_PERT value (tuple or list of tuples)."""
        if val is None:
            return []
        if isinstance(val, tuple):
            return [tuple(val[i] for i in key_indices)]
        return [tuple(t[i] for i in key_indices) for t in val]

    def _build_pert_indices(self) -> None:
        """Precompute gene_pert and mol_pert key -> row indices per source for O(1) lookup."""
        gene_key_ix = (1, 2)
        mol_key_ix = (1, 2)
        for src, rows in self.metadata_map.items():
            gene_map: dict[tuple, list[int]] = {}
            mol_map: dict[tuple, list[int]] = {}
            for i, row in enumerate(rows):
                for k in self._pert_keys(row.get(GENE_PERT), gene_key_ix):
                    gene_map.setdefault(k, []).append(i)
                for k in self._pert_keys(row.get(MOL_PERT), mol_key_ix):
                    mol_map.setdefault(k, []).append(i)
            self._gene_pert_indices[src] = {k: np.array(v) for k, v in gene_map.items()}
            self._mol_pert_indices[src] = {k: np.array(v) for k, v in mol_map.items()}

    def cache_base_state_indices(self) -> None:
        pass

    def setup(
        self,
        split_indices_dict: dict[str, list[int]],
        selected_ensembl_gene_ids: list[str],
        composition_args: dict[str, bool],
        routing_args: dict[str, dict[str, float]],
        sample_base_states: str = "random",
        target_sum: int = 300_000,
        log1p: bool = True,
        embedder: str = None,
    ) -> None:
        self.sample_base_states = sample_base_states
        self.target_sum = target_sum
        self.log1p = log1p
        self.embedder = embedder
        self.routing_args = routing_args

        # Load X matrices
        self.X_dict = {
            src: zarr.open(self.X_path_dict[src], mode="r") for src in self.X_path_dict.keys()
        }

        # Map from item indices to source-index pairs
        self.source_index_map = []
        use_split_filter = isinstance(split_indices_dict, dict) and any(
            len(v) > 0 for v in split_indices_dict.values()
        )
        for src, metadata_list in tqdm(self.metadata_map.items(), desc="Building metadata map..."):
            for idx, metadata in enumerate(metadata_list):
                # filter based on desired composition
                if any([not composition_args[key] and metadata[key] for key in composition_args.keys()]):
                    continue

                if use_split_filter and src in split_indices_dict and len(split_indices_dict[src]) > 0:
                    if idx not in split_indices_dict[src]:
                        continue

                self.source_index_map.append((src, idx))

        # Identify ensembl gene id indices
        self.gene_index_map = {}
        for src, src_var in self.var_dict.items():
            ensembl_gene_id_col = src_var["ensembl_gene_id"]
            ensembl_gene_id_list = ensembl_gene_id_col.to_list() if hasattr(ensembl_gene_id_col, "to_list") else ensembl_gene_id_col.tolist()
            assert all(g in ensembl_gene_id_list for g in selected_ensembl_gene_ids), f"Missing ensembl gene ids in {src}"
            self.gene_index_map[src] = np.array([ensembl_gene_id_list.index(g) for g in selected_ensembl_gene_ids])