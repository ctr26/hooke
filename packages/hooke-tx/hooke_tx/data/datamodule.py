from __future__ import annotations

from typing import Any

import os
from tqdm import tqdm
import json
import pandas as pd
import pyarrow.parquet as pq

import numpy as np
import torch
import lightning.pytorch as pl

from loguru import logger

from torch.utils.data import DataLoader

from hooke_tx.data.constants import DATA_SOURCES
from hooke_tx.data.constants import ASSAY, CELL_TYPE, EXPERIMENT, WELL, PERTURBATIONS, GENE_PERT, EFFECT_CLASS, MOL_PERT, DOSE, ROUTING, EMPTY, GENE_ONLY, MOL_ONLY
from hooke_tx.data.dataset import TaskDataset


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate list of dataset items into batch dict with x0, x1, covariates for the model."""
    x0 = torch.from_numpy(np.stack([b["base_expression"] for b in batch])).float()
    x1 = torch.from_numpy(np.stack([b["target_expression"] for b in batch])).float()
    metadatas = [b["metadata"] for b in batch]
    covariates = {
        ROUTING: [b["routing"] for b in batch],
        ASSAY: [m[ASSAY] for m in metadatas],
        CELL_TYPE: [m[CELL_TYPE] for m in metadatas],
        EXPERIMENT: [m[EXPERIMENT] for m in metadatas],
        WELL: [m[WELL] for m in metadatas],
        GENE_PERT: [m[GENE_PERT] for m in metadatas],
        MOL_PERT: [m[MOL_PERT] for m in metadatas],
    }
    return {"x0": x0, "x1": x1, "covariates": covariates, "metadata": metadatas}


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_args: dict[str, Any],
        task_args: dict[str, Any],
        batch_size_train: int = 512,
        batch_size_eval: int = 8,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.metadata_dict = {}

        self.data_sources = self.data_args.pop("data_sources")
        self.cache_dir = self.data_args.pop("cache_dir")

        self.task_datasets = {
            "fit": None,
            "validate": None,
            "test": None,
            "predict": None,
        }

        self.setup_flage = {
            "fit": False,
            "validate": False,
            "test": False,
            "predict": False,
        }

    def prepare_data(self, splits_to_load: list[str] | None = None) -> None:
        """
        Prepare data for task.
        """
        stages = ["fit", "validate", "test", "predict"] if splits_to_load is None else splits_to_load

        for split in stages:
            # Check if dataset is already loaded
            if self.task_datasets[split] is not None:
                continue
            
            split_hash = self.create_split_hash(split, self.data_sources, self.cache_dir)
            logger.info(f"Hash for {split}: {split_hash}")

            if os.path.exists(f"{self.cache_dir}/cached_datasets/{split_hash}"):
                logger.info(f"Loading cached {split} dataset...")
                dataset = torch.load(f"{self.cache_dir}/cached_datasets/{split_hash}", weights_only=False)

                self.task_datasets[split] = dataset
                self.metadata_dict = dataset.metadata_dict
                self.var_dict = dataset.var_dict
                self.X_path_dict = dataset.X_path_dict
            else:
                logger.info(f"Processing {split} data...")

                split_data_sources = self.data_sources.get(split)
                
                if split_data_sources is None:
                    continue

                if isinstance(split_data_sources, str):
                    split_data_sources = [split_data_sources]

                self.metadata_dict, self.var_dict, self.X_path_dict = {}, {}, {}
                for src in split_data_sources:
                    src_dir = DATA_SOURCES.get(src)

                    self.metadata_dict[src] = pq.read_table(f"{src_dir}/obs.parquet")
                    self.var_dict[src] = pd.read_parquet(f"{src_dir}/var.parquet")
                    self.X_path_dict[src] = f"{src_dir}/X.zarr"

                self.task_datasets[split] = TaskDataset(
                    metadata_dict=self.metadata_dict,
                    var_dict=self.var_dict,
                    X_path_dict=self.X_path_dict,
                )

                torch.save(self.task_datasets[split], f"{self.cache_dir}/cached_datasets/{split_hash}")

    def create_split_hash(
        self,
        split: str,
        data_sources: dict,
        cache_dir: str
    ) -> str:
        """
        Create a unique hash for the data.
        """
        split_data_sources = data_sources.get(split)
        if isinstance(split_data_sources, str):
            split_data_sources = [split_data_sources]
        if split_data_sources is None:
            split_data_sources = []
        return f"{':'.join(split_data_sources)}"

    def setup(self, stage="fit"):
        if stage == "fit":
            stages_to_setup = ["fit", "validate"]
        else:
            stages_to_setup = [stage]

        splits_path = self.task_args.get("splits_path")
        selected_ensembl_gene_ids_path = self.task_args.get("selected_ensembl_gene_ids_path")
        with open(selected_ensembl_gene_ids_path) as f:
            self.selected_ensembl_gene_ids = [line.strip() for line in f if line.strip()]

        for this_stage in stages_to_setup:
            # Check if dataset is already setup
            if self.setup_flage.get(this_stage):
                continue
            
            if self.task_datasets.get(this_stage) is None:
                raise ValueError(f"Dataset for {this_stage} is not loaded")
            
            if splits_path is not None:
                with open(splits_path) as f:
                    split_indices = json.load(f)[this_stage]
            else:
                split_indices = []

            self.task_datasets[this_stage].setup(
                split_indices_dict=split_indices,
                selected_ensembl_gene_ids=self.selected_ensembl_gene_ids,
                composition_args=self.task_args.get("composition"),
                routing_args=self.task_args.get("routing"),
                **self.data_args,
            )
            self.setup_flage[this_stage] = True

    def gather_covariates(self):
        covariates_dict = {
            ASSAY: [],
            CELL_TYPE: [],
            EXPERIMENT: [],
            WELL: [],
            GENE_PERT: [],
            EFFECT_CLASS: [],
            MOL_PERT: [],
            DOSE: [],
            ROUTING: [],
        }

        for metadata in self.metadata_dict.values():
            md = metadata.to_pandas() if hasattr(metadata, "to_pandas") else metadata
            covariates_dict[ASSAY].extend(md[ASSAY].unique().tolist())
            covariates_dict[CELL_TYPE].extend(md[CELL_TYPE].unique().tolist())
            covariates_dict[EXPERIMENT].extend(md[EXPERIMENT].unique().tolist())
            covariates_dict[WELL].extend(md[WELL].unique().tolist())

            for ps in tqdm(md[PERTURBATIONS], desc="Gathering covariates..."):
                ps = list(ps)
                for p in ps:
                    if p[0] == "genetic":
                        covariates_dict[GENE_PERT].append(p[1])
                        covariates_dict[EFFECT_CLASS].append(p[2])
                    elif p[0] == "compound":
                        covariates_dict[MOL_PERT].append(p[1])
                        covariates_dict[DOSE].append(p[2])

        # Make lists unique
        for k, v in covariates_dict.items():
            covariates_dict[k] = sorted(list(set(v)))

        # TODO: Currently supports a total of 2 perturbations
        possible_target_states = [
            "genetic",
            "compound",
            "genetic_compound",
            "genetic_genetic",
            "compound_compound",
        ]

        covariates_dict[ROUTING] = (
            [f"noise:{s}" for s in possible_target_states]
            + [f"{EMPTY}:{s}" for s in possible_target_states]
            + [f"genetic:{s}" for s in possible_target_states[2:]]
            + [f"compound:{s}" for s in possible_target_states[-1:]]
        ) + [f"genetic:{s}" for s in possible_target_states[2:]] + [f"compound:{s}" for s in possible_target_states[-1:]]
        
        return covariates_dict

    def data_dim(self):
        return len(self.selected_ensembl_gene_ids)

    def train_dataloader(self):
        return DataLoader(
            self.task_datasets["fit"],
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=_collate_batch,
        ) if self.task_datasets["fit"] is not None else None

    def val_dataloader(self):
        return DataLoader(
            self.task_datasets["validate"],
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=_collate_batch,
        ) if self.task_datasets["validate"] is not None else None

    def test_dataloader(self):
        return DataLoader(
            self.task_datasets["test"],
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=_collate_batch,
        ) if self.task_datasets["test"] is not None else None

    def predict_dataloader(self):
        return DataLoader(
            self.task_datasets["predict"],
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=_collate_batch,
        ) if self.task_datasets["predict"] is not None else None
