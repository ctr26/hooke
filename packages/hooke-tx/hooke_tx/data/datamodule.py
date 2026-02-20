from __future__ import annotations

from typing import Any

import os
import hashlib
from loguru import logger

import json
import pandas as pd

import numpy as np
import torch
import lightning.pytorch as pl

from torch.utils.data import DataLoader

from hooke_tx.data.constants import DATA_SOURCES
from hooke_tx.data.constants import ASSAY, CELL_TYPE, EXPERIMENT, WELL, GENE_PERT, EFFECT_CLASS, MOL_PERT, DOSE, ROUTING, EMPTY, GENE_ONLY, MOL_ONLY
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
    return {"x0": x0, "x1": x1, "covariates": covariates}


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_args: dict[str, Any],
        task_args: dict[str, Any],
        trainer_args: dict[str, Any],
    ) -> None:
        super().__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.trainer_args = trainer_args
        self.metadata_dict = {}

        self.task_datasets = {
            "fit": None,
            "validate": None,
            "test": None,
            "predict": None,
        }

    def prepare_data(self, splits_to_load: list[str] | None = None) -> None:
        """
        Prepare data for task.
        """
        stages = ["fit", "validate", "test"] if splits_to_load is None else splits_to_load

        data_sources = self.data_args.get("data_sources")
        cache_dir = self.data_args.get("cache_dir")

        for split in stages:
            split_hash = self.create_split_hash(split, data_sources, cache_dir)
            logger.info(f"Hash for {split}: {split_hash}")

            if os.path.exists(f"{cache_dir}/cached_datasets/task_{split_hash}"):
                logger.info(f"Loading cached {split} dataset...")
                dataset = torch.load(f"{cache_dir}/cached_datasets/task_{split_hash}")

                self.task_datasets[split] = dataset
                md = getattr(dataset, "metadata_dict", None)
                if md is not None:
                    self.metadata_dict = {**getattr(self, "metadata_dict", {}), **md}
            else:
                logger.info(f"Processing {split} data...")

                split_data_sources = data_sources.get(split)
                
                if split_data_sources is None:
                    continue

                if isinstance(split_data_sources, str):
                    split_data_sources = [split_data_sources]

                self.metadata_dict, self.var_dict, self.X_path_dict = {}, {}, {}
                for src in split_data_sources:
                    src_dir = DATA_SOURCES.get(src)

                    self.metadata_dict[src] = pd.read_parquet(f"{src_dir}/metadata.parquet")
                    self.var_dict[src] = pd.read_parquet(f"{src_dir}/var.parquet")
                    self.X_path_dict[src] = f"{src_dir}/X.zarr"

                self.task_datasets[split] = TaskDataset(
                    metadata_dict=self.metadata_dict,
                    var_dict=self.var_dict,
                    X_path_dict=self.X_path_dict,
                )

                torch.save(self.task_datasets[split], f"{cache_dir}/cached_datasets/task_{split_hash}")

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
        return f"{split}_{cache_dir}_{'_'.join(split_data_sources)}"

    def setup(self, stage="training"):
        if stage == "training":
            stages_to_setup = ["fit", "validate"]

        splits_path = self.task_args.get("splits_path")
        selected_genes_path = self.task_args.get("selected_genes_path")
        self.selected_genes = json.load(open(selected_genes_path))

        for this_stage in stages_to_setup:
            if self.task_datasets.get(this_stage) is None:
                continue
            split_indices = (
                json.load(open(splits_path))[this_stage] if splits_path is not None else []
            )
            self.task_datasets[this_stage].setup(
                split_indices_dict=split_indices,
                selected_genes=self.selected_genes,
                composition_args=self.task_args.get("composition"),
                routing_args=self.task_args.get("routing"),
                **self.data_args,
            )

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
            covariates_dict[ASSAY].extend(metadata[ASSAY].unique().tolist())
            covariates_dict[CELL_TYPE].extend(metadata[CELL_TYPE].unique().tolist())
            covariates_dict[EXPERIMENT].extend(metadata[EXPERIMENT].unique().tolist())
            covariates_dict[WELL].extend(metadata[WELL].unique().tolist())

            for ps in metadata[GENE_PERT].unique().tolist():
                for p in ps:
                    if p["type"] == "genetic":
                        covariates_dict[GENE_PERT].append(p["ensembl_gene_id"])
                        covariates_dict[EFFECT_CLASS].append(p["effect_class"])
                    elif p["type"] == "compound":
                        covariates_dict[MOL_PERT].append(p["smiles"])
                        covariates_dict[DOSE].append(p["concentration"])

        # Make lists unique
        for k, v in covariates_dict.items():
            covariates_dict[k] = sorted(list(set(v)))

        possible_target_states = [
            "genetic",
            "compound",
            "genetic_compound",
            "genetic_genetic",
            "compound_compound",
        ]

        covariates_dict[ROUTING] = (
            [f"{EMPTY}:{s}" for s in possible_target_states]
            + [f"{GENE_ONLY}:{s}" for s in possible_target_states[2:]]
            + [f"{MOL_ONLY}:{s}" for s in possible_target_states[-1:]]
        )
        
        return covariates_dict

    def data_dim(self):
        return len(self.selected_genes)

    def train_dataloader(self):
        return DataLoader(
            self.task_datasets["fit"],
            batch_size=self.trainer_args.get("batch_size_train", 512),
            num_workers=self.trainer_args.get("num_workers", 0),
            shuffle=True,
            collate_fn=_collate_batch,
        ) if self.task_datasets["fit"] is not None else None

    def val_dataloader(self):
        return DataLoader(
            self.task_datasets["validate"],
            batch_size=self.trainer_args.get("batch_size_eval", 8),
            num_workers=self.trainer_args.get("num_workers", 0),
            shuffle=False,
            collate_fn=_collate_batch,
        ) if self.task_datasets["validate"] is not None else None

    def test_dataloader(self):
        return DataLoader(
            self.task_datasets["test"],
            batch_size=self.trainer_args.get("batch_size_eval", 8),
            num_workers=self.trainer_args.get("num_workers", 0),
            shuffle=False,
            collate_fn=_collate_batch,
        ) if self.task_datasets["test"] is not None else None

    def predict_dataloader(self):
        return DataLoader(
            self.task_datasets["predict"],
            batch_size=self.trainer_args.get("batch_size_eval", 8),
            num_workers=self.trainer_args.get("num_workers", 0),
            shuffle=False,
            collate_fn=_collate_batch,
        ) if self.task_datasets["predict"] is not None else None
