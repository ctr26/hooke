from copy import deepcopy
import os

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from hooke_tx.data.datamodule import DataModule
from hooke_tx.trainer import TxPredictor
from hooke_tx.callbacks import create_callbacks


@hydra.main(config_path="configs/templates/trek_drugscreen", config_name="cfg", version_base=None)
def main(cfg: DictConfig) -> None:
    data_args = deepcopy(OmegaConf.to_container(cfg.data, resolve=True))
    task_args = deepcopy(OmegaConf.to_container(cfg.task, resolve=True))

    model_args = deepcopy(OmegaConf.to_container(cfg.model, resolve=True))
    trainer_args = deepcopy(OmegaConf.to_container(cfg.trainer, resolve=True))
    eval_args = deepcopy(OmegaConf.to_container(cfg.eval, resolve=True))
    metric_args = deepcopy(OmegaConf.to_container(cfg.metrics, resolve=True))
    compute_spec = deepcopy(OmegaConf.to_container(cfg.compute, resolve=True))
    checkpoint_args = deepcopy(OmegaConf.to_container(cfg.checkpoint, resolve=True))

    # Translate relative paths to absolute paths
    _root = os.path.dirname(os.path.abspath(__file__))
    for _key in ("selected_ensembl_gene_ids_path", "splits_path"):
        _path = task_args.get(_key)
        if _path and not os.path.isabs(_path):
            task_args[_key] = os.path.normpath(os.path.join(_root, _path))
    
    # Set seed
    pl.seed_everything(cfg.constants.seed)

    # Avoid cuBLAS init issues on A100 (CUBLAS_STATUS_NOT_INITIALIZED)
    torch.set_float32_matmul_precision("medium")

    # Initialize datamodule
    datamodule = DataModule(
        data_args=data_args,
        task_args=task_args,
        batch_size_train=trainer_args.pop("batch_size_train", 32),
        batch_size_eval=trainer_args.pop("batch_size_eval", 8),
        num_workers=trainer_args.pop("num_workers", 0),
    )
    datamodule.prepare_data()
    datamodule.setup()


    callbacks, logging_kwargs = create_callbacks(
        metric_args, trainer_args, eval_args, checkpoint_args,
        data_args=data_args, task_args=task_args,
    )

    log_dir = f"/rxrx/data/user/{os.getenv('USER')}/outgoing/hooke-tx"
    logger = WandbLogger(
        project=cfg.wandb.get("project", "Hooke-Tx"),
        entity=cfg.wandb.get("entity", "valencelabs"),
        name=cfg.wandb.get("name", None),
        save_dir=log_dir,
        dir=log_dir,
    )

    trainer = pl.Trainer(
        max_epochs=trainer_args.get("max_epochs"),
        callbacks=callbacks,
        logger=logger,
        **logging_kwargs,
        **compute_spec
    )

    model = TxPredictor(
        covariates=datamodule.gather_covariates(),
        data_dim=datamodule.data_dim(),
        model_args=model_args,
        **trainer_args
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
