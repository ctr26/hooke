from copy import deepcopy

import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from hooke_tx.data.datamodule import DataModule
from hooke_tx.trainer import TrainerModule


@hydra.main(config_path="configs/template", config_name="example", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.constants.seed)

    data_args = deepcopy(OmegaConf.to_container(cfg.data, resolve=True))
    task_args = deepcopy(OmegaConf.to_container(cfg.task, resolve=True))
    model_args = deepcopy(OmegaConf.to_container(cfg.model, resolve=True))
    trainer_args = deepcopy(OmegaConf.to_container(cfg.trainer, resolve=True))
    accelerator_args = deepcopy(OmegaConf.to_container(cfg.accelerator, resolve=True))

    datamodule = DataModule(
        data_args=data_args,
        task_args=task_args,
        trainer_args=trainer_args,
    )
    datamodule.prepare_data()
    datamodule.setup()

    callbacks = [ModelCheckpoint(save_last=True)]
    logger = WandbLogger(
        project=cfg.wandb.get("project", "Hooke-Tx"),
        entity=cfg.wandb.get("entity", "valencelabs"),
    )

    trainer = pl.Trainer(
        max_epochs=trainer_args.get("max_epochs"),
        callbacks=callbacks,
        logger=logger,
        **accelerator_args
    )
    
    model = TrainerModule(
        covariates=datamodule.gather_covariates(),
        data_dim=datamodule.data_dim(),
        model_args=model_args,
        **trainer_args
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
