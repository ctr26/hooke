from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MFUConfig(BaseModel):
    """
    Configuration for the Model Flops Utilization (MFU) Callback.
    """

    use_mfu: bool = True
    available_flops: float = 60e12
    use_backward: bool = True
    logging_interval: int = 10
    window_size: int = 2


class WandbConfig(BaseModel):
    """Weight and Bias (WandB) configuration."""

    wandb_track: bool = False
    entity: str | None = None
    project: str | None = None
    local_wandb_dir: Path | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_local_wandb_dir(self) -> "WandbConfig":
        if not self.wandb_track:
            return self
        if self.entity is None:
            raise ValueError("entity is required when wandb_track is True")
        if self.project is None:
            raise ValueError("project is required when wandb_track is True")
        if self.local_wandb_dir is None:
            self.local_wandb_dir = Path.cwd() / "wandb_logs"
        return self


class TransformerBackboneConfig(BaseModel):
    """Configuration for the transformer backbone."""

    key: Literal["GPT2"] = "GPT2"
    max_position_embeddings: int
    n_positions: int
    hidden_size: int
    n_embd: int
    n_layer: int
    n_head: int
    resid_pdrop: float
    embd_pdrop: float
    attn_pdrop: float
    use_cache: bool


class ModelConfig(BaseModel):
    """
    Configuration for the model.
    """

    # How do these two things differ?
    checkpoint: Path | None = None
    init_from: Path | None = None

    device: Literal["cuda", "cpu"] = "cuda"

    # ===== Unsure =====
    # how many cells to group together into a single set of cells
    cell_set_len: int = 512

    # configurable buffer for confidence/special tokens
    extra_tokens: int = 1

    # ===== Architecture =====
    decoder_hidden_dims: list[int] = [1024, 1024, 512]
    hidden_dim: int = 328
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    transformer_backbone: TransformerBackboneConfig = Field(
        default_factory=TransformerBackboneConfig
    )

    # ===== Flags =====
    batch_encoder: bool = False
    nb_decoder: bool = False
    mask_attn: bool = False
    predict_residual: bool = True
    freeze_pert_backbone: bool = False
    finetune_vci_decoder: bool = False
    residual_decoder: bool = False
    confidence_token: bool = False
    use_basal_projection: bool = False

    # ===== Loss =====
    blur: float = 0.05
    loss: str = "energy"
    decoder_loss_weight: float = 1.0
    distributional_loss: str = "energy"
    regularization: float = 0.0


class TrainingConfig(BaseModel):
    """
    Configuration for the training of the transition baseline.
    """

    weight_decay: float = 0.0005
    batch_size: int = 16
    lr: float = 1e-4
    max_steps: int = 40000
    train_seed: int = 42
    val_freq: 2000
    ckpt_every_n_steps: int = 2000
    gradient_clip_val: int = 10  # 0 means no clipping
    loss_fn: str = "mse"
    devices: int = 1  # Number of GPUs to use for training
    strategy: str = "auto"  # DDP strategy for multi-GPU training
    mfu: MFUConfig = Field(default_factory=MFUConfig)


class Config(BaseModel):
    """
    Configuration for the transition baseline.
    """

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
