"""Evaluation script for big-img checkpoints.

Runs evaluation on saved checkpoints using the evaluation functions from trainer.py,
with model configuration parsed from launch_cmd.txt.

Usage (via SLURM):
    python eval.py \
        --eval.checkpoints_dir /path/to/outputs/1768305605/12583183/checkpoints \
        --eval.output_dir /path/to/eval_output \
        --eval.num_checkpoints 3 \
        --launcher.cluster slurm

Usage (local debug):
    python eval.py \
        --eval.checkpoints_dir /path/to/checkpoints \
        --launcher.cluster debug
"""

import logging
import os
import pathlib
import re
import sys
import time
from pathlib import Path
from typing import Callable, Literal

import ornamentalist
import submitit
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from hooke_forge.data import dataset
from hooke_forge.model.tokenizer import DataFrameTokenizer, MetaDataConfig
from hooke_forge.data.dataset import CellPaintConverter
from hooke_forge.model.architecture import get_model_cls
from hooke_forge.evaluation.px_metrics import (
    compute_phenomics_metrics,
    evaluate_px,
    visualise_phenomics,
)
from hooke_forge.training.state import set_metrics_log_path
from hooke_forge.utils.distributed import Distributed
from hooke_forge.utils.ema import KarrasEMA
from hooke_forge.utils.encoders import StabilityCPEncoder

_meta_defaults = MetaDataConfig()
REC_ID_DIM = _meta_defaults.rec_id_dim
CONCENTRATION_DIM = _meta_defaults.concentration_dim
CELL_TYPE_DIM = _meta_defaults.cell_type_dim
ASSAY_TYPE_DIM = _meta_defaults.assay_type_dim
EXPERIMENT_DIM = _meta_defaults.experiment_dim
WELL_ADDRESS_DIM = _meta_defaults.well_address_dim

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Model hyperparameter keys to extract from launch_cmd.txt
MODEL_CONFIG_KEYS = ["model.name"]


def _list_checkpoints(dir_path: str) -> list[str]:
    """List all checkpoints in a directory, sorted by step number."""
    pattern = r"step_(\d+).ckpt"
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {dir_path}")
    files = [
        f.name
        for f in os.scandir(dir_path)
        if f.is_file() and re.fullmatch(pattern, f.name)
    ]
    if len(files) == 0:
        raise FileNotFoundError(f"No checkpoints matching 'step_*.ckpt' in {dir_path}")
    # Sort by step number
    files.sort(key=lambda n: int(re.fullmatch(pattern, n).group(1)))  # type: ignore
    return [os.path.join(dir_path, f) for f in files]


def parse_model_config_from_checkpoint(checkpoint_path: str) -> dict | None:
    """Parse model hyperparameters from launch_cmd.txt in the checkpoint's grandparent directory.

    Expected structure:
        outputs/[jobnumber]/[id]/checkpoints/step_*.ckpt
        outputs/[jobnumber]/launch_cmd.txt

    Returns dict with model config or None if not found.
    """
    ckpt_path = Path(checkpoint_path)

    # Navigate: checkpoints/ -> [id]/ -> [jobnumber]/ -> launch_cmd.txt
    launch_cmd_path = ckpt_path.parent.parent.parent / "launch_cmd.txt"

    if not launch_cmd_path.exists():
        log.warning(f"launch_cmd.txt not found at {launch_cmd_path}")
        return None

    log.info(f"Found launch_cmd.txt at {launch_cmd_path}")
    launch_cmd = launch_cmd_path.read_text().strip()
    log.info(f"Launch command: {launch_cmd}")

    config = {}
    for key in MODEL_CONFIG_KEYS:
        pattern = rf"--{re.escape(key)}\s+(\S+)"
        match = re.search(pattern, launch_cmd)
        if match:
            value = match.group(1)
            short_key = key.replace("model.", "")
            if value.lower() == "true":
                config[short_key] = True
            elif value.lower() == "false":
                config[short_key] = False
            else:
                config[short_key] = value

    if config:
        log.info(f"Parsed model config: {config}")
        return config

    log.info("No model config flags found in launch_cmd.txt, using defaults")
    return None


def strip_orig_mod_prefix(state_dict: dict) -> dict:
    """Strip _orig_mod prefix from state dict keys (added by torch.compile)."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("._orig_mod", "")
        new_state_dict[new_key] = value
    return new_state_dict


def load_model_and_tokenizer(
    checkpoint_path: str, device: torch.device
) -> tuple[torch.nn.Module, KarrasEMA, DataFrameTokenizer]:
    """Load model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.

    Returns:
        Tuple of (model, ema, tokenizer)
    """
    state = torch.load(checkpoint_path, weights_only=False, map_location=device)

    # Restore tokenizer from checkpoint
    tokenizer = DataFrameTokenizer.from_state_dict(state["tokenizer"])

    # Create model with correct vocab sizes
    model_cls = get_model_cls()
    net = model_cls(
        input_size=32,
        in_channels=8,
        learn_sigma=False,
        rec_id_dim=REC_ID_DIM,
        concentration_dim=CONCENTRATION_DIM,
        cell_type_dim=CELL_TYPE_DIM,
        experiment_dim=EXPERIMENT_DIM,
        assay_type_dim=ASSAY_TYPE_DIM,
        well_address_dim=WELL_ADDRESS_DIM,
    )
    net.to(device)

    # Load EMA weights (strip _orig_mod prefix if checkpoint was saved with torch.compile)
    ema = KarrasEMA(net)
    ema_state = strip_orig_mod_prefix(state["ema"])
    ema.load_state_dict(ema_state)

    return net, ema, tokenizer


@ornamentalist.configure(name="eval")
def run_eval_on_checkpoints(
    checkpoints_dir: str = ornamentalist.Configurable[""],
    output_dir: str = ornamentalist.Configurable["./eval_outputs"],
    data_path: str = ornamentalist.Configurable[
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/joint-model/metadata/pretraining_v6.parquet"
    ],
    num_checkpoints: int = ornamentalist.Configurable[3],
    do_visualise: bool = ornamentalist.Configurable[True],
    do_evaluate: bool = ornamentalist.Configurable[True],
    do_metrics: bool = ornamentalist.Configurable[True],
    batch_size: int = ornamentalist.Configurable[64],
    num_workers: int = ornamentalist.Configurable[16],
    wandb_project: str | None = ornamentalist.Configurable[None],
) -> None:
    """Run evaluations on checkpoints from a trained model.

    Args:
        checkpoints_dir: Path to directory containing step_*.ckpt files
        output_dir: Directory to save evaluation outputs
        data_path: Path to parquet file with evaluation data
        num_checkpoints: Number of checkpoints to evaluate (evenly spaced)
        do_visualise: Whether to generate sample visualizations
        do_evaluate: Whether to compute validation loss
        do_metrics: Whether to compute FD/PRDC metrics
        batch_size: Batch size for evaluation
        num_workers: Number of dataloader workers
        wandb_project: Optional wandb project name for logging
    """
    if not checkpoints_dir:
        raise ValueError("checkpoints_dir is required")

    # Parse model config from launch_cmd.txt
    ckpts = _list_checkpoints(checkpoints_dir)
    model_config = parse_model_config_from_checkpoint(ckpts[0])

    # Setup ornamentalist with model config if found
    if model_config:
        current_config = ornamentalist.get_config()
        current_config["model"] = model_config
        ornamentalist.setup(current_config, force=True)

    # Initialize distributed context
    with Distributed() as D:
        # Setup wandb logging
        use_wandb = wandb_project is not None
        if use_wandb and D.rank == 0:
            run_name = Path(checkpoints_dir).parent.name
            wandb.init(
                project=wandb_project,
                name=f"eval_{run_name}",
                config=ornamentalist.get_config(),
            )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Setup file-based metrics logging if wandb is not enabled
        if not use_wandb and D.rank == 0:
            metrics_path = os.path.join(output_dir, "metrics.jsonl")
            set_metrics_log_path(metrics_path)

        # Select checkpoints to evaluate (evenly spaced)
        if num_checkpoints < len(ckpts):
            step = len(ckpts) // num_checkpoints
            idx = list(range(len(ckpts) - 1, -1, -step))[:num_checkpoints]
            ckpts = [ckpts[i] for i in sorted(idx)]

        log.info(f"Evaluating {len(ckpts)} checkpoints")
        log.info(f"Checkpoints: {ckpts}")

        # Load first checkpoint to get tokenizer for dataloader
        first_ckpt_state = torch.load(ckpts[0], weights_only=False, map_location="cpu")
        tokenizer = DataFrameTokenizer.from_state_dict(first_ckpt_state["tokenizer"])

        # Get dataloaders with existing tokenizer
        train_loader, val_loader, vocab, _ = dataset.get_dataloaders(
            path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            tokenizer=tokenizer,
        )

        # Initialize VAE and color converter (disable compile to avoid dynamo issues with sample())
        vae = StabilityCPEncoder(device=D.device, compile=False)
        cp2rgb = CellPaintConverter(device=D.device)

        # Evaluate each checkpoint
        for ckpt_path in ckpts:
            log.info(f"Loading checkpoint: {ckpt_path}")

            # Load model and EMA from checkpoint
            net, ema, tokenizer = load_model_and_tokenizer(ckpt_path, D.device)
            net = torch.compile(net)  # type: ignore

            # Wrap in DDP for compatibility with trainer functions
            ddp = DDP(net)

            # Create a minimal state object for trainer functions
            # We need to extract global_step from checkpoint
            state_dict = torch.load(
                ckpt_path, weights_only=False, map_location=D.device
            )
            global_step = state_dict["global_step"]

            # Create a simple namespace to hold state
            class EvalState:
                def __init__(self, ddp, ema, global_step, tokenizer):
                    self.ddp = ddp
                    self.ema = ema
                    self.global_step = global_step
                    self.tokenizer = tokenizer

            state = EvalState(
                ddp=ddp, ema=ema, global_step=global_step, tokenizer=tokenizer
            )

            # Visualization (EMA)
            if do_visualise:
                log.info(f"Running visualisation for step {global_step}")
                visualise_phenomics(
                    state=state,
                    loader=val_loader,
                    output_dir=output_dir,
                    vae=vae,
                    cp2rgb=cp2rgb,
                    D=D,
                    use_ema=True,
                )

            # Evaluate loss (EMA and DDP)
            if do_evaluate:
                log.info(f"Running evaluation for step {global_step}")
                evaluate_px(
                    state=state,
                    loader=val_loader,
                    vae=vae,
                    D=D,
                    use_ema=True,
                )
                evaluate_px(
                    state=state,
                    loader=val_loader,
                    vae=vae,
                    D=D,
                    use_ema=False,
                )

            # Compute metrics (EMA)
            if do_metrics:
                log.info(f"Computing metrics for step {global_step}")
                compute_phenomics_metrics(
                    state=state,
                    name="val",
                    loader=val_loader,
                    vae=vae,
                    cp2rgb=cp2rgb,
                    D=D,
                    use_ema=True,
                )

            D.barrier()

        if use_wandb and D.rank == 0:
            wandb.finish()

        log.info("Evaluation complete!")


def main(config: ornamentalist.ConfigDict):
    """Main entry point called by submitit."""
    ornamentalist.setup(config, force=True)
    run_eval_on_checkpoints()


def prepare_submission(config: ornamentalist.ConfigDict) -> Callable:
    """Prepare a submission closure for submitit."""

    def submission():
        return main(config)

    def checkpoint(*args, **kwargs):
        return submitit.helpers.DelayedSubmission(submission)

    setattr(submission, "checkpoint", checkpoint)
    return submission


@ornamentalist.configure(name="eval_launcher")
def eval_launcher(
    configs: list[dict],
    nodes: int = ornamentalist.Configurable[1],
    gpus: int = ornamentalist.Configurable[1],
    cpus: int = ornamentalist.Configurable[24],
    ram: int = ornamentalist.Configurable[128],
    timeout: int = ornamentalist.Configurable[480],
    partition: str = ornamentalist.Configurable["hopper"],
    qos: str = ornamentalist.Configurable["hooke-predict"],
    output_dir: str = ornamentalist.Configurable["./eval_outputs/"],
    cluster: Literal["debug", "local", "slurm"] = ornamentalist.Configurable["slurm"],
):
    """Launch the evaluation job with submitit.

    Args:
        configs: List of config dicts (from ornamentalist.cli())
        nodes: Number of nodes
        gpus: Number of GPUs per node
        cpus: CPUs per task
        ram: RAM per GPU in GB
        timeout: Timeout in minutes
        partition: SLURM partition
        qos: SLURM QoS
        output_dir: Base output directory for submitit logs
        cluster: Execution mode (debug, local, slurm)
    """
    output_dir = os.path.join(output_dir, f"{time.time():.0f}")
    executor = submitit.AutoExecutor(folder=output_dir, cluster=cluster)
    executor.update_parameters(
        slurm_partition=partition,
        slurm_qos=qos,
        nodes=nodes,
        tasks_per_node=gpus,
        gpus_per_node=gpus,
        cpus_per_task=cpus,
        slurm_mem_per_gpu=f"{ram}G",
        timeout_min=timeout,
        stderr_to_stdout=True,
        slurm_signal_delay_s=120,
        slurm_wckey="hooke-predict",
        slurm_additional_parameters={
            "requeue": True,
            "exclude": "hop08,hop61,hop62",
        },
    )

    os.makedirs(output_dir, exist_ok=True)
    launch_cmd = f"{sys.executable} {' '.join(sys.argv)}"
    with open(os.path.join(output_dir, "launch_cmd.txt"), "w") as f:
        f.write(launch_cmd)

    snapshot_dir = os.path.join(output_dir, "snapshot")
    with submitit.helpers.RsyncSnapshot(pathlib.Path(snapshot_dir)):
        fns = [prepare_submission(config=config) for config in configs]
        jobs = executor.submit_array(fns)
        log.info(f"Submitted {jobs=}")

        if cluster == "local":
            log.info("Running job(s) locally using multiprocessing...")
            log.info(f"stdout and stderr for each process are logged to {output_dir}.")
            _ = [j.results()[0] for j in jobs]

        elif cluster == "debug":
            log.info("Running job(s) in this process in debug mode...")
            log.info("pdb will open automatically on crash.")
            _ = [j.results()[0] for j in jobs]


def cli():
    configs = ornamentalist.cli()
    assert all(
        config["eval_launcher"] == configs[0]["eval_launcher"] for config in configs
    )
    ornamentalist.setup(configs[0], force=True)
    eval_launcher(configs=configs)


if __name__ == "__main__":
    cli()
