import logging
import os
import pathlib
import re
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Literal

import ornamentalist
import submitit
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from hooke_forge.data import dataset
from hooke_forge.model.drifting import JointDrifting
from hooke_forge.model.flow_matching import JointFlowMatching, get_model
from hooke_forge.model.mean_flow import JointMeanFlow
from hooke_forge.model.tokenizer import DataFrameTokenizer
from hooke_forge.training.state import TrainState
from hooke_forge.training.trainer import train
from hooke_forge.utils.distributed import Distributed
from hooke_forge.utils.ema import KarrasEMA
from hooke_forge.utils.name_run import generate_random_name

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_tokenizer_from_checkpoint(ckpt_dir: str) -> DataFrameTokenizer | None:
    """Load the tokenizer from the latest checkpoint in the directory.

    This is useful for finetuning on a subset of data while preserving
    the original vocabulary from pretraining.

    Returns None if no checkpoint is found.
    """
    pattern = r"step_(\d+).ckpt"
    if not os.path.exists(ckpt_dir):
        return None
    fnames = [f.name for f in os.scandir(ckpt_dir) if f.is_file() and re.fullmatch(pattern, f.name)]
    if len(fnames) == 0:
        return None

    latest = max(fnames, key=lambda x: int(re.fullmatch(pattern, x).group(1)))  # type: ignore
    path = os.path.join(ckpt_dir, latest)
    log.info(f"Loading tokenizer from checkpoint: {path}")
    state = torch.load(path, weights_only=False, map_location="cpu")
    if "tokenizer" not in state:
        log.warning("Checkpoint does not contain tokenizer, will fit new one")
        return None
    return DataFrameTokenizer.from_state_dict(state["tokenizer"])


@ornamentalist.configure()
def prng(rank: int, seed: int = ornamentalist.Configurable[42]):
    local_seed = seed + rank
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    log.info(f"Rank {rank} setting seed to {local_seed}")


@ornamentalist.configure(name="ckpt")
def get_resume_checkpoint_dir(
    resume_from: str = ornamentalist.Configurable[""],
    strict: bool = ornamentalist.Configurable[True],
) -> tuple[str | None, bool]:
    """Get the checkpoint directory to resume from.

    Args:
        resume_from: Path to a checkpoint directory or a specific .ckpt file.
                     If a directory, will look for checkpoints in that directory.
                     If a .ckpt file, will use that file's parent directory.
                     If empty string, returns None (no resume).
        strict:      If False, allows loading a checkpoint with mismatched keys.
                     Set to False when finetuning on a different modality
                     (e.g. loading a Px checkpoint for Tx finetuning).

    Example usage::

        python main.py --ckpt.resume_from=/path/to/checkpoints
        python main.py --ckpt.resume_from=/path/to/step_100000.ckpt --ckpt.strict=False
    """
    ckpt_dir: str | None
    if not resume_from:
        ckpt_dir = None
    elif resume_from.endswith(".ckpt"):
        ckpt_dir = os.path.dirname(resume_from)
    else:
        ckpt_dir = resume_from
    return ckpt_dir, strict


def main(config: ornamentalist.ConfigDict):
    ornamentalist.setup(config, force=True)
    with Distributed() as D:
        job_env = submitit.JobEnvironment()

        cwd = pathlib.Path.cwd()
        output_dir = str(cwd.parent / job_env.job_id)
        name = generate_random_name(output_dir)

        if D.rank == 0:
            try:
                subprocess.run(
                    [
                        "scontrol",
                        "update",
                        f"JobId={job_env.job_id}",
                        f"JobName={name}",
                    ],
                    check=True,
                    capture_output=True,
                )
                log.info(f"Successfully renamed SLURM job to '{name}'")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                log.warning(f"Could not rename SLURM job to '{name}'. Reason: {e}")

        if D.rank == 0:
            wandb.init(
                name=name,
                id=name,
                dir=output_dir,
                notes=f"Outputs saved to: {output_dir}",
                project="big-img",
                entity="valencelabs",
                resume="allow",
                save_code=False,
                config=config,
            )

        log.info(f"Running job ID {job_env.job_id}")
        log.info(f"{output_dir=}")
        log.info(f"{name=}")
        log.info(f"{config=}")

        prng(D.rank)
        torch.set_float32_matmul_precision("medium")

        # Checkpoint directories:
        # - ckpt_dir: where new checkpoints will be saved (job's own directory)
        # - resume_ckpt_dir: where to load existing checkpoints from (can be different)
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        resume_ckpt_dir, strict_ckpt = get_resume_checkpoint_dir()
        if resume_ckpt_dir:
            log.info(f"Will resume from checkpoint directory: {resume_ckpt_dir}")
            log.info(f"Checkpoint loading strict={strict_ckpt}")
        else:
            # If no explicit resume path, try to resume from the job's own checkpoint dir
            resume_ckpt_dir = ckpt_dir

        # Load tokenizer from checkpoint if resuming, to preserve vocabulary
        existing_tokenizer = load_tokenizer_from_checkpoint(resume_ckpt_dir)

        # ------------------------------------------------------------------
        # Build model (dispatches on --flow_model.modality)
        # ------------------------------------------------------------------
        net: JointFlowMatching | JointDrifting = get_model()
        # Infer the training modality from which vector fields were created
        active_modalities = list(net.vector_fields.keys())
        has_px = "px" in active_modalities
        has_tx = "tx" in active_modalities
        if has_px and has_tx:
            modality = "joint"
        elif has_px:
            modality = "px"
        else:
            modality = "tx"
        log.info(f"Training modality: {modality} (active vector fields: {active_modalities})")

        # ------------------------------------------------------------------
        # Build dataloaders
        # ------------------------------------------------------------------
        px_train_loader = px_val_loader = px_test_loaders = None
        tx_train_loader = tx_val_loader = None

        if has_px:
            px_train_loader, px_val_loader, _vocab, tokenizer = dataset.get_dataloaders(tokenizer=existing_tokenizer)
            px_test_loaders = {"iid": px_val_loader}
        else:
            tokenizer = existing_tokenizer  # may be None

        if has_tx:
            # Reuse the Px tokenizer if available (joint or finetune from Px)
            tx_train_loader, tx_val_loader, tokenizer = dataset.get_tx_dataloaders(tokenizer=tokenizer)

        assert tokenizer is not None, (
            "No tokenizer available. Either provide training data or resume from a checkpoint."
        )

        # ------------------------------------------------------------------
        # Set up training state
        # ------------------------------------------------------------------
        net.to(D.device)
        # MeanFlow uses forward-mode AD (JVP) which is incompatible with torch.compile
        if not isinstance(net, JointMeanFlow):
            net = torch.compile(net)  # type: ignore
        ddp = DDP(net)
        ema = KarrasEMA(net)
        opt = torch.optim.Adam(
            net.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),  # from appendix of arxiv.org/abs/2411.03177v1
            eps=1e-8,
            weight_decay=0.0,
        )
        state = TrainState(ddp=ddp, ema=ema, opt=opt, global_step=0, tokenizer=tokenizer)

        nparams = sum(p.numel() for p in net.parameters())
        log.info(f"Model has {nparams / 1e6:.0f}M parameters")
        log.info(f"Loading checkpoints from: {resume_ckpt_dir}")
        log.info(f"Saving checkpoints to: {ckpt_dir}")
        state.load_latest_ckpt(dir=resume_ckpt_dir, device=D.device, strict=strict_ckpt)

        train(
            state=state,
            output_dir=output_dir,
            D=D,
            modality=modality,
            px_train_loader=px_train_loader,
            px_val_loader=px_val_loader,
            px_test_loaders=px_test_loaders,
            tx_train_loader=tx_train_loader,
            tx_val_loader=tx_val_loader,
        )

        if D.rank == 0:
            wandb.finish(quiet=True)


def prepare_submission(config: ornamentalist.ConfigDict) -> Callable:
    # closure captures the config -- equivalant to partial(main, config=config)
    def submission():
        return main(config)

    # attach a checkpoint method so submitit knows that it should auto-requeue on timeout
    def checkpoint(*args, **kwargs):
        return submitit.helpers.DelayedSubmission(submission)

    setattr(submission, "checkpoint", checkpoint)
    return submission


@ornamentalist.configure()
def launcher(
    configs: list[dict],
    nodes: int = ornamentalist.Configurable[1],
    gpus: int = ornamentalist.Configurable[1],
    cpus: int = ornamentalist.Configurable[24],
    ram: int = ornamentalist.Configurable[128],
    timeout: int = ornamentalist.Configurable[1440 * 4],
    partition: str = ornamentalist.Configurable["hopper"],
    qos: str = ornamentalist.Configurable["hooke-predict"],
    output_dir: str = ornamentalist.Configurable["./outputs/"],
    cluster: Literal["debug", "local", "slurm"] = ornamentalist.Configurable["debug"],
    desc: str = ornamentalist.Configurable[""],
):
    """Thin wrapper that launches the main function with submitit.
    If multiple configs are provided, they will be launched as an array job/sweep.
    Note that all jobs in the array will be launched with the same SLURM parameters."""

    del desc  # desc is just some free text we can use to filter with wandb in the UI

    output_dir = os.path.join(output_dir, f"{time.time():.0f}")
    executor = submitit.AutoExecutor(folder=output_dir, cluster=cluster)
    executor.update_parameters(
        slurm_partition=partition,
        slurm_qos=qos,
        nodes=nodes,
        tasks_per_node=gpus,  # set ntasks = ngpus
        gpus_per_node=gpus,
        cpus_per_task=cpus,
        slurm_mem_per_gpu=f"{ram}G",
        timeout_min=timeout,
        stderr_to_stdout=True,
        slurm_signal_delay_s=120,
        slurm_wckey="hooke-predict",
        slurm_additional_parameters={
            "requeue": True,
            "exclude": "hop08,hop61,hop62",  # Exclude nodes with MIG or small GPU slices
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

        # if local or debug, wait for job to finish, otherwise exit script as soon as job is submitted
        if cluster == "local":
            log.info("Running job(s) locally using multiprocessing...")
            log.info(f"stdout and stderr for each process are logged to {output_dir}.")
            log.info("The job is in another process so you won't see anything here.")
            log.info("(But ctrl-c will still kill the job.)")
            _ = [j.results()[0] for j in jobs]

        elif cluster == "debug":
            log.info("Running job(s) in this process in debug mode...")
            log.info("pdb will open automatically on crash.")
            log.info("It's best to only use 1 GPU in this mode.")
            _ = [j.results()[0] for j in jobs]


def cli():
    configs = ornamentalist.cli()
    assert all(config["launcher"] == configs[0]["launcher"] for config in configs)
    ornamentalist.setup(configs[0], force=True)
    launcher(configs=configs)


if __name__ == "__main__":
    cli()
