import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import Callable, Literal

import ornamentalist
import submitit
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

import dataset
from model import get_model_cls
from trainer import TrainState, train
from utils.distributed import Distributed
from utils.ema import KarrasEMA
from utils.name_run import generate_random_name

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@ornamentalist.configure()
def prng(rank: int, seed: int = ornamentalist.Configurable[42]):
    local_seed = seed + rank
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    log.info(f"Rank {rank} setting seed to {local_seed}")


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

        model_cls = get_model_cls()
        net = model_cls(
            input_size=32,
            in_channels=8,
            learn_sigma=False,
            y_dim=dataset.NUM_PERTURBATIONS,
            e_dim=dataset.NUM_EXPERIMENTS,
            c_dim=dataset.NUM_CELL_TYPES,
        )
        net.to(D.device)
        net: torch.nn.Module = torch.compile(net)  # type: ignore
        ddp = DDP(net)
        ema = KarrasEMA(net)
        opt = torch.optim.Adam(
            net.parameters(),
            lr=1e-4,  # scheduler will override lr
            betas=(0.9, 0.95),  # from appendix of arxiv.org/abs/2411.03177v1
            eps=1e-8,
            weight_decay=0.0,
        )
        state = TrainState(ddp=ddp, ema=ema, opt=opt, global_step=0)

        nparams = sum(p.numel() for p in net.parameters())
        log.info(f"Model has {nparams / 1e6:.0f}M parameters")

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Using checkpoint directory: {ckpt_dir}")
        state.load_latest_ckpt(dir=ckpt_dir, device=D.device)

        train_loader, val_loader = dataset.get_dataloaders()
        train(
            state=state,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loaders={"val_iid": val_loader},
            output_dir=output_dir,
            D=D,
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
    cpus: int = ornamentalist.Configurable[16],
    ram: int = ornamentalist.Configurable[64],
    timeout: int = ornamentalist.Configurable[1440],
    partition: str = ornamentalist.Configurable["hopper"],
    qos: str = ornamentalist.Configurable["normal"],
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
        exclude="hop02",  # hop02 has issues
        stderr_to_stdout=True,
        slurm_signal_delay_s=120,
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


if __name__ == "__main__":
    configs = ornamentalist.cli()
    assert all(config["launcher"] == configs[0]["launcher"] for config in configs)
    ornamentalist.setup(configs[0], force=True)
    launcher(configs=configs)
