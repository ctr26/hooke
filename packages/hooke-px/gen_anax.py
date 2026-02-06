import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import Callable, Literal

import numpy as np
import ornamentalist
import polars as pl
import submitit
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import dataset
from model import get_model_cls
from trainer import TrainState, generate
from utils.distributed import Distributed
from utils.ema import KarrasEMA
from utils.evaluation import DINOv2Detector, Phenom2Detector
from utils.name_run import generate_random_name

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@ornamentalist.configure()
def prng(rank: int, seed: int = ornamentalist.Configurable[42]):
    local_seed = seed + rank
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    log.info(f"Rank {rank} setting seed to {local_seed}")


@ornamentalist.configure(name="ckpt")
def load_ckpt(
    state: TrainState, D: Distributed, path: str = ornamentalist.Configurable
):
    log.info(f"Loading checkpoint from {path}")
    state.load_ckpt(path=path, device=D.device)


@torch.inference_mode()
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
            lr=1e-4,
            betas=(0.9, 0.95),  # from appendix of arxiv.org/abs/2411.03177v1
            eps=1e-8,
            weight_decay=0.0,
        )
        state = TrainState(ddp=ddp, ema=ema, opt=opt, global_step=0)
        nparams = sum(p.numel() for p in net.parameters())
        log.info(f"Model has {nparams / 1e6:.0f}M parameters")
        load_ckpt(state, D)
        vae = dataset.StabilityCPEncoder(device=D.device)
        phenom = Phenom2Detector(device=D.device)
        dino = DINOv2Detector(device=D.device)
        cp2rgb = dataset.CellPaintConverter(device=D.device)

        df = pl.read_parquet(
            "/mnt/ps/home/CORP/charlie.jones/project/big-img/simulated_data/anax_rxrx3_subset_with_CORRECT_conditioning.parquet"
        )
        ds = dataset.CellDataset(df, train=False, size=256, multiscale=False)
        sampler = torch.utils.data.DistributedSampler(
            ds, num_replicas=D.world_size, rank=D.rank, shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=32,
            num_workers=12,
            drop_last=False,
            pin_memory=True,
            sampler=sampler,
        )

        N = len(ds)
        log.info(f"Generating {N} images...")

        real_imgs = []
        fake_imgs = []
        real_features_phenom = []
        fake_features_phenom = []
        real_features_dino = []
        fake_features_dino = []
        y1s = []
        exps = []
        cell_types = []
        zooms = []

        for batch in tqdm(loader, total=len(loader)):
            x1 = batch["img"].to(D.device, non_blocking=True)
            y1 = batch["perturbation_id"].to(D.device, non_blocking=True)
            exp = batch["experiment_id"].to(D.device, non_blocking=True)
            cell_type = batch["cell_type_id"].to(D.device, non_blocking=True)
            zoom = batch["zoom_id"].to(D.device, non_blocking=True)

            B, _, H, W = x1.shape
            x0 = torch.randn(B, 8, H // 8, W // 8, device=D.device)
            preds, _ = generate(
                model=state.ema.module,  # type: ignore
                x0=x0,
                y1=y1,
                z1=zoom,
                e1=exp,
                c1=cell_type,
            )
            preds = vae.decode(preds)

            real_features_phenom.append(phenom(x1).cpu())
            fake_features_phenom.append(phenom(preds).cpu())

            fake_features_dino.append(dino(cp2rgb(preds)).cpu())
            real_features_dino.append(dino(cp2rgb(x1)).cpu())

            real_imgs.append(x1.cpu())
            fake_imgs.append(preds.cpu())
            y1s.append(y1.cpu())
            exps.append(exp.cpu())
            cell_types.append(cell_type.cpu())
            zooms.append(zoom.cpu())

        D.barrier()

        # move to gpu, all_gather, move to cpu
        # we do it this way because all_gather is not supported on CPU
        def gather_cpu(x) -> np.ndarray:
            x = torch.cat(x, dim=0).to(D.device)
            x = D.gather_concat(x).cpu().numpy()
            return x[:N]  # truncate duplicates

        real_imgs = gather_cpu(real_imgs)
        fake_imgs = gather_cpu(fake_imgs)
        real_features_phenom = gather_cpu(real_features_phenom)
        fake_features_phenom = gather_cpu(fake_features_phenom)
        real_features_dino = gather_cpu(real_features_dino)
        fake_features_dino = gather_cpu(fake_features_dino)
        y1s = gather_cpu(y1s)
        exps = gather_cpu(exps)
        cell_types = gather_cpu(cell_types)
        zooms = gather_cpu(zooms)

        D.barrier()

        if D.rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            np.savez(
                os.path.join(output_dir, "results.npz"),
                real_imgs=real_imgs,
                fake_imgs=fake_imgs,
                real_features_phenom=real_features_phenom,
                fake_features_phenom=fake_features_phenom,
                real_features_dino=real_features_dino,
                fake_features_dino=fake_features_dino,
                y1s=y1s,
                exps=exps,
                cell_types=cell_types,
                zooms=zooms,
            )
            log.info(f"Saved results to {output_dir}/results.npz")

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
