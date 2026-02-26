"""Training entry point for the transcriptomics Perceiver autoencoder.

Mirrors the structure of train.py but with:
  - Two models (autoencoder + discriminator) and two optimizers
  - ZINB reconstruction + perceptual + adversarial losses
  - Discriminator warm-up (adversarial loss turns on after ``adv_start_step``)
"""

import dataclasses
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from typing import Callable, Literal

import ornamentalist
import submitit
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from hooke_forge.data import dataset
from hooke_forge.model.tokenizer import DataFrameTokenizer, MetaDataConfig
from hooke_forge.model.tx_autoencoders import (
    TxPerceiverAE,
    TxDiscriminator,
    TxAMPerceptualLoss,
    zinb_nll,
    sample_zinb,
    hinge_disc_loss,
)
from hooke_forge.training.trainer import log
from hooke_forge.utils.distributed import Distributed, rank_zero
from hooke_forge.utils.ema import KarrasEMA
from hooke_forge.utils.infinite_dataloader import infinite_dataloader
from hooke_forge.utils.name_run import generate_random_name

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

@dataclasses.dataclass
class TxAETrainState:
    ddp_ae: DDP
    ddp_disc: DDP
    ema_ae: KarrasEMA
    opt_ae: torch.optim.Optimizer
    opt_disc: torch.optim.Optimizer
    global_step: int
    tokenizer: DataFrameTokenizer

    def _to_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "ddp_ae": self.ddp_ae.state_dict(),
            "ddp_disc": self.ddp_disc.state_dict(),
            "ema_ae": self.ema_ae.state_dict(),
            "opt_ae": self.opt_ae.state_dict(),
            "opt_disc": self.opt_disc.state_dict(),
            "tokenizer": self.tokenizer.state_dict(),
        }

    @rank_zero()
    def save_ckpt(self, path: str) -> None:
        d = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(delete=False, dir=d, suffix=".tmp") as f:
            temp_path = f.name
            torch.save(self._to_dict(), temp_path)
        os.rename(temp_path, path)

    def load_ckpt(self, path: str, device: torch.device, strict: bool = True) -> None:
        state = torch.load(path, weights_only=False, map_location=device)
        self.global_step = state["global_step"]
        self.ddp_ae.load_state_dict(state["ddp_ae"], strict=strict)
        self.ddp_disc.load_state_dict(state["ddp_disc"], strict=strict)
        self.ema_ae.load_state_dict(state["ema_ae"], strict=strict)
        if strict:
            self.opt_ae.load_state_dict(state["opt_ae"])
            self.opt_disc.load_state_dict(state["opt_disc"])
        else:
            _log.info("Skipping optimizer state load (strict=False)")
        if "tokenizer" in state:
            self.tokenizer = DataFrameTokenizer.from_state_dict(state["tokenizer"])

    def save_latest_ckpt(self, dir: str) -> None:
        path = os.path.join(dir, f"step_{self.global_step}.ckpt")
        self.save_ckpt(path)

    def load_latest_ckpt(self, dir: str, device: torch.device, strict: bool = True) -> None:
        pattern = r"step_(\d+).ckpt"
        fnames = [
            f.name
            for f in os.scandir(dir)
            if f.is_file() and re.fullmatch(pattern, f.name)
        ]
        if len(fnames) == 0:
            _log.info("No existing Tx AE checkpoints found in %s, skipping load.", dir)
            return
        latest = max(fnames, key=lambda x: int(re.fullmatch(pattern, x).group(1)))
        path = os.path.join(dir, latest)
        _log.info("Loading Tx AE checkpoint from %s", path)
        self.load_ckpt(path, device, strict=strict)


@torch.inference_mode()
def evaluate_tx_ae(
    *,
    state: TxAETrainState,
    loader: torch.utils.data.DataLoader,
    D: Distributed,
    use_ema: bool,
) -> None:
    """Compute validation ZINB NLL and reconstruction MAE (log1p space)."""
    ae: TxPerceiverAE = state.ema_ae.module if use_ema else state.ddp_ae.module  # type: ignore
    ae.eval()

    running_nll = torch.tensor(0.0, device=D.device)
    running_mae = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)

    for batch in tqdm(
        loader, desc="Evaluating Tx AE", total=len(loader),
        unit="step", disable=D.rank != 0,
    ):
        meta = {k: v.to(D.device, non_blocking=True) for k, v in batch["meta"].items()}
        tx_raw = batch["tx_raw"].to(D.device, non_blocking=True)

        _z, zinb_params = ae(
            tx_raw, meta["cell_type"], meta["assay_type"], train=False,
        )
        nll = zinb_nll(
            zinb_params["log_mu"], zinb_params["log_theta"],
            zinb_params["logit_pi"], tx_raw,
        )

        # Reconstruction MAE in log1p-normalised space
        mu_hat = (
            (1.0 - torch.sigmoid(zinb_params["logit_pi"]))
            * torch.exp(zinb_params["log_mu"])
        )
        target_sum = 4_000.0
        mu_norm = torch.log1p(
            mu_hat / mu_hat.sum(dim=-1, keepdim=True).clamp(min=1e-8) * target_sum
        )
        tx_norm = batch["tx"].to(D.device, non_blocking=True)
        mae = (mu_norm - tx_norm).abs().mean()

        B = tx_raw.shape[0]
        running_nll += nll * B
        running_mae += mae * B
        num_samples += B

    running_nll = D.all_reduce(running_nll, op="sum")
    running_mae = D.all_reduce(running_mae, op="sum")
    num_samples = D.all_reduce(num_samples, op="sum")

    prefix = "ema" if use_ema else "ddp"
    log(
        step=state.global_step,
        data={
            f"val/{prefix}_zinb_nll": (running_nll / num_samples).item(),
            f"val/{prefix}_recon_mae": (running_mae / num_samples).item(),
        },
    )
    D.barrier()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@ornamentalist.configure(name="tx_ae_train")
def train_tx_ae(
    *,
    state: TxAETrainState,
    output_dir: str,
    D: Distributed,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None = None,
    # Loss weights
    lambda_perc: float = ornamentalist.Configurable[1.0],
    lambda_adv: float = ornamentalist.Configurable[0.1],
    adv_start_step: int = ornamentalist.Configurable[50_000],
    zinb_sample_tau: float = ornamentalist.Configurable[0.1],
    # Schedule
    num_steps: int = ornamentalist.Configurable[500_000],
    log_every_n_steps: int = ornamentalist.Configurable[250],
    eval_every_n_steps: int = ornamentalist.Configurable[10_000],
    ckpt_every_n_steps: int = ornamentalist.Configurable[20_000],
) -> None:
    start_step = state.global_step
    data_iter = infinite_dataloader(train_loader, start_step=start_step, fast_resume=True)

    txam_loss = TxAMPerceptualLoss().to(D.device)

    running = {
        "zinb": torch.tensor(0.0, device=D.device),
        "perc": torch.tensor(0.0, device=D.device),
        "adv_g": torch.tensor(0.0, device=D.device),
        "disc": torch.tensor(0.0, device=D.device),
    }
    num_samples = torch.tensor(0, device=D.device)
    start_time = time.time()

    for step_idx in tqdm(
        range(start_step, num_steps),
        desc="Training Tx AE",
        initial=start_step,
        total=num_steps,
        unit="step",
        disable=D.rank != 0,
    ):
        state.ddp_ae.train()
        state.ddp_disc.train()

        batch = next(data_iter)
        meta = {k: v.to(D.device, non_blocking=True) for k, v in batch["meta"].items()}
        tx_raw = batch["tx_raw"].to(D.device, non_blocking=True)
        B = tx_raw.shape[0]

        # Access the underlying AE module for gene_input and sample_zinb
        ae_module: TxPerceiverAE = state.ddp_ae.module  # type: ignore

        # ---- Generator step ----
        _z, zinb_params = state.ddp_ae(
            tx_raw, meta["cell_type"], meta["assay_type"], train=True,
        )
        l_zinb = zinb_nll(
            zinb_params["log_mu"], zinb_params["log_theta"],
            zinb_params["logit_pi"], tx_raw,
        )

        zinb_samples = sample_zinb(
            zinb_params["log_mu"], zinb_params["log_theta"],
            zinb_params["logit_pi"], tau=zinb_sample_tau,
        )

        l_perc = txam_loss(zinb_samples, tx_raw)

        l_adv_g = torch.tensor(0.0, device=D.device)
        adv_active = state.global_step >= adv_start_step
        if adv_active:
            fake_logits = state.ddp_disc(zinb_samples, ae_module.gene_input)
            l_adv_g = -fake_logits.mean()

        l_total = l_zinb + lambda_perc * l_perc + lambda_adv * l_adv_g

        state.opt_ae.zero_grad()
        l_total.backward()
        grad_norm_ae = torch.nn.utils.clip_grad_norm_(
            state.ddp_ae.parameters(), max_norm=1.0,
        )
        state.opt_ae.step()
        state.ema_ae.update(model=state.ddp_ae.module, step=state.global_step + 1)

        # ---- Discriminator step ----
        l_disc = torch.tensor(0.0, device=D.device)
        grad_norm_disc = torch.tensor(0.0, device=D.device)
        if adv_active:
            real_logits = state.ddp_disc(tx_raw, ae_module.gene_input)
            fake_logits_d = state.ddp_disc(
                zinb_samples.detach(), ae_module.gene_input,
            )
            l_disc = hinge_disc_loss(real_logits, fake_logits_d)

            state.opt_disc.zero_grad()
            l_disc.backward()
            grad_norm_disc = torch.nn.utils.clip_grad_norm_(
                state.ddp_disc.parameters(), max_norm=1.0,
            )
            state.opt_disc.step()

        state.global_step += 1

        # ---- Accumulate running losses ----
        running["zinb"] += l_zinb.detach() * B
        running["perc"] += l_perc.detach() * B
        running["adv_g"] += l_adv_g.detach() * B
        running["disc"] += l_disc.detach() * B
        num_samples += B

        # ---- Logging ----
        if state.global_step % log_every_n_steps == 0:
            elapsed = time.time() - start_time
            start_time = time.time()

            for k in running:
                running[k] = D.all_reduce(running[k], op="sum")
            num_samples = D.all_reduce(num_samples, op="sum")

            log_data = {
                "train/zinb_nll": (running["zinb"] / num_samples).item(),
                "train/perc_loss": (running["perc"] / num_samples).item(),
                "train/adv_g_loss": (running["adv_g"] / num_samples).item(),
                "train/disc_loss": (running["disc"] / num_samples).item(),
                "train/total_loss": (
                    (running["zinb"] + lambda_perc * running["perc"]
                     + lambda_adv * running["adv_g"]) / num_samples
                ).item(),
                "train/grad_norm_ae": grad_norm_ae.item(),
                "train/grad_norm_disc": grad_norm_disc.item(),
                "train/throughput": (num_samples / elapsed).item()
                    if isinstance(num_samples, torch.Tensor)
                    else num_samples / elapsed,
                "train/adv_active": float(adv_active),
            }
            log(step=state.global_step, data=log_data)

            for k in running:
                running[k] = torch.tensor(0.0, device=D.device)
            num_samples = torch.tensor(0, device=D.device)

        # ---- Checkpointing ----
        if state.global_step % ckpt_every_n_steps == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            log(step=state.global_step, msg=f"Saving Tx AE checkpoint to {ckpt_dir}")
            state.save_latest_ckpt(dir=ckpt_dir)

        # ---- Evaluation ----
        if state.global_step % eval_every_n_steps == 0 and val_loader is not None:
            evaluate_tx_ae(state=state, loader=val_loader, D=D, use_ema=False)
            evaluate_tx_ae(state=state, loader=val_loader, D=D, use_ema=True)

    log(step=state.global_step, msg="Tx AE training complete")


# ---------------------------------------------------------------------------
# Checkpoint loading helper
# ---------------------------------------------------------------------------


def load_tokenizer_from_checkpoint(ckpt_dir: str) -> DataFrameTokenizer | None:
    pattern = r"step_(\d+).ckpt"
    if not os.path.exists(ckpt_dir):
        return None
    fnames = [
        f.name
        for f in os.scandir(ckpt_dir)
        if f.is_file() and re.fullmatch(pattern, f.name)
    ]
    if len(fnames) == 0:
        return None
    latest = max(fnames, key=lambda x: int(re.fullmatch(pattern, x).group(1)))
    path = os.path.join(ckpt_dir, latest)
    _log.info("Loading tokenizer from Tx AE checkpoint: %s", path)
    state = torch.load(path, weights_only=False, map_location="cpu")
    if "tokenizer" not in state:
        _log.warning("Checkpoint does not contain tokenizer")
        return None
    return DataFrameTokenizer.from_state_dict(state["tokenizer"])


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


@ornamentalist.configure(name="tx_ae_model")
def get_tx_ae_models(
    *,
    n_genes: int,
    n_slots: int = ornamentalist.Configurable[64],
    n_embd: int = ornamentalist.Configurable[512],
    n_heads: int = ornamentalist.Configurable[8],
    n_layers: int = ornamentalist.Configurable[6],
    dropout: float = ornamentalist.Configurable[0.0],
    bias: bool = ornamentalist.Configurable[True],
    label_dropout_prob: float = ornamentalist.Configurable[0.15],
    # Discriminator
    n_disc_genes: int = ornamentalist.Configurable[1000],
    disc_n_layer: int = ornamentalist.Configurable[3],
    disc_n_head: int = ornamentalist.Configurable[8],
    # Vocab sizes
    cell_type_vocab: int = ornamentalist.Configurable[55],
    assay_type_vocab: int = ornamentalist.Configurable[6],
) -> tuple[TxPerceiverAE, TxDiscriminator]:
    ae = TxPerceiverAE(
        n_genes=n_genes,
        n_slots=n_slots,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        bias=bias,
        cell_type_vocab=cell_type_vocab,
        assay_type_vocab=assay_type_vocab,
        label_dropout_prob=label_dropout_prob,
    )
    disc = TxDiscriminator(
        n_embd=n_embd,
        n_disc_genes=n_disc_genes,
        n_layer=disc_n_layer,
        n_head=disc_n_head,
        dropout=dropout,
        bias=bias,
    )
    return ae, disc


# ---------------------------------------------------------------------------
# PRNG
# ---------------------------------------------------------------------------


@ornamentalist.configure()
def prng(rank: int, seed: int = ornamentalist.Configurable[42]):
    local_seed = seed + rank
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    _log.info("Rank %d setting seed to %d", rank, local_seed)


# ---------------------------------------------------------------------------
# Checkpoint config
# ---------------------------------------------------------------------------


@ornamentalist.configure(name="ckpt")
def get_resume_checkpoint_dir(
    resume_from: str = ornamentalist.Configurable[""],
    strict: bool = ornamentalist.Configurable[True],
) -> tuple[str | None, bool]:
    ckpt_dir: str | None
    if not resume_from:
        ckpt_dir = None
    elif resume_from.endswith(".ckpt"):
        ckpt_dir = os.path.dirname(resume_from)
    else:
        ckpt_dir = resume_from
    return ckpt_dir, strict


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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
                    ["scontrol", "update", f"JobId={job_env.job_id}", f"JobName={name}"],
                    check=True, capture_output=True,
                )
                _log.info("Renamed SLURM job to '%s'", name)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                _log.warning("Could not rename SLURM job: %s", e)

        if D.rank == 0:
            wandb.init(
                name=name,
                id=name,
                dir=output_dir,
                notes=f"Tx AE — outputs: {output_dir}",
                project="big-img",
                entity="valencelabs",
                resume="allow",
                save_code=False,
                config=config,
            )

        _log.info("Job ID %s  output_dir=%s  name=%s", job_env.job_id, output_dir, name)
        _log.info("Config: %s", config)

        prng(D.rank)
        torch.set_float32_matmul_precision("medium")

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        resume_ckpt_dir, strict_ckpt = get_resume_checkpoint_dir()
        if resume_ckpt_dir:
            _log.info("Resuming from %s (strict=%s)", resume_ckpt_dir, strict_ckpt)
        else:
            resume_ckpt_dir = ckpt_dir

        existing_tokenizer = load_tokenizer_from_checkpoint(resume_ckpt_dir)

        # Data
        train_loader, val_loader, tokenizer, n_genes = dataset.get_tx_ae_dataloaders(
            tokenizer=existing_tokenizer,
        )
        assert tokenizer is not None

        # Models
        ae, disc = get_tx_ae_models(n_genes=n_genes)
        ae.to(D.device)
        disc.to(D.device)

        ae = torch.compile(ae)  # type: ignore
        disc = torch.compile(disc)  # type: ignore

        ddp_ae = DDP(ae)
        ddp_disc = DDP(disc)

        ema_ae = KarrasEMA(ae)

        opt_ae = torch.optim.Adam(
            ae.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
        )
        opt_disc = torch.optim.Adam(
            disc.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
        )

        state = TxAETrainState(
            ddp_ae=ddp_ae,
            ddp_disc=ddp_disc,
            ema_ae=ema_ae,
            opt_ae=opt_ae,
            opt_disc=opt_disc,
            global_step=0,
            tokenizer=tokenizer,
        )

        nparams_ae = sum(p.numel() for p in ae.parameters())
        nparams_disc = sum(p.numel() for p in disc.parameters())
        _log.info("AE: %.1fM params  |  Disc: %.1fM params", nparams_ae / 1e6, nparams_disc / 1e6)

        state.load_latest_ckpt(dir=resume_ckpt_dir, device=D.device, strict=strict_ckpt)

        train_tx_ae(
            state=state,
            output_dir=output_dir,
            D=D,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        if D.rank == 0:
            wandb.finish(quiet=True)


# ---------------------------------------------------------------------------
# Launcher (mirrors train.py)
# ---------------------------------------------------------------------------


def prepare_submission(config: ornamentalist.ConfigDict) -> Callable:
    def submission():
        return main(config)

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
    del desc

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
        _log.info("Submitted %s", jobs)

        if cluster == "local":
            _log.info("Running locally — stdout/stderr logged to %s", output_dir)
            _ = [j.results()[0] for j in jobs]
        elif cluster == "debug":
            _log.info("Running in debug mode (pdb on crash)")
            _ = [j.results()[0] for j in jobs]


def cli():
    configs = ornamentalist.cli()
    assert all(config["launcher"] == configs[0]["launcher"] for config in configs)
    ornamentalist.setup(configs[0], force=True)
    launcher(configs=configs)


if __name__ == "__main__":
    cli()
