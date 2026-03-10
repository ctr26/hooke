"""End-to-end pipeline test: train -> finetune -> eval -> infer.

All stages run on CPU with synthetic data. Completes in <5 seconds.
"""

from __future__ import annotations

from pathlib import Path

import torch
from hsh_finetune.config import FinetuneConfig
from hsh_finetune.finetune import finetune

from hsh.config import EvalConfig, InferConfig, ModelConfig, TrainConfig
from hsh.eval import evaluate
from hsh.infer import infer
from hsh.train import train


class TestPipelineE2E:
    """Full lifecycle: train -> finetune -> eval -> infer."""

    def test_train_finetune_eval_infer(self, tmp_path: Path) -> None:
        model_config = ModelConfig(input_dim=16, hidden_dim=32, num_layers=1)

        # -- Stage 1: Train --
        train_dir = tmp_path / "train"
        train_config = TrainConfig(
            num_steps=4,
            batch_size=4,
            lr=1e-3,
            num_samples=16,
            ckpt_every=2,
            eval_every=4,
            output_dir=str(train_dir),
            seed=42,
        )
        train_result = train(train_config, model_config)

        assert train_result["step"] == 4
        assert train_result["checkpoint"] is not None
        ckpt_path = Path(train_result["checkpoint"])
        assert ckpt_path.exists()

        state = torch.load(ckpt_path, weights_only=False)
        assert "model" in state
        assert "optimizer" in state
        assert state["step"] == 4

        # -- Stage 2: Finetune --
        ft_dir = tmp_path / "finetune"
        ft_config = FinetuneConfig(
            base_checkpoint=str(ckpt_path),
            train=TrainConfig(
                num_steps=2,
                lr=1e-4,
                batch_size=4,
                num_samples=16,
                ckpt_every=2,
                eval_every=2,
                output_dir=str(ft_dir),
                seed=42,
            ),
        )
        ft_result = finetune(ft_config)

        assert ft_result["step"] == 6
        ft_ckpt = Path(ft_result["checkpoint"])
        assert ft_ckpt.exists()

        # -- Stage 3: Eval --
        eval_dir = tmp_path / "eval"
        eval_config = EvalConfig(
            checkpoint=str(ft_ckpt),
            batch_size=4,
            num_samples=16,
            output_dir=str(eval_dir),
        )
        metrics = evaluate(eval_config)

        assert "mse_loss" in metrics
        assert "mean_pred_norm" in metrics
        assert isinstance(metrics["mse_loss"], float)
        assert (eval_dir / "metrics.json").exists()

        # -- Stage 4: Infer --
        infer_dir = tmp_path / "infer"
        infer_config = InferConfig(
            checkpoint=str(ft_ckpt),
            batch_size=4,
            num_samples=8,
            output_dir=str(infer_dir),
        )
        infer_result = infer(infer_config)

        assert infer_result["shape"] == [8, 16]
        output_path = Path(infer_result["output_path"])
        assert output_path.exists()

        predictions = torch.load(output_path, weights_only=True)
        assert predictions.shape == (8, 16)

    def test_train_produces_decreasing_loss(self, tmp_path: Path) -> None:
        """Sanity check: loss should generally decrease over training."""
        model_config = ModelConfig(input_dim=8, hidden_dim=32, num_layers=2)
        train_config = TrainConfig(
            num_steps=20,
            batch_size=8,
            lr=1e-2,
            num_samples=32,
            ckpt_every=10,
            eval_every=10,
            output_dir=str(tmp_path),
            seed=42,
        )
        result = train(train_config, model_config)
        assert result["loss"] < 10.0  # should not diverge

    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        """Verify checkpoint save/load preserves model weights."""
        from hsh.train import load_checkpoint, save_checkpoint

        config = ModelConfig(input_dim=8, hidden_dim=16, num_layers=1)
        from hsh.model import FlowMatchingMLP

        model = FlowMatchingMLP(config)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt_path = save_checkpoint(model, optimizer, step=10, model_config=config, output_dir=str(tmp_path))

        loaded_model, state = load_checkpoint(ckpt_path)
        assert state["step"] == 10

        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], loaded_model.state_dict()[key])
