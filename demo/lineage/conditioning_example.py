#!/usr/bin/env python
"""Example: Model conditioning update with W&B lineage.

Shows how W&B tracks when conditioning changes propagate through pipeline:
1. Update conditioning config (new cell types, vocab)
2. Retrain model (linked to new config)
3. Query which checkpoints are stale

Run:
    python demo/lineage/conditioning_example.py
"""

import json
from pathlib import Path

import wandb

from demo.lineage.artifacts import log_artifact, use_artifact


def update_conditioning(output_dir: Path) -> str:
    """Update conditioning config (new cell types added)."""
    with wandb.init(project="hooke", job_type="conditioning") as run:
        conditioning = {
            "cell_types": ["ARPE19", "HUVEC", "HepG2", "U2OS"],  # Added U2OS
            "assay_types": ["cell_paint", "brightfield", "trek"],
            "vocab_size": 2048,
            "version": "v2",
        }

        config_path = output_dir / "conditioning.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(conditioning, indent=2))

        log_artifact(
            "conditioning-config",
            config_path,
            artifact_type="config",
            metadata=conditioning,
        )

        print(f"Logged conditioning config: {conditioning['cell_types']}")
        return f"conditioning-config:v{run.id}"


def pretrain_with_conditioning(config_artifact: str, output_dir: Path) -> str:
    """Pretrain model using conditioning config."""
    with wandb.init(project="hooke", job_type="pretrain") as run:
        # Consume conditioning — creates lineage edge
        config_path = use_artifact(config_artifact)
        config_file = list(config_path.glob("*.json"))[0]
        conditioning = json.loads(config_file.read_text())

        print(f"Building model with cell types: {conditioning['cell_types']}")
        print(f"Vocab size: {conditioning['vocab_size']}")

        # Mock: build model with conditioning
        # model = build_model(
        #     cell_type_vocab=conditioning["cell_types"],
        #     assay_type_vocab=conditioning["assay_types"],
        #     vocab_size=conditioning["vocab_size"],
        # )

        # Mock training
        checkpoint_path = output_dir / "checkpoint.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("mock checkpoint")

        # Log checkpoint — linked to conditioning via lineage
        log_artifact(
            "pretrain-checkpoint",
            checkpoint_path,
            artifact_type="model",
            metadata={
                "conditioning_artifact": config_artifact,
                "cell_types": conditioning["cell_types"],
                "vocab_size": conditioning["vocab_size"],
                "step": 200000,
            },
        )

        return f"pretrain-checkpoint:v{run.id}"


def find_stale_checkpoints(required_cell_type: str):
    """Find checkpoints that don't include a required cell type."""
    api = wandb.Api()

    print(f"\nSearching for checkpoints missing '{required_cell_type}'...")

    # This would query your actual project
    # for artifact in api.artifacts("your-entity/hooke", "model"):
    #     cell_types = artifact.metadata.get("cell_types", [])
    #     if required_cell_type not in cell_types:
    #         print(f"  Stale: {artifact.name} (has: {cell_types})")

    # Mock output
    print("  Stale: pretrain-checkpoint:v1 (has: [ARPE19, HUVEC])")
    print("  Stale: finetune-checkpoint:v1 (has: [ARPE19, HUVEC])")
    print("  Current: pretrain-checkpoint:v2 (has: [ARPE19, HUVEC, HepG2, U2OS])")


def main():
    output = Path("outputs/conditioning_demo")

    print("=" * 60)
    print("Step 1: Update conditioning config")
    print("=" * 60)
    config_artifact = update_conditioning(output / "config")

    print()
    print("=" * 60)
    print("Step 2: Pretrain with new conditioning")
    print("=" * 60)
    checkpoint = pretrain_with_conditioning(config_artifact, output / "pretrain")

    print()
    print("=" * 60)
    print("Step 3: Find stale checkpoints")
    print("=" * 60)
    find_stale_checkpoints("U2OS")

    print()
    print("=" * 60)
    print("Lineage")
    print("=" * 60)
    print("""
conditioning-config:v1 (old)
         ↓
  pretrain-checkpoint:v1  ← stale (missing U2OS)

conditioning-config:v2 (added U2OS)
         ↓
  pretrain-checkpoint:v2  ← current
    """)


if __name__ == "__main__":
    main()
