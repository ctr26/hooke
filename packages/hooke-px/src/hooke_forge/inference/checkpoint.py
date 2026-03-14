"""Checkpoint discovery and configuration extraction."""

import re
from pathlib import Path


def find_checkpoint(training_dir: Path, step: int) -> Path:
    """Find checkpoint file from training directory and step number.

    Handles multiple directory structures:
    - training_dir/checkpoints/step_{N}.ckpt (direct job directory)
    - training_dir/{job_id}/checkpoints/step_{N}.ckpt (timestamp directory)

    Args:
        training_dir: Training output directory (job dir or timestamp dir)
        step: Checkpoint step number

    Returns:
        Path to checkpoint file

    Raises:
        FileNotFoundError: If checkpoint not found
    """
    training_dir = Path(training_dir)
    checkpoint_name = f"step_{step}.ckpt"

    # Try direct path: training_dir/checkpoints/step_N.ckpt
    direct_path = training_dir / "checkpoints" / checkpoint_name
    if direct_path.exists():
        return direct_path

    # Try subdirectory: training_dir/{job_id}/checkpoints/step_N.ckpt
    for subdir in training_dir.iterdir():
        if subdir.is_dir():
            ckpt_path = subdir / "checkpoints" / checkpoint_name
            if ckpt_path.exists():
                return ckpt_path

    # List available checkpoints for helpful error
    available = []
    for ckpt_dir in training_dir.rglob("checkpoints"):
        available.extend(ckpt_dir.glob("step_*.ckpt"))

    if available:
        steps = sorted([int(p.stem.split("_")[1]) for p in available])
        raise FileNotFoundError(f"Checkpoint step_{step}.ckpt not found in {training_dir}. Available steps: {steps}")
    else:
        raise FileNotFoundError(f"No checkpoints directory found in {training_dir}")


def extract_model_config(training_dir: Path) -> dict:
    """Extract model configuration from launch_cmd.txt.

    Looks for launch_cmd.txt in training_dir or parent directories.

    Args:
        training_dir: Training output directory

    Returns:
        Dict with model config (e.g., {"name": "DiT-XL/2"})
    """
    training_dir = Path(training_dir)

    # Search for launch_cmd.txt in training_dir and parents
    search_dirs = [training_dir, training_dir.parent, training_dir.parent.parent]
    launch_cmd_path = None

    for search_dir in search_dirs:
        candidate = search_dir / "launch_cmd.txt"
        if candidate.exists():
            launch_cmd_path = candidate
            break

    if launch_cmd_path is None:
        return {}

    launch_cmd = launch_cmd_path.read_text().strip()

    # Extract model.name
    config = {}
    model_keys = ["model.name"]

    for key in model_keys:
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

    return config


def get_checkpoint_training_dir(checkpoint_path: Path) -> Path:
    """Get training directory from checkpoint path.

    Expected structure: outputs/{timestamp}/{job_id}/checkpoints/step_N.ckpt

    Returns the job_id directory (parent of checkpoints).
    """
    checkpoint_path = Path(checkpoint_path)
    # checkpoints/ -> job_id/
    return checkpoint_path.parent.parent
