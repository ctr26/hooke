"""Model lineage tracing from training logs.

Traces the training history through resume_from chains to construct
data version strings for structured output paths.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


def parse_config_from_log(log_path: Path) -> dict | None:
    """Parse config dict from INFO:__main__:config={...} line.

    Args:
        log_path: Path to a training log file (e.g., 12583183_0_log.out)

    Returns:
        Parsed config dictionary, or None if not found
    """
    pattern = re.compile(r"INFO:__main__:config=(\{.*\})")
    try:
        with open(log_path) as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    return ast.literal_eval(match.group(1))
    except OSError as e:
        log.warning(f"Could not read log file {log_path}: {e}")
    return None


def extract_parquet_name(path: str) -> str:
    """Extract version name from parquet path.

    Args:
        path: Full path to parquet file

    Returns:
        Stem of the parquet file (e.g., 'pretraining_v6' from
        '/path/to/pretraining_v6.parquet')
    """
    return Path(path).stem


def get_job_id_from_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
    """Extract (timestamp_dir, job_id) from checkpoint path.

    Args:
        checkpoint_path: Path like '.../outputs/1768305605/12583183/checkpoints'

    Returns:
        Tuple of (timestamp_dir, job_id)
    """
    parts = Path(checkpoint_path).parts
    try:
        idx = parts.index("outputs")
        return parts[idx + 1], parts[idx + 2]
    except (ValueError, IndexError):
        log.warning(f"Could not parse checkpoint path: {checkpoint_path}")
        return "", ""


def find_log_file(training_dir: Path) -> Path | None:
    """Find the rank 0 log file for a training directory.

    Args:
        training_dir: Training job directory (e.g., outputs/1768305605/12583183)

    Returns:
        Path to the log file, or None if not found
    """
    training_dir = Path(training_dir)
    timestamp_dir = training_dir.parent
    job_id = training_dir.name

    # Log file is at: outputs/{timestamp}/{job_id}_0_log.out
    log_file = timestamp_dir / f"{job_id}_0_log.out"
    if log_file.exists():
        return log_file

    # Also try looking in the job directory itself
    alt_log = training_dir / f"{job_id}_0_log.out"
    if alt_log.exists():
        return alt_log

    return None


def get_model_lineage(training_dir: Path) -> dict:
    """Trace model lineage through resume_from chain.

    Follows the resume_from chain in training configs to build the full
    lineage of data versions used to train the model.

    Args:
        training_dir: Training job directory containing checkpoints

    Returns:
        Dict with:
            - data_version: str (e.g., "pretraining_v6/cross_cell_line_v2")
            - model_config: str (e.g., "DiT-XL")
            - lineage_chain: list of training configs for metadata
    """
    training_dir = Path(training_dir)
    lineage_chain = []
    parquet_versions = []
    model_config = None

    current_dir = training_dir
    visited = set()  # Prevent infinite loops

    while current_dir and str(current_dir) not in visited:
        visited.add(str(current_dir))

        log_file = find_log_file(current_dir)
        if not log_file:
            log.warning(f"No log file found for {current_dir}")
            break

        config = parse_config_from_log(log_file)
        if not config:
            log.warning(f"Could not parse config from {log_file}")
            break

        lineage_chain.append(
            {
                "training_dir": str(current_dir),
                "log_file": str(log_file),
                "config": config,
            }
        )

        # Extract model config from first (most recent) entry
        if model_config is None and "model" in config:
            name = config["model"].get("name", "")
            # DiT-XL/2 -> DiT-XL (strip the patch size suffix)
            model_config = name.split("/")[0] if name else None

        # Get parquet version
        parquet_path = config.get("get_dataloaders", {}).get("path", "")
        if parquet_path:
            parquet_versions.append(extract_parquet_name(parquet_path))

        # Follow resume_from chain
        resume_from = config.get("ckpt", {}).get("resume_from", "")
        if not resume_from:
            break

        # Parse resume_from to get parent training dir
        timestamp, job_id = get_job_id_from_checkpoint_path(resume_from)
        if not timestamp or not job_id:
            log.warning(f"Could not parse resume_from path: {resume_from}")
            break

        # Find the outputs base directory from the resume_from path
        # Path is like: /path/to/outputs/1768305605/12583183/checkpoints
        resume_path = Path(resume_from)
        try:
            parts = resume_path.parts
            idx = parts.index("outputs")
            outputs_base = Path(*parts[: idx + 1])
            current_dir = outputs_base / timestamp / job_id
        except (ValueError, IndexError):
            log.warning(f"Could not determine outputs base from: {resume_from}")
            break

    # Deduplicate consecutive identical versions
    # Reverse to chronological order (oldest first)
    parquet_versions.reverse()
    deduplicated = []
    for v in parquet_versions:
        if not deduplicated or deduplicated[-1] != v:
            deduplicated.append(v)

    data_version = "/".join(deduplicated) if deduplicated else "unknown"
    model_config = model_config or "unknown"

    log.info(f"Traced lineage: {len(lineage_chain)} training runs")
    log.info(f"Data version: {data_version}")
    log.info(f"Model config: {model_config}")

    return {
        "data_version": data_version,
        "model_config": model_config,
        "lineage_chain": lineage_chain,
    }
