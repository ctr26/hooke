"""Smoke tests: all CLI entry points respond to --help."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("command", ["hsh-train", "hsh-finetune", "hsh-eval", "hsh-infer"])
def test_cli_help(command: str) -> None:
    module = command.replace("hsh-", "")
    result = subprocess.run(
        [sys.executable, "-m", f"hsh.{module}", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"{command} --help failed:\n{result.stderr}"
    output = result.stdout.lower()
    assert "config" in output or "override" in output, (
        f"{command} --help did not show Hydra config output:\n{result.stdout}"
    )
