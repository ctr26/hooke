"""Smoke tests: all CLI entry points respond to --help."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("command", ["hsh-train", "hsh-finetune", "hsh-eval", "hsh-infer"])
def test_cli_help(command: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"hsh.{command.replace('hsh-', '')}",  "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"{command} --help failed:\n{result.stderr}"
    assert "usage:" in result.stdout.lower() or "options:" in result.stdout.lower()
