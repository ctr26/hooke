"""Smoke tests: all CLI entry points respond to --help."""

from __future__ import annotations

import subprocess
import sys

import pytest

CLI_MODULES = [
    ("hsh-train", "hsh.train"),
    ("hsh-finetune", "hsh_finetune.finetune"),
    ("hsh-eval", "hsh.eval"),
    ("hsh-infer", "hsh.infer"),
]


@pytest.mark.parametrize("command,module", CLI_MODULES, ids=[c for c, _ in CLI_MODULES])
def test_cli_help(command: str, module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"{command} --help failed:\n{result.stderr}"
    assert "usage:" in result.stdout.lower() or "options:" in result.stdout.lower()
