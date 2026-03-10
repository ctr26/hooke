from __future__ import annotations

import pytest
import torch


@pytest.fixture
def seed() -> int:
    torch.manual_seed(42)
    return 42
