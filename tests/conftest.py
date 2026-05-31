"""
Shared pytest fixtures. Keeps the config singleton clean between tests and
provides a few small temp-project helpers. Everything here is offline.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the global config singleton before and after every test."""
    from code_review_agent.config import reset_config

    reset_config()
    yield
    reset_config()


@pytest.fixture
def sample_project(tmp_path):
    """A tiny multi-file Python project for detector/intel tests."""
    (tmp_path / "model.py").write_text(
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "def train(a, b, c, d, e, f, g):\n"
        + "    x = 1\n" * 50
        + "    return x\n",
        encoding="utf-8",
    )
    (tmp_path / "big.py").write_text(
        "class Big:\n"
        + "".join(f"    def m{i}(self):\n        return {i}\n" for i in range(20)),
        encoding="utf-8",
    )
    return tmp_path
