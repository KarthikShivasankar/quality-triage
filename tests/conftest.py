"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest

from code_review_agent.config import reset_config


@pytest.fixture(autouse=True)
def _reset_config():
    reset_config()
    yield
    reset_config()


def pytest_collection_modifyitems(config, items):
    if os.environ.get("QUALITY_TRIAGE_SKIP_E2E") == "1":
        skip_e2e = pytest.mark.skip(reason="QUALITY_TRIAGE_SKIP_E2E=1")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)

    if os.environ.get("QUALITY_TRIAGE_LIVE_LLM") == "1":
        return
    skip = pytest.mark.skip(
        reason="set QUALITY_TRIAGE_LIVE_LLM=1 to run live LLM tests"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
