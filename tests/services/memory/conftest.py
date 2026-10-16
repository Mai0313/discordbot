"""Fixtures shared by every test in this directory."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def memory_isolated(memory_isolated_dir: Path) -> Path:
    """Runs every test here on a throwaway memory store, `memory_job` table and caches.

    A test that reads the store's directory still requests `memory_isolated_dir` by name.
    """
    return memory_isolated_dir
