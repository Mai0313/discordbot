"""Reads back what a `UsageRecorder` wrote, for tests that assert on usage records."""

import json
from typing import Any
from pathlib import Path


def usage_records(directory: Path) -> list[dict[str, Any]]:
    """Reads every recorded line under `directory` back, parsed as its own JSON object."""
    return [
        json.loads(line)
        for path in sorted(directory.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
