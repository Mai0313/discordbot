"""The suite's own isolation from the network and the developer's `.env` (`tests/conftest.py`)."""

import socket
from pathlib import Path

import dotenv
import pytest

from tests.conftest import NetworkAccessError, _remote_attempts


def test_a_lookup_past_loopback_is_refused() -> None:
    """A DNS lookup for a public host raises instead of leaving the machine."""
    with pytest.raises(NetworkAccessError):
        socket.getaddrinfo("example.com", 443)

    # Recorded as well, which would fail this test at teardown; here it was the point.
    assert _remote_attempts == ["example.com"]
    _remote_attempts.clear()


def test_a_dotenv_file_is_never_loaded(tmp_path: Path) -> None:
    """A `.env` the package would load at import is ignored for the whole run."""
    env_file = tmp_path / ".env"
    env_file.write_text("DISCORDBOT_SUITE_PROBE=loaded\n", encoding="utf-8")

    assert not dotenv.load_dotenv(dotenv_path=env_file)
