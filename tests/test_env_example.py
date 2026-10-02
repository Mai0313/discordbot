"""The settings template contributors copy into `.env`."""

import os

from dotenv import dotenv_values
import pytest

from discordbot.utils.media_delivery import MediaHostingConfig

from tests.helpers.source_tree import REPO_ROOT


def test_copied_env_example_leaves_media_hosting_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `.env` copied verbatim from `.env.example` leaves media hosting unavailable.

    Hosting variables already in the environment are cleared first, so the result is the file's
    values alone.
    """
    for name in [name for name in os.environ if name.startswith("MEDIA_HOSTING_")]:
        monkeypatch.delenv(name=name)
    for name, value in dotenv_values(dotenv_path=REPO_ROOT / ".env.example").items():
        if name.startswith("MEDIA_HOSTING_") and value is not None:
            monkeypatch.setenv(name=name, value=value)

    assert not MediaHostingConfig().available
