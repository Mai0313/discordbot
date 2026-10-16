"""The settings template contributors copy into `.env`."""

import pkgutil
import importlib

from dotenv import dotenv_values
import pytest
from pydantic import AliasChoices
from pydantic_settings import BaseSettings

import discordbot
from discordbot.utils.media_delivery import MediaHostingConfig

from tests.helpers.source_tree import REPO_ROOT, env_example_names


def test_copied_env_example_leaves_media_hosting_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `.env` copied verbatim from `.env.example` leaves media hosting unavailable."""
    for name, value in dotenv_values(dotenv_path=REPO_ROOT / ".env.example").items():
        if name.startswith("MEDIA_HOSTING_") and value is not None:
            monkeypatch.setenv(name=name, value=value)

    assert not MediaHostingConfig().available


def test_env_example_names_every_setting() -> None:
    """`.env.example` names every variable a settings class reads.

    It is the only settings list a deployer sees, and `tests/conftest.py` clears exactly these
    names before any test runs, so one missing here would reach the suite from the shell.
    """
    for module in pkgutil.walk_packages(path=discordbot.__path__, prefix="discordbot."):
        importlib.import_module(name=module.name)
    classes: list[type[BaseSettings]] = [BaseSettings]
    for settings_class in classes:
        classes.extend(settings_class.__subclasses__())
    read: set[str] = set()
    for settings_class in classes:
        if not settings_class.__module__.startswith("discordbot."):
            continue
        for field_name, field in settings_class.model_fields.items():
            alias = field.validation_alias
            if isinstance(alias, AliasChoices):
                read.update(choice for choice in alias.choices if isinstance(choice, str))
            else:
                read.add(alias if isinstance(alias, str) else field_name.upper())

    assert read
    assert read - env_example_names() == set()
