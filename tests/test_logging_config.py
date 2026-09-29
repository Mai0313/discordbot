"""Tests for the logging settings read from the environment."""

from typing import get_args

from logfire._internal.constants import LEVEL_NUMBERS

from discordbot.typings.config import LoggingConfig


def test_log_level_setting_accepts_only_real_logfire_levels() -> None:
    """`LOG_LEVEL` is checked against logfire's own table, not a hand-copied list."""
    accepted = set(get_args(LoggingConfig.model_fields["log_level"].annotation))
    assert accepted == set(LEVEL_NUMBERS)
