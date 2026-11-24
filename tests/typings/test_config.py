"""Tests for TOML bot configuration loading, deep-merging, strict validation, and LLMConfig wiring."""

from typing import get_args
from pathlib import Path

import pytest
from pydantic import ValidationError
from logfire._internal.constants import LEVEL_NUMBERS

from discordbot.typings.llm import LLMConfig
from discordbot.typings.config import (
    DEFAULT_CONFIG_PATH,
    BotConfig,
    LoggingConfig,
    load_bot_config,
    clear_bot_config_cache,
)


def test_log_level_setting_accepts_only_real_logfire_levels() -> None:
    """`LOG_LEVEL` is checked against logfire's own table, not a hand-copied list."""
    accepted = set(get_args(LoggingConfig.model_fields["log_level"].annotation))
    assert accepted == set(LEVEL_NUMBERS)


def test_packaged_config_toml_parses_cleanly_with_expected_defaults() -> None:
    """Packaged `src/discordbot/config.toml` validates cleanly with neutral defaults."""
    assert DEFAULT_CONFIG_PATH.is_file()

    config = load_bot_config()

    assert isinstance(config, BotConfig)
    assert config.global_config.bot.activity == "Discord"
    assert config.global_config.emojis.twitter == "🐦"
    assert config.global_config.reply.file_api_enabled is True
    assert config.global_config.reply.streaming_enabled is True
    assert config.global_config.reply.feedback_enabled is True
    assert config.global_config.reply.slow_model_blind_test_enabled is False
    assert config.server.auto_expand.model_dump() == {
        "twitter": True,
        "threads": True,
        "facebook": True,
        "instagram": True,
        "douyin": True,
    }


def test_local_config_toml_deep_merges_partial_overrides(bot_config_isolated: Path) -> None:
    """Partial tables in local `config.toml` override only the named keys and preserve siblings."""
    bot_config_isolated.write_text(
        data=(
            '[global.bot]\nactivity = "破貓上線"\n\n'
            '[global.emojis]\ntwitter = "<:twitter:1234567890>"\n\n'
            "[global.reply]\nstreaming_enabled = false\nslow_model_blind_test_enabled = true\n\n"
            "[server.auto_expand]\ntwitter = false\n"
        ),
        encoding="utf-8",
    )

    config = load_bot_config()

    assert config.global_config.bot.activity == "破貓上線"
    assert config.global_config.emojis.twitter == "<:twitter:1234567890>"
    assert config.global_config.emojis.threads == "🧵"
    assert config.global_config.reply.streaming_enabled is False
    assert config.global_config.reply.slow_model_blind_test_enabled is True
    assert config.global_config.reply.file_api_enabled is True
    assert config.global_config.reply.feedback_enabled is True
    assert config.server.auto_expand.twitter is False
    assert config.server.auto_expand.threads is True


@pytest.mark.parametrize(
    "bad_toml",
    [
        "unknown_top = true\n",
        "[global]\nunknown_global = true\n",
        "[global.bot]\nunknown_bot = true\n",
        '[global.emojis]\nunknown_emoji = "x"\n',
        "[global.reply]\nanswer_streaming_enabled = false\n",
        "[server]\nunknown_server = true\n",
        "[server.auto_expand]\nbilibili = true\n",
    ],
    ids=["root", "global", "bot", "emojis", "reply", "server", "auto-expand"],
)
def test_unknown_keys_fail_fast_at_every_nesting_level(tmp_path: Path, bad_toml: str) -> None:
    """Every config model forbids extra keys so typos fail at load rather than silently."""
    bad_file = tmp_path / "bad.toml"
    bad_file.write_text(data=bad_toml, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_bot_config(local_path=bad_file)


def test_explicit_missing_local_path_falls_back_to_packaged_defaults(tmp_path: Path) -> None:
    """Passing a non-existent `local_path` returns the packaged defaults cleanly."""
    config = load_bot_config(local_path=tmp_path / "missing.toml")

    assert config.global_config.bot.activity == "Discord"
    assert config.server.auto_expand.twitter is True


def test_llm_config_combines_env_credentials_with_toml_reply_switches(
    bot_config_isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`LLMConfig()` loads API credentials from env and switches from `[global.reply]`."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="http://localhost:4000/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="sk-test")
    monkeypatch.setenv(name="GEMINI_API_KEY", value="AIza-test")
    bot_config_isolated.write_text(
        data=(
            "[global.reply]\n"
            "file_api_enabled = false\n"
            "streaming_enabled = false\n"
            "feedback_enabled = false\n"
            "slow_model_blind_test_enabled = true\n"
        ),
        encoding="utf-8",
    )
    clear_bot_config_cache()

    loaded = LLMConfig()
    assert loaded.base_url == "http://localhost:4000/v1"
    assert loaded.api_key == "sk-test"
    assert loaded.gemini_api_key == "AIza-test"
    assert loaded.file_api_enabled is False
    assert loaded.answer_streaming_enabled is False
    assert loaded.feedback_enabled is False
    assert loaded.slow_model_blind_test_enabled is True

    # Explicit kwargs override both env and TOML, while `model_construct` stays pure static.
    overridden = LLMConfig(api_key="explicit", answer_streaming_enabled=True)
    assert overridden.api_key == "explicit"
    assert overridden.answer_streaming_enabled is True
    assert overridden.file_api_enabled is False

    constructed = LLMConfig.model_construct()
    assert constructed.api_key == ""
    assert constructed.answer_streaming_enabled is True
    assert constructed.file_api_enabled is True
