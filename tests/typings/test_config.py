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
from discordbot.cogs.economy.cog import EconomyCogs
from discordbot.admin.usage_report import _parse_args as parse_usage_report_args
from discordbot.utils.media_delivery import build_media_delivery_planner
from discordbot.cogs.media_cleanup.cog import MediaCleanupCogs

from tests.helpers.discord_mocks import stub_bot


def test_log_level_setting_accepts_only_real_logfire_levels() -> None:
    """`[global.logging].level` is checked against logfire's own table, not a hand-copied list."""
    accepted = set(get_args(LoggingConfig.model_fields["level"].annotation))
    assert accepted == set(LEVEL_NUMBERS)


def test_packaged_config_toml_parses_cleanly_with_expected_defaults() -> None:
    """Packaged `src/discordbot/config.toml` validates cleanly with neutral defaults."""
    assert DEFAULT_CONFIG_PATH.is_file()

    config = load_bot_config()

    assert isinstance(config, BotConfig)
    assert config.global_config.bot.activity == "Discord"
    assert config.global_config.logging.level == "debug"
    assert config.global_config.usage.enabled is True
    assert config.global_config.usage.directory == "./data/usage"
    assert config.global_config.memory.git_enabled is True
    assert config.global_config.media_hosting.enabled is True
    assert config.global_config.media_hosting.available is False
    assert config.global_config.media_hosting.cleanup_enabled is False
    assert config.global_config.economy.allow_central_bank_self_approval is False
    assert config.global_config.emojis.twitter == "🐦"
    assert config.global_config.reply.file_api_enabled is True
    assert config.global_config.reply.streaming_enabled is True
    assert config.global_config.reply.feedback_enabled is True
    assert config.global_config.models.image == "gemini-nano-banana-2.1"
    assert config.global_config.models.video == "gemini-omni-1.2-flash"
    assert config.global_config.models.music == "lyria-3.5"
    assert config.global_config.models.tts == "gemini-3.8-flash-tts"
    assert config.global_config.models.antigravity == "antigravity-preview-09-2026"
    assert config.global_config.models.antigravity_reasoning == "gemini-4-hydrogen"
    assert (
        config.global_config.models.triage.name,
        config.global_config.models.triage.effort,
    ) == ("gemini-3.5-flash-lite", "minimal")
    assert (config.global_config.models.fast.name, config.global_config.models.fast.effort) == (
        "gemini-4-hydrogen",
        "low",
    )
    assert (
        config.global_config.models.memory.name,
        config.global_config.models.memory.effort,
    ) == ("gemini-4-neon", "medium")
    assert config.global_config.models.slow.display_name == ""
    assert [(c.name, c.effort) for c in config.global_config.models.slow.candidates] == [
        ("gemini-4-neon", "low")
    ]
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
            '[global.logging]\nlevel = "info"\n\n'
            "[global.usage]\nenabled = false\n\n"
            "[global.memory]\ngit_enabled = false\n\n"
            '[global.media_hosting]\nbase_url = "https://media.example"\nserve_dir = "/srv/media"\n\n'
            "[global.economy]\nallow_central_bank_self_approval = true\n\n"
            '[global.emojis]\ntwitter = "<:twitter:1234567890>"\n\n'
            "[global.reply]\nstreaming_enabled = false\n\n"
            '[global.models]\nimage = "gemini-image-override"\n'
            'fast = { effort = "medium" }\n\n'
            '[global.models.slow]\ndisplay_name = "Gemini Next"\n'
            "candidates = [\n"
            '  { name = "gemini-4-neon", effort = "low" },\n'
            '  { name = "gemini-4-hydrogen", effort = "medium" },\n'
            "]\n\n"
            "[server.auto_expand]\ntwitter = false\n"
        ),
        encoding="utf-8",
    )

    config = load_bot_config()

    assert config.global_config.bot.activity == "破貓上線"
    assert config.global_config.logging.level == "info"
    assert config.global_config.usage.enabled is False
    assert config.global_config.usage.directory == "./data/usage"
    assert config.global_config.memory.git_enabled is False
    assert config.global_config.media_hosting.available is True
    assert config.global_config.media_hosting.max_bytes == 8 * 1024**3
    assert config.global_config.media_hosting.retention_hours == 168.0
    assert config.global_config.economy.allow_central_bank_self_approval is True
    assert config.global_config.emojis.twitter == "<:twitter:1234567890>"
    assert config.global_config.emojis.threads == "🧵"
    assert config.global_config.reply.streaming_enabled is False
    assert config.global_config.reply.file_api_enabled is True
    assert config.global_config.reply.feedback_enabled is True
    assert config.global_config.models.image == "gemini-image-override"
    assert config.global_config.models.video == "gemini-omni-1.2-flash"
    assert (config.global_config.models.fast.name, config.global_config.models.fast.effort) == (
        "gemini-4-hydrogen",
        "medium",
    )
    assert config.global_config.models.slow.display_name == "Gemini Next"
    assert [(c.name, c.effort) for c in config.global_config.models.slow.candidates] == [
        ("gemini-4-neon", "low"),
        ("gemini-4-hydrogen", "medium"),
    ]
    assert config.server.auto_expand.twitter is False
    assert config.server.auto_expand.threads is True


@pytest.mark.parametrize(
    "bad_toml",
    [
        "unknown_top = true\n",
        "[global]\nunknown_global = true\n",
        "[global.bot]\nunknown_bot = true\n",
        "[global.logging]\nunknown_logging = true\n",
        "[global.usage]\nunknown_usage = true\n",
        "[global.memory]\nunknown_memory = true\n",
        "[global.media_hosting]\nunknown_host = true\n",
        "[global.economy]\nunknown_economy = true\n",
        '[global.emojis]\nunknown_emoji = "x"\n',
        "[global.reply]\nanswer_streaming_enabled = false\n",
        "[global.reply]\nslow_model_blind_test_enabled = true\n",
        '[global.models]\nunknown_tier = "x"\n',
        '[global.models]\nimage = "   "\n',
        '[global.models]\ntriage = { name = "gemini-3.5-flash-lite", effort = "minimal", extra = 1 }\n',
        "[global.models.slow]\nunknown_slow = true\n",
        "[global.models.slow]\ncandidates = []\n",
        '[global.models.slow]\ncandidates = [{ name = "gemini-4-neon", effort = "low" }, { name = "gemini-4-neon", effort = "medium" }]\n',
        '[global.models.slow]\ncandidates = [{ name = "gemini-4-neon", effort = "minimal" }]\n',
        '[global.models.slow]\ndisplay_name = "line1\\nline2"\n',
        "[server]\nunknown_server = true\n",
        "[server.auto_expand]\nbilibili = true\n",
    ],
    ids=[
        "root",
        "global",
        "bot",
        "logging",
        "usage",
        "memory",
        "media-hosting",
        "economy",
        "emojis",
        "reply",
        "removed-blind-test-flag",
        "models-extra",
        "models-blank-name",
        "reasoning-model-extra",
        "slow-extra",
        "slow-empty-candidates",
        "slow-duplicate-candidates",
        "slow-invalid-effort",
        "slow-multiline-display-name",
        "server",
        "auto-expand",
    ],
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

    # Explicit kwargs override both env and TOML, while `model_construct` stays pure static.
    overridden = LLMConfig(api_key="explicit", answer_streaming_enabled=True)
    assert overridden.api_key == "explicit"
    assert overridden.answer_streaming_enabled is True
    assert overridden.file_api_enabled is False

    constructed = LLMConfig.model_construct()
    assert constructed.api_key == ""
    assert constructed.answer_streaming_enabled is True
    assert constructed.file_api_enabled is True


def test_operational_consumers_read_local_config_overrides(
    bot_config_isolated: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Media hosting, economy, and usage-report consumers reflect local `config.toml` overrides."""
    serve_dir = tmp_path / "media"
    serve_dir.mkdir()
    usage_dir = tmp_path / "custom-usage"
    bot_config_isolated.write_text(
        data=(
            f'[global.usage]\ndirectory = "{usage_dir}"\n\n'
            f'[global.media_hosting]\nbase_url = "https://media.example"\nserve_dir = "{serve_dir}"\n\n'
            "[global.economy]\nallow_central_bank_self_approval = true\n"
        ),
        encoding="utf-8",
    )
    clear_bot_config_cache()
    monkeypatch.setattr("discordbot.utils.usage_log._DEFAULT_USAGE_DIR_OVERRIDE", None)

    planner = build_media_delivery_planner()
    assert planner.media_hosting.config.available is True
    assert planner.media_hosting.config.serve_dir == str(serve_dir)

    bot = stub_bot()
    cleanup_cog = MediaCleanupCogs(bot=bot)
    assert cleanup_cog.media_hosting.config.available is True

    economy_cog = EconomyCogs(bot=bot)
    assert economy_cog.economy_config.allow_central_bank_self_approval is True

    parsed = parse_usage_report_args(argv=[])
    assert parsed.dir == str(usage_dir)
