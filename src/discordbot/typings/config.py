"""Environment-backed core credentials and TOML-backed bot configuration."""

import os
from typing import Any, Final, Literal
from pathlib import Path
import tomllib

from pydantic import Field, BaseModel, ConfigDict, AliasChoices
from pydantic_settings import BaseSettings

AutoExpandPlatform = Literal["twitter", "threads", "facebook", "instagram", "douyin"]

DEFAULT_CONFIG_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "config.toml"
LOCAL_CONFIG_PATH: Path = Path("./data/config.toml")

_cached_bot_config: "BotConfig | None" = None


class DiscordConfig(BaseSettings):
    """Discord gateway credentials loaded from environment variables."""

    discord_bot_token: str = Field(
        ...,
        description="The token from Discord the bot authenticates its gateway session with.",
        examples=["MTEz-..."],
        validation_alias=AliasChoices("DISCORD_BOT_TOKEN"),
    )


class LoggingConfig(BaseModel):
    """Console and log-file verbosity loaded from `[global.logging]`."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["trace", "debug", "info", "notice", "warn", "warning", "error", "fatal"] = (
        Field(
            default="debug",
            description=(
                "Lowest severity written to the console and to ./data/logs. Defaults to debug so the "
                "log file keeps the full trace; raise it to info on a deployment that only wants "
                "outcomes."
            ),
            examples=["debug", "info"],
        )
    )


class UsageLogConfig(BaseModel):
    """Append-only monthly usage-recording settings loaded from `[global.usage]`."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Whether feature usage is recorded at all; when false no file is created.",
        examples=[True],
    )
    directory: str = Field(
        default="./data/usage",
        description="Directory the monthly usage record files are written into.",
        examples=["./data/usage"],
    )


class MemoryConfig(BaseModel):
    """Deployment settings for long-term memory storage loaded from `[global.memory]`."""

    model_config = ConfigDict(extra="forbid")

    git_enabled: bool = Field(
        default=True,
        description=(
            "Whether memory changes are committed to the store's git repository. Best-effort "
            "either way: the bot never creates that repository, and a missing one disables "
            "this silently."
        ),
    )


class MediaHostingConfig(BaseModel):
    """Configuration for the external static media host loaded from `[global.media_hosting]`."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description=(
            "Kill-switch for hosting oversized media externally and linking it; when false the "
            "URL fallback is inert and every oversize item takes the call site's host-free path."
        ),
    )
    base_url: str = Field(
        default="",
        description="Public base URL the external static host serves from.",
        examples=["https://media.mai0313.com"],
    )
    serve_dir: str = Field(
        default="",
        description=(
            "In-container directory (bind-mounted from the host) hosted files are written into; "
            "nginx serves those same files from the host path."
        ),
        examples=["/mnt/share/media"],
    )
    max_bytes: int = Field(
        default=8 * 1024**3,
        description="Soft cap on total hosted bytes; oldest files are evicted past it (<=0 disables).",
    )
    retention_hours: float = Field(
        default=168.0,
        description="Hosted files older than this many hours are reaped, cap or not (<=0 disables).",
    )

    @property
    def available(self) -> bool:
        """Returns whether hosting can run: enabled and both base URL and serve dir are set."""
        return self.enabled and bool(self.base_url.strip()) and bool(self.serve_dir.strip())

    @property
    def cleanup_enabled(self) -> bool:
        """Returns whether the cleanup loop should run: hosting available and at least one cap is set."""
        return self.available and (self.max_bytes > 0 or self.retention_hours > 0)


class EconomyConfig(BaseModel):
    """Economy feature settings loaded from `[global.economy]`."""

    model_config = ConfigDict(extra="forbid")

    allow_central_bank_self_approval: bool = Field(
        default=False,
        description="Allow central-bank borrowers to approve their own loan requests for local testing.",
        examples=[False],
    )


class BotAppearanceConfig(BaseModel):
    """Bot-wide presence settings loaded from `[global.bot]`."""

    model_config = ConfigDict(extra="forbid")

    activity: str = Field(
        default="Discord",
        description="Playing-activity label shown in the bot's Discord presence.",
        examples=["Discord"],
    )


class EmojiConfig(BaseModel):
    """Discord reaction and status emojis loaded from `[global.emojis]`."""

    model_config = ConfigDict(extra="forbid")

    facebook: str = Field(
        default="📘",
        description="Reaction emoji marking a Facebook link.",
        examples=["📘", "<:facebook:1234567890123456789>"],
    )
    instagram: str = Field(
        default="📷",
        description="Reaction emoji marking an Instagram link.",
        examples=["📷", "<:instagram:1234567890123456789>"],
    )
    threads: str = Field(
        default="🧵",
        description="Reaction emoji marking a Threads link.",
        examples=["🧵", "<:threads:1234567890123456789>"],
    )
    twitter: str = Field(
        default="🐦",
        description="Reaction emoji marking a Twitter/X link.",
        examples=["🐦", "<:twitter:1234567890123456789>"],
    )
    douyin: str = Field(
        default="🎵",
        description="Reaction emoji marking a Douyin link.",
        examples=["🎵", "<:douyin:1234567890123456789>"],
    )
    bilibili: str = Field(
        default="📺",
        description="Reaction emoji marking a Bilibili link.",
        examples=["📺", "<:bilibili:1234567890123456789>"],
    )
    done: str = Field(
        default="✅",
        description="Reaction emoji marking a completed reply or link expansion.",
        examples=["✅", "<:greencheck:1234567890123456789>"],
    )
    failed: str = Field(
        default="❌",
        description="Reaction emoji marking a failed reply or link expansion.",
        examples=["❌", "<:redcross:1234567890123456789>"],
    )
    routing: str = Field(
        default="🔀",
        description="Reaction emoji shown while routing a reply turn.",
        examples=["🔀", "<:flowchart:1234567890123456789>"],
    )
    image: str = Field(
        default="🖼️",
        description="Reaction emoji shown when generating an image.",
        examples=["🖼️", "<:image:1234567890123456789>"],
    )
    video: str = Field(
        default="🎬",
        description="Reaction emoji shown when generating a video.",
        examples=["🎬", "<:video:1234567890123456789>"],
    )
    answer: str = Field(
        default="💬",
        description="Reaction and thinking-preview emoji shown while streaming an answer.",
        examples=["💬", "<:message:1234567890123456789>"],
    )
    voice: str = Field(
        default="🗣️",
        description="Reaction emoji shown when generating an inline voice clip.",
        examples=["🗣️", "<:voice:1234567890123456789>"],
    )
    youtube: str = Field(
        default="▶️",
        description="Reaction emoji shown when watching a linked YouTube video.",
        examples=["▶️", "<:youtube:1234567890123456789>"],
    )
    research: str = Field(
        default="🔬",
        description="Reaction emoji placed on the anchor message of a deep-research thread.",
        examples=["🔬", "<:dino:1234567890123456789>"],
    )


class ReplyGlobalConfig(BaseModel):
    """Operator-only AI reply switches loaded from `[global.reply]`."""

    model_config = ConfigDict(extra="forbid")

    file_api_enabled: bool = Field(
        default=True,
        description=(
            "Whether media may reach the answer model as a provider Files API reference; when "
            "false attachments inline as base64 instead and link media is not uploaded at all."
        ),
    )
    streaming_enabled: bool = Field(
        default=True,
        description=(
            "Whether QA replies and media-persona follow-ups stream live preview edits onto "
            "Discord; when false they fetch complete non-streaming responses and post once."
        ),
    )
    feedback_enabled: bool = Field(
        default=True,
        description=(
            "Whether finalized replies attach persistent feedback/regenerate buttons and persist "
            "turn snapshots in `reply.db`."
        ),
    )
    slow_model_blind_test_enabled: bool = Field(
        default=False,
        description=(
            "Whether QA turns randomly sample `slow_model` from the blind-test candidate pool and "
            "mask the visible model label as `blind-test-model`."
        ),
    )


class GlobalConfig(BaseModel):
    """Bot-wide configuration loaded from `[global]` (not overridable per server)."""

    model_config = ConfigDict(extra="forbid")

    bot: BotAppearanceConfig = Field(
        default_factory=BotAppearanceConfig, description="Bot presence settings."
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Console and log-file verbosity settings."
    )
    usage: UsageLogConfig = Field(
        default_factory=UsageLogConfig, description="Append-only monthly usage-recording settings."
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="Long-term memory deployment settings."
    )
    media_hosting: MediaHostingConfig = Field(
        default_factory=MediaHostingConfig,
        description="External static media hosting configuration.",
    )
    economy: EconomyConfig = Field(
        default_factory=EconomyConfig, description="Economy feature settings."
    )
    emojis: EmojiConfig = Field(
        default_factory=EmojiConfig, description="Reaction and status emoji markers."
    )
    reply: ReplyGlobalConfig = Field(
        default_factory=ReplyGlobalConfig, description="Operator-only AI reply switches."
    )


class AutoExpandConfig(BaseModel):
    """Default per-platform link auto-expansion flags loaded from `[server.auto_expand]`."""

    model_config = ConfigDict(extra="forbid")

    twitter: bool = Field(
        default=True, description="Whether Twitter/X post links auto-expand by default."
    )
    threads: bool = Field(
        default=True, description="Whether Threads post links auto-expand by default."
    )
    facebook: bool = Field(
        default=True, description="Whether Facebook post links auto-expand by default."
    )
    instagram: bool = Field(
        default=True, description="Whether Instagram post links auto-expand by default."
    )
    douyin: bool = Field(
        default=True, description="Whether Douyin post links auto-expand by default."
    )

    def enabled_for(self, platform: AutoExpandPlatform) -> bool:
        """Returns the default auto-expansion flag for `platform`.

        Args:
            platform (AutoExpandPlatform): Expansion platform key.
        """
        return bool(getattr(self, platform))


class ServerDefaultsConfig(BaseModel):
    """Default per-server settings loaded from `[server]` (overridable per guild)."""

    model_config = ConfigDict(extra="forbid")

    auto_expand: AutoExpandConfig = Field(
        default_factory=AutoExpandConfig,
        description="Default per-platform link auto-expansion switches.",
    )


class BotConfig(BaseModel):
    """Root TOML configuration combining packaged defaults and optional local overrides."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    global_config: GlobalConfig = Field(
        default_factory=GlobalConfig,
        alias="global",
        description="Bot-wide settings that apply to the whole deployment.",
    )
    server: ServerDefaultsConfig = Field(
        default_factory=ServerDefaultsConfig,
        description="Default per-server settings overridable per guild via `/settings`.",
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Returns a new dict with nested dicts in `override` recursively merged onto `base`."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base=existing, override=value)
        else:
            merged[key] = value
    return merged


def clear_bot_config_cache() -> None:
    """Clears the process-wide cached `BotConfig` so the next load re-reads TOML files."""
    global _cached_bot_config  # noqa: PLW0603 -- process-wide config cache reset seam
    _cached_bot_config = None


def load_bot_config(local_path: Path | None = None) -> BotConfig:
    """Loads `src/discordbot/config.toml` deep-merged with optional local overrides.

    When `local_path` is omitted, reads `LOCAL_CONFIG_PATH` (`./data/config.toml`) if present
    (unless `DISCORDBOT_LOCAL_CONFIG_DISABLED=1`) and caches the validated result for the process.
    Passing `local_path` bypasses the cache and reads that path directly when it exists.

    Args:
        local_path (Path | None): Explicit override TOML file path, or None for `LOCAL_CONFIG_PATH`.
    """
    global _cached_bot_config  # noqa: PLW0603 -- process-wide config cache
    if local_path is None and _cached_bot_config is not None:
        return _cached_bot_config

    raw: dict[str, Any] = tomllib.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    override_path: Path | None
    if local_path is not None:
        override_path = local_path
    elif os.environ.get("DISCORDBOT_LOCAL_CONFIG_DISABLED") == "1":
        override_path = None
    else:
        override_path = LOCAL_CONFIG_PATH

    if override_path is not None and override_path.is_file():
        override = tomllib.loads(override_path.read_text(encoding="utf-8"))
        raw = _deep_merge(base=raw, override=override)

    config = BotConfig.model_validate(obj=raw)
    if local_path is None:
        _cached_bot_config = config
    return config
