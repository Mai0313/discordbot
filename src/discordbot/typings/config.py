"""Environment-backed core credentials and TOML-backed bot configuration."""

import os
from typing import Any, Final, Literal
from pathlib import Path
import tomllib

from pydantic import Field, BaseModel, ConfigDict, AliasChoices, field_validator
from pydantic_settings import BaseSettings

AutoExpandPlatform = Literal["twitter", "threads", "facebook", "instagram", "douyin"]
ReasoningEffortGrade = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
SlowEffortGrade = Literal["low", "medium", "high"]

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


def _require_non_blank(value: str) -> str:
    """Strips `value` and rejects empty or whitespace-only strings."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("must not be blank")
    return cleaned


class ReplyGlobalConfig(BaseModel):
    """Operator-only AI reply switches loaded from `[global.reply]`."""

    model_config = ConfigDict(extra="forbid")

    file_api_enabled: bool = Field(
        default=True,
        description=(
            "Whether media may reach the answer model as a provider Files API reference; when "
            "false attachments and linked-post images inline as base64 instead, while "
            "linked-post video clips are skipped."
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


class ReasoningModelConfig(BaseModel):
    """Model name and Responses API reasoning effort loaded from `[global.models]`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="LiteLLM model string dispatched on the Responses API.")
    effort: ReasoningEffortGrade = Field(
        ..., description="Reasoning effort passed to the Responses API for this tier."
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Rejects blank model names."""
        return _require_non_blank(value=value)


class SlowModelCandidate(BaseModel):
    """One QA answer model candidate, its router-failure fallback effort, and sampling weight."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="LiteLLM model string dispatched for QA replies.")
    effort: SlowEffortGrade = Field(
        default="low",
        description="Fallback reasoning effort used only when the triage router fails.",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "Relative sampling weight for this candidate; 0 disables it without removing its entry."
        ),
        examples=[1.0, 70, 0],
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Rejects blank candidate model names."""
        return _require_non_blank(value=value)


class SlowModelsConfig(BaseModel):
    """QA answer candidate pool and optional display-label mask loaded from `[global.models.slow]`."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(
        default="",
        description=(
            "Optional single-line display label shown in the system prompt and Discord usage "
            "footer in place of the real model and effort (empty leaves them unmasked)."
        ),
        examples=["", "Gemini Next"],
    )
    candidates: list[SlowModelCandidate] = Field(
        default_factory=lambda: [SlowModelCandidate(name="gemini-4-neon", effort="low")],
        min_length=1,
        description=(
            "Non-empty list of distinct QA candidate models sampled by `weight` per turn; each "
            "entry's `effort` acts as the router-failure fallback for turns drawing that candidate."
        ),
    )

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str) -> str:
        """Rejects multiline display labels so usage-footer parsing stays single-line."""
        if "\n" in value or "\r" in value:
            raise ValueError("display_name must be a single line")
        return value.strip()

    @field_validator("candidates")
    @classmethod
    def _validate_candidates(cls, value: list[SlowModelCandidate]) -> list[SlowModelCandidate]:
        """Rejects duplicate model names and all-zero weight pools."""
        names = [candidate.name for candidate in value]
        if len(set(names)) != len(names):
            raise ValueError("slow.candidates must not contain duplicate model names")
        if sum(candidate.weight for candidate in value) <= 0:
            raise ValueError("slow.candidates must have a positive total weight")
        return value


class ModelsConfig(BaseModel):
    """Runtime model tiers loaded from `[global.models]`."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(
        default="gemini-nano-banana-2.1",
        description="Model string for `images.generate` and `images.edit`.",
    )
    video: str = Field(
        default="gemini-omni-1.2-flash",
        description="Bare Gemini model name for direct `interactions.create` video generation.",
    )
    music: str = Field(
        default="lyria-3.5",
        description="Bare Gemini model name for direct `interactions.create` music generation.",
    )
    tts: str = Field(
        default="gemini-3.8-flash-tts",
        description="Model string for spoken-reply synthesis (`audio.speech`).",
    )
    antigravity: str = Field(
        default="antigravity-preview-09-2026",
        description="Managed agent identifier for direct-to-Google deep research.",
    )
    antigravity_reasoning: str = Field(
        default="gemini-4-hydrogen",
        description="Reasoning model passed as `agent_config.model` for deep research.",
    )
    triage: ReasoningModelConfig = Field(
        default_factory=lambda: ReasoningModelConfig(
            name="gemini-3.5-flash-lite", effort="minimal"
        ),
        description="Model and effort for fixed-schema route/effort/recall triage calls.",
    )
    fast: ReasoningModelConfig = Field(
        default_factory=lambda: ReasoningModelConfig(name="gemini-4-hydrogen", effort="low"),
        description="Model and effort for prompt refinement and media persona replies.",
    )
    memory: ReasoningModelConfig = Field(
        default_factory=lambda: ReasoningModelConfig(name="gemini-4-neon", effort="medium"),
        description="Model and effort for background long-term memory review and consolidation.",
    )
    slow: SlowModelsConfig = Field(
        default_factory=SlowModelsConfig,
        description="QA answer candidate pool and optional display-label mask.",
    )

    @field_validator("image", "video", "music", "tts", "antigravity", "antigravity_reasoning")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        """Rejects blank model identifiers."""
        return _require_non_blank(value=value)


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
    models: ModelsConfig = Field(
        default_factory=ModelsConfig, description="Runtime model catalog configuration."
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
