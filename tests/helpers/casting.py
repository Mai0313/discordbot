"""Typed boundaries for handing test doubles to production signatures.

The suite's fakes (``FakeMessage``, ``FakeInteraction``, recorder clients, ...)
stay plain classes so tests can read their recorded attributes back, while
production signatures take the real nextcord/OpenAI types. These adapters
centralize the one ``cast`` each boundary needs, so call sites stay clean and
the cast target stays consistent. The ``object`` parameters are deliberate:
every fake family across the test modules funnels through the same adapter.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from nextcord import Message, NotFound, Interaction
from nextcord.ext import commands

from discordbot.utils.media_delivery import MediaHostingConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from google import genai
    from aiohttp import ClientResponse
    from nextcord import Guild
    from google.genai.interactions import InteractionSSEEvent

    from discordbot import cli
    from discordbot.utils.avatars import AvatarUser


def as_message(fake: object) -> Message:
    """Views a message double as the nextcord Message a production signature expects."""
    return cast("Message", fake)


def as_interaction(fake: object) -> Interaction[Any]:
    """Views an interaction double as the nextcord Interaction a production signature expects."""
    return cast("Interaction[Any]", fake)


def as_bot(fake: object) -> commands.Bot:
    """Views a bot double as the commands.Bot a cog constructor or ``setup`` expects."""
    return cast("commands.Bot", fake)


def as_discord_bot(fake: object) -> "cli.DiscordBot":
    """Views a bot double as cli.DiscordBot for its own unbound method calls."""
    return cast("cli.DiscordBot", fake)


def as_command_context(fake: object) -> "commands.Context[commands.Bot]":
    """Views a context double as the commands.Context a command handler expects."""
    return cast("commands.Context[commands.Bot]", fake)


def as_avatar_user(fake: object) -> "AvatarUser":
    """Views a fake user double as the AvatarUser protocol ``guild_avatar_url`` expects."""
    return cast("AvatarUser", fake)


def as_guild(fake: object) -> "Guild | None":
    """Views a fake guild double as the nextcord Guild ``guild_avatar_url`` expects."""
    return cast("Guild | None", fake)


def step_dicts(steps: object) -> list[dict[str, Any]]:
    """Views a typed input/step list as plain dicts for key-level assertions.

    The Responses/Interactions input lists are unions of TypedDicts; indexing a
    key only some members carry is an invalid-key error even when the test just
    built the concrete member. Assertions only read keys, so the plain-dict view
    is safe.
    """
    return cast("list[dict[str, Any]]", steps)


def as_client(fake: object) -> "genai.Client":
    """Views a fake Gemini client double as the genai.Client a production signature expects."""
    return cast("genai.Client", fake)


def make_stub_gemini_client() -> "genai.Client":
    """Builds an empty Gemini client stub for paths that never call through it."""
    return as_client(fake=SimpleNamespace())


def as_interaction_event_stream(fake: object) -> "AsyncIterator[InteractionSSEEvent]":
    """Views a fabricated SSE stream double as the SDK event stream a signature expects.

    Production discriminates on ``.event_type``, not isinstance, so ``SimpleNamespace``
    events are safe.
    """
    return cast("AsyncIterator[InteractionSSEEvent]", fake)


def make_not_found(message: str = "missing") -> NotFound:
    """Builds the ``NotFound`` nextcord raises for a deleted Discord entity.

    The constructor only reads ``status``/``reason`` off the response, so a
    minimal stub stands in for the aiohttp response.
    """
    response = cast("ClientResponse", SimpleNamespace(status=404, reason="Not Found"))
    return NotFound(response=response, message=message)


def make_media_hosting_config(
    enabled: bool,
    base_url: str = "",
    serve_dir: str = "",
    max_bytes: int | None = None,
    retention_hours: float | None = None,
) -> MediaHostingConfig:
    """Builds a MediaHostingConfig through its env-alias names.

    ``model_validate`` keeps the alias spelling type-clean (the alias kwargs on
    ``__init__`` are invisible to a checker without a pydantic plugin) and stays
    hermetic: unlike ``__init__``, it never merges environment values in.
    """
    payload: dict[str, object] = {
        "MEDIA_HOSTING_ENABLED": enabled,
        "MEDIA_HOSTING_BASE_URL": base_url,
        "MEDIA_HOSTING_SERVE_DIR": serve_dir,
    }
    if max_bytes is not None:
        payload["MEDIA_HOSTING_MAX_BYTES"] = max_bytes
    if retention_hours is not None:
        payload["MEDIA_HOSTING_RETENTION_HOURS"] = retention_hours
    return MediaHostingConfig.model_validate(payload)
