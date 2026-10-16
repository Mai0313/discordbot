"""A `/ask` interaction and the real `nextcord.Message` it synthesizes.

These fakes are local rather than pulled from `tests/helpers/discord_mocks.py` because this
route needs something that package deliberately does not model: real `nextcord.Message` objects,
built over a connection state, on a channel the bot is not a member of. Everything else about a
`/ask` turn is the ordinary pipeline, which the rest of `tests/cogs/gen_reply/` covers.
"""

from types import SimpleNamespace
from typing import Any
from datetime import datetime, timedelta

from nextcord import Message, ChannelType, PartialMessageable
from nextcord.enums import InteractionContextType
from nextcord.utils import utcnow

from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.ask_message import build_ask_message

# A real Discord snowflake, so `Message.created_at` resolves to a real moment rather than 1970.
ASK_SNOWFLAKE = 1517561877973045349
BOT_USER_ID = 999
ASKER_ID = 4242
CHANNEL_ID = 777
GUILD_ID = 31337


class _FakeState:
    """The slice of `nextcord.ConnectionState` a synthesized message actually reads."""

    # Stored by every `Attachment` for its own download path, which nothing here follows.
    http = None

    def _get_guild(self, guild_id: int | None) -> None:
        """The bot is in no guild here, which is the whole premise of this route."""
        del guild_id

    def _get_guild_channel(self, data: object) -> tuple[None, None]:
        """Only reached for a resolved reference, which a `/ask` message never carries."""
        del data
        return None, None


class _FakeFollowup:
    """Records follow-up POSTs and hands back a message that can be edited."""

    def __init__(self) -> None:
        """Initializes the recorded sends."""
        self.sent: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 -- passthrough Discord payload
        """Records one follow-up and returns a stand-in for the message it created."""
        self.sent.append(kwargs)
        return SimpleNamespace(id=len(self.sent), content=kwargs.get("content"))


class _FakeAskInteraction:
    """The interaction surface of a user-installed `/ask`, with nothing the route never reads."""

    def __init__(
        self, guild_id: int | None = GUILD_ID, context: InteractionContextType | None = None
    ) -> None:
        """Initializes the invocation's identity, its channel and its response records."""
        self.id = ASK_SNOWFLAKE
        self.guild_id = guild_id
        self.context = context or (
            InteractionContextType.guild if guild_id is not None else InteractionContextType.bot_dm
        )
        self.channel_id = CHANNEL_ID
        self.user = SimpleNamespace(id=ASKER_ID, name="asker", display_name="Asker")
        self.data: dict[str, Any] | None = None
        self._state = _FakeState()
        self.channel = PartialMessageable(
            state=self._state,  # ty: ignore[invalid-argument-type] -- the state slice a partial channel reads
            id=CHANNEL_ID,
            type=ChannelType.text if guild_id is not None else ChannelType.private,
        )
        self.followup = _FakeFollowup()
        self.edits: list[dict[str, Any]] = []
        self.deferred = False
        # Settable so a test can move the token's window without inventing a snowflake; the real
        # property derives it from `id` the same way.
        self.created_at = utcnow()
        self.response = SimpleNamespace(defer=self._defer, is_done=lambda: self.deferred)
        self.client = SimpleNamespace(
            user=SimpleNamespace(id=BOT_USER_ID, name="pocat", discriminator="0")
        )

    @property
    def expires_at(self) -> datetime:
        """When Discord invalidates this token, mirroring `nextcord.Interaction.expires_at`.

        Fifteen minutes once the response is deferred and three seconds before it, which is the
        same split the real property makes. Modelled rather than stubbed to a fixed number,
        because that split is exactly what `delivery_budget_seconds` relies on being told.
        """
        return self.created_at + (
            timedelta(minutes=15) if self.response.is_done() else timedelta(seconds=3)
        )

    async def _defer(self) -> None:
        """Records that the three-second window was answered before any work started."""
        self.deferred = True

    async def edit_original_message(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 -- passthrough Discord payload
        """Records an edit of the deferred response and returns its message."""
        self.edits.append(kwargs)
        return SimpleNamespace(id=1, content=kwargs.get("content"))


# One attachment option's resolved payload, as Discord sends it back in `interaction.data`.
_CAT_PNG: dict[str, Any] = {
    "id": "9",
    "filename": "cat.png",
    "size": 1024,
    "url": "https://cdn.example/cat.png",
    "proxy_url": "https://cdn.example/cat.png",
    "content_type": "image/png",
}


def _interaction(**kwargs: Any) -> Any:  # noqa: ANN401 -- the fake stands in for a generic Interaction
    """Builds the fake invocation, untyped so it can stand in for `Interaction[Bot]`."""
    return _FakeAskInteraction(**kwargs)


def _ask_message(interaction: Any) -> Message:  # noqa: ANN401 -- see `_interaction`
    """The message the pipeline would answer for this invocation."""
    return build_ask_message(interaction=interaction, question="在幹嘛")


def _toolkit(interaction: Any) -> ReplyToolkit:  # noqa: ANN401 -- see `_interaction`
    """A toolkit whose clients no test here ever calls through."""
    return ReplyToolkit(bot=interaction.client, openai_client=SimpleNamespace(), gemini_api_key="")
