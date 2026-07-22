"""Shared Discord interaction/message test doubles.

Several cog test modules each grew their own ``FakeInteraction`` / ``FakeUser`` /
``FakeResponse`` families that drifted apart. These are the unified superset:
they satisfy the strictest consumer (the cog smoke tests) and expose the extra
knobs lighter consumers need as optional keyword arguments. Plain classes, not
pydantic, to match the existing test-double style and carry heterogeneous
recorded payloads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Unpack, TypedDict
from datetime import UTC, datetime, timedelta

if TYPE_CHECKING:
    from nextcord import File, Embed, Attachment, AllowedMentions
    from nextcord.ui import View


class DiscordPayload(TypedDict, total=False):
    """Payload captured from fake message, response, and followup sends."""

    content: str | None
    embed: Embed
    embeds: list[Embed]
    file: File
    files: list[File]
    view: View | None
    wait: bool
    ephemeral: bool
    suppress: bool
    allowed_mentions: AllowedMentions
    attachments: list[Attachment]
    message_id: int


class OriginalEditPayload(TypedDict, total=False):
    """Payload captured from fake original interaction edits."""

    content: str
    file: File
    allowed_mentions: AllowedMentions


class FakeUser:
    """Minimal Discord user/member stub recording identity and avatar fields."""

    def __init__(
        self,
        user_id: int = 1,
        name: str = "alice",
        display_name: str = "Alice",
        bot: bool = False,
        avatar_url: str = "https://example.test/avatar.png",
    ) -> None:
        """Initializes identity, avatar, bot flag, and account-age fields."""
        self.id = user_id
        self.name = name
        self.display_name = display_name
        self.bot = bot
        self.mention = f"<@{user_id}>"
        self.display_avatar = SimpleNamespace(url=avatar_url)
        # Commands that surface snowflake-derived account age read created_at; pin
        # it well into the past so the value is never surprising under freezegun.
        self.created_at = datetime.now(tz=UTC) - timedelta(days=365 * 5)


class FakeResponse:
    """Interaction response stub that records sends, edits, and deferral."""

    def __init__(self) -> None:
        """Initializes response state records."""
        self.deferred = False
        self.deferred_ephemeral = False
        self.sent: list[DiscordPayload] = []
        self.edited: list[DiscordPayload] = []

    async def defer(self, ephemeral: bool = False) -> None:
        """Records that the interaction response was deferred."""
        self.deferred = True
        self.deferred_ephemeral = ephemeral

    async def send_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an interaction response message."""
        self.sent.append(kwargs)

    async def edit_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an interaction response edit."""
        self.edited.append(kwargs)

    def is_done(self) -> bool:
        """Returns whether the fake response has already been used."""
        return self.deferred or bool(self.sent)


class FakeFollowup:
    """Interaction followup stub that records sends and edits."""

    def __init__(self) -> None:
        """Initializes recorded followup sends and edits."""
        self.sent: list[DiscordPayload] = []
        self.edited: list[DiscordPayload] = []

    async def send(self, **kwargs: Unpack[DiscordPayload]) -> FakeDiscordMessage:
        """Records the followup payload and returns a fake message."""
        self.sent.append(kwargs)
        return FakeDiscordMessage()

    async def edit_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records a followup message edit payload."""
        self.edited.append(kwargs)


class FakeDiscordMessage:
    """Discord message stub that records mutations."""

    def __init__(self) -> None:
        """Initializes message mutation records."""
        self.id = 1
        self.channel = SimpleNamespace(id=2)
        self.edits: list[DiscordPayload] = []
        self.reactions: list[str] = []
        self.removed: list[tuple[str, FakeUser]] = []
        self.replies: list[DiscordPayload] = []
        self.deleted = False
        self.suppressed = False

    async def edit(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an edit payload and suppress flag."""
        if "suppress" in kwargs:
            self.suppressed = bool(kwargs["suppress"])
        self.edits.append(kwargs)

    async def add_reaction(self, emoji: str) -> None:
        """Records an added reaction."""
        self.reactions.append(emoji)

    async def remove_reaction(self, emoji: str, member: FakeUser) -> None:
        """Records a removed reaction."""
        self.removed.append((emoji, member))

    async def reply(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records a message reply payload."""
        self.replies.append(kwargs)

    async def delete(self) -> None:
        """Records message deletion."""
        self.deleted = True


class FakeInteraction:
    """Interaction stub shared by cog command and view tests."""

    def __init__(
        self,
        user: FakeUser | None = None,
        message: FakeDiscordMessage | object | None = None,
        filesize_limit: int = 25 * 1024 * 1024,
    ) -> None:
        """Initializes user, guild upload limit, response, followup, message, and edit records."""
        self.user = user or FakeUser()
        self.message = message
        self.guild = SimpleNamespace(filesize_limit=filesize_limit)
        self.response = FakeResponse()
        self.followup = FakeFollowup()
        self.edits: list[OriginalEditPayload] = []

    async def edit_original_message(self, **kwargs: Unpack[OriginalEditPayload]) -> None:
        """Records an edit to the deferred original response."""
        self.edits.append(kwargs)
