"""Shared Discord interaction/message test doubles.

Each double answers what its strictest consumer reads and takes the knobs a lighter consumer
needs as optional keyword arguments, so a test extends one of these rather than growing its
own. Plain classes, not pydantic, to match the test-double style and carry heterogeneous
recorded payloads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Unpack, TypedDict
from datetime import UTC, datetime, timedelta

from tests.helpers.casting import make_not_found

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
    mention_author: bool
    attachments: list[Attachment]


class OriginalEditPayload(TypedDict, total=False):
    """Payload captured from fake original interaction edits."""

    content: str
    embed: Embed
    embeds: list[Embed]
    view: View | None
    file: File
    files: list[File]
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
        # Commands that surface snowflake-derived account age read created_at (`/balance`
        # renders `now - created_at`); pinning it years back keeps that a plausible number
        # instead of the zero a stub created "now" would show.
        self.created_at = datetime.now(tz=UTC) - timedelta(days=365 * 5)


class FakeResponse:
    """Interaction response stub that records sends, edits, and deferral."""

    def __init__(self, slash_command: bool = False) -> None:
        """Initializes response state records.

        `slash_command` makes a defer post the "thinking" placeholder a slash command's does,
        which the next followup or edit of the original fills; a component's defer posts nothing.
        """
        self.slash_command = slash_command
        self.deferred = False
        self.deferred_ephemeral = False
        self.placeholder_pending = False
        self.sent: list[DiscordPayload] = []
        self.edited: list[DiscordPayload] = []

    async def defer(self, ephemeral: bool = False) -> None:
        """Records that the interaction response was deferred."""
        self.deferred = True
        self.deferred_ephemeral = ephemeral
        self.placeholder_pending = self.slash_command

    async def send_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an interaction response message."""
        self.sent.append(kwargs)

    async def edit_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an interaction response edit."""
        self.edited.append(kwargs)

    async def send_modal(self, modal: object) -> None:
        """Accepts a modal opened in response to the interaction."""
        del modal

    def is_done(self) -> bool:
        """Returns whether the fake response has already been used."""
        return self.deferred or bool(self.sent)


class FakeFollowup:
    """Interaction followup stub that records sends."""

    def __init__(self, response: FakeResponse) -> None:
        """Initializes recorded followup sends against the response they follow."""
        self.sent: list[DiscordPayload] = []
        self._response = response

    async def send(self, **kwargs: Unpack[DiscordPayload]) -> FakeDiscordMessage:
        """Records the followup payload and returns a fake message.

        The first followup after a slash command's defer fills its placeholder, and Discord keeps
        the defer's ephemeral flag over this one's, so that is the flag recorded.
        """
        if self._response.placeholder_pending:
            self._response.placeholder_pending = False
            kwargs["ephemeral"] = self._response.deferred_ephemeral
        self.sent.append(kwargs)
        return FakeDiscordMessage()


class FakeDiscordMessage:
    """Discord message stub that records mutations."""

    def __init__(
        self,
        author: FakeUser | None = None,
        content: str = "",
        guild: FakeGuild | SimpleNamespace | None = None,
    ) -> None:
        """Initializes the fields a listener reads and the message mutation records.

        `guild` defaults to None, which a real message carries only in a DM. Setting
        `edit_failure` makes every later edit raise it, the way Discord refuses one.
        """
        self.author = author or FakeUser()
        self.content = content
        self.guild = guild
        self.id = 1
        self.channel = SimpleNamespace(id=2)
        self.edit_failure: Exception | None = None
        self.edits: list[DiscordPayload] = []
        self.reactions: list[str] = []
        self.removed: list[tuple[str, FakeUser]] = []
        self.replies: list[DiscordPayload] = []
        self.reply_messages: list[FakeDiscordMessage] = []
        self.deleted = False
        self.suppressed = False

    async def edit(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an edit payload and suppress flag, or raises `edit_failure` when set."""
        if self.edit_failure is not None:
            raise self.edit_failure
        if "suppress" in kwargs:
            self.suppressed = bool(kwargs["suppress"])
        self.edits.append(kwargs)

    async def add_reaction(self, emoji: str) -> None:
        """Records an added reaction."""
        self.reactions.append(emoji)

    async def remove_reaction(self, emoji: str, member: FakeUser) -> None:
        """Records a removed reaction."""
        self.removed.append((emoji, member))

    async def reply(self, **kwargs: Unpack[DiscordPayload]) -> FakeDiscordMessage:
        """Records a reply payload and answers with the message it created.

        A real `Message.reply` hands back the posted message, and the expansion cogs keep
        theirs to edit later. A double answering None turns that into an `AttributeError`
        deep inside the cog instead of the assertion the test came for.
        """
        self.replies.append(kwargs)
        posted = FakeDiscordMessage()
        self.reply_messages.append(posted)
        return posted

    async def delete(self) -> None:
        """Records message deletion."""
        self.deleted = True


def expansion_payload(message: FakeDiscordMessage) -> DiscordPayload:
    """Returns what an expansion cog edited onto the placeholder it replied with.

    Every link expansion claims its reply slot before it has anything to show, so the card a
    test is looking for is an edit of the first reply rather than a reply of its own.
    """
    placeholder = message.reply_messages[0]
    assert placeholder.edits, "the expansion never reached its placeholder"
    return placeholder.edits[-1]


def placeholder_withdrawn(message: FakeDiscordMessage) -> bool:
    """Whether the placeholder was taken back with nothing delivered onto it.

    That is what a failed expansion leaves: the reaction says what happened and the channel
    keeps no trace of a card that never came. The reply count is part of it, so a cog that
    withdrew the placeholder and then explained itself in a second message still fails.
    """
    placeholder = message.reply_messages[0]
    return len(message.replies) == 1 and placeholder.deleted and not placeholder.edits


class FakeGuild:
    """Guild stub that answers the member lookups a real `nextcord.Guild` always answers.

    A real Guild has `get_member` and `fetch_member` unconditionally, so production calls them
    unguarded and this double owes both. The bot runs without the members intent, so an
    uncached member is the ordinary case: this answers None from the cache and a `NotFound`
    from the fetch, which `guild_avatar_url` handles by falling back to the global avatar. A
    test wanting the guild-avatar branch hands in a member of its own.
    """

    def __init__(
        self,
        filesize_limit: int = 25 * 1024 * 1024,
        guild_id: int = 100,
        guild_name: str = "test guild",
    ) -> None:
        """Initializes the upload limit and identity a guild is read for."""
        self.filesize_limit = filesize_limit
        self.id = guild_id
        self.name = guild_name

    def get_member(self, user_id: int) -> None:
        """Answers the member cache, which is empty without the members intent."""
        del user_id

    async def fetch_member(self, user_id: int) -> None:
        """Answers the REST lookup the way Discord does for a member this guild has not got."""
        del user_id
        raise make_not_found(message="member not found")


class FakeInteraction:
    """Interaction stub shared by cog command and view tests."""

    def __init__(  # noqa: PLR0913 -- optional knobs for the strictest consumer
        self,
        user: FakeUser | None = None,
        message: FakeDiscordMessage | object | None = None,
        filesize_limit: int = 25 * 1024 * 1024,
        in_guild: bool = True,
        guild_id: int = 100,
        guild_name: str = "test guild",
        channel_id: int = 200,
        administrator: bool = False,
        custom_id: str | None = None,
        slash_command: bool = False,
    ) -> None:
        """Initializes user, origin, guild upload limit, response, followup, and edit records.

        `custom_id` names the pressed control for a component interaction, which a view's
        `interaction_check` reads off `data`; a slash command carries no component payload.
        `slash_command` gives a defer the placeholder only a slash command's posts.
        """
        self.user = user or FakeUser()
        self.message = message
        self.data: dict[str, str] | None = {"custom_id": custom_id} if custom_id else None
        # Union rather than `FakeGuild`: a few tests hand in a stricter guild of their own (one
        # whose `fetch_member` asserts it is never reached), and the real attribute is a Guild
        # this package deliberately does not model in full.
        self.guild: FakeGuild | SimpleNamespace | None = (
            FakeGuild(filesize_limit=filesize_limit, guild_id=guild_id, guild_name=guild_name)
            if in_guild
            else None
        )
        # Both of these come off the interaction payload rather than the cache, which is why
        # production reads them instead of `guild`: a user-installed command in a server the
        # bot was never added to resolves no guild at all but still carries these two.
        self.guild_id: int | None = guild_id if in_guild else None
        self.permissions = SimpleNamespace(administrator=administrator)
        self.channel_id = channel_id
        self.response = FakeResponse(slash_command=slash_command)
        self.followup = FakeFollowup(response=self.response)
        self.edit_failure: Exception | None = None
        self.edits: list[OriginalEditPayload] = []
        # A live token by default; set to model one past Discord's 15-minute life.
        self.expired = False
        self.delete_failure: Exception | None = None
        self.original_deleted = False

    def is_expired(self) -> bool:
        """Answers whether the token has outlived its life, as `expired` says."""
        return self.expired

    async def delete_original_message(self) -> None:
        """Records a delete of the original response, or raises `delete_failure` when set.

        Recorded here rather than on the message, so a test tells the token's delete from the
        channel's.
        """
        if self.delete_failure is not None:
            raise self.delete_failure
        self.original_deleted = True

    async def edit_original_message(self, **kwargs: Unpack[OriginalEditPayload]) -> None:
        """Records an edit to the deferred original response, or raises `edit_failure` when set.

        A pressed control's original response is the message the control sits on, so a
        `FakeDiscordMessage` handed in shows the edit too. Its own `edit_failure` stays out of
        it: that is the channel refusing, and the token edits the message whatever the channel
        allows.
        """
        if self.edit_failure is not None:
            raise self.edit_failure
        self.response.placeholder_pending = False
        self.edits.append(kwargs)
        if isinstance(self.message, FakeDiscordMessage):
            self.message.edits.append(DiscordPayload(**kwargs))
