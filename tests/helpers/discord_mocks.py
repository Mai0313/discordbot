"""Shared Discord interaction/message test doubles.

Several cog test modules each grew their own ``FakeInteraction`` / ``FakeUser`` /
``FakeResponse`` families that drifted apart. These are the unified superset:
they satisfy the strictest consumer (the cog smoke tests) and expose the extra
knobs lighter consumers need as optional keyword arguments. Plain classes, not
pydantic, to match the existing test-double style and carry heterogeneous
recorded payloads.

Every double here records rather than simulates: each call appends its keyword payload to a list
and returns, so a test asserts on what the cog sent instead of on a mock's configured behavior.
Nothing validates a payload the way Discord would, and nothing refuses a second response the way
nextcord does, so a path that answers twice is recorded twice rather than raising.

`DiscordPayload` and `OriginalEditPayload` are what keep those recordings readable without `Any`:
they type the `**kwargs` each recorder takes, so `sent[0]["content"]` stays a checked index rather
than a dict lookup on an untyped mapping. Both are `total=False` because a caller passes whatever
subset it needs, and `edit_original_message` gets its own because it records the narrower edit
payload the cogs hand that call.

A double owns only the fields it records, so a test grafts on whatever else the production path
under it reads (`author`, `content`, `guild`) by writing into the instance `__dict__`. Handing one
to a production signature goes through `tests/helpers/casting.py` (`as_message`, `as_interaction`),
which owns the single cast each boundary needs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Unpack, TypedDict
from datetime import UTC, datetime, timedelta

if TYPE_CHECKING:
    from nextcord import File, Embed, Attachment, AllowedMentions
    from nextcord.ui import View


class DiscordPayload(TypedDict, total=False):
    """Payload captured from fake message, response, and followup sends and edits."""

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
        """Builds a user double carrying the identity fields an embed or a listener reads back.

        `mention` is derived from `user_id` rather than taken separately, so a test asserting on
        rendered `<@id>` text cannot disagree with the id the same double reports.

        Args:
            user_id (int): Value exposed as `id`, and the id rendered into `mention`.
            name (str): Discord account name.
            display_name (str): Per-guild display name, which the embeds render.
            bot (bool): Whether this user is a bot; the message listeners filter on it.
            avatar_url (str): Url exposed as `display_avatar.url`.
        """
        self.id = user_id
        self.name = name
        self.display_name = display_name
        self.bot = bot
        self.mention = f"<@{user_id}>"
        self.display_avatar = SimpleNamespace(url=avatar_url)
        # `/balance` renders the account age as `now - created_at`, so this is stamped five years
        # back: stamping it at `now` would render an age of zero days.
        self.created_at = datetime.now(tz=UTC) - timedelta(days=365 * 5)


class FakeResponse:
    """Interaction response stub that records sends, edits, and deferral."""

    def __init__(self) -> None:
        """Starts a response double with nothing deferred, sent or edited."""
        self.deferred = False
        self.deferred_ephemeral = False
        self.sent: list[DiscordPayload] = []
        self.edited: list[DiscordPayload] = []

    async def defer(self, ephemeral: bool = False) -> None:
        """Records the deferral and the ephemeral flag it carried.

        The ephemeral flag is recorded on its own so a test can assert a command deferred
        privately. A second defer overwrites both here, where nextcord raises
        `InteractionResponded`.

        Args:
            ephemeral (bool): Whether the deferred response was requested ephemeral.
        """
        self.deferred = True
        self.deferred_ephemeral = ephemeral

    async def send_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an interaction response message.

        Args:
            **kwargs (Unpack[DiscordPayload]): The payload the caller responded with.
        """
        self.sent.append(kwargs)

    async def edit_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an interaction response edit.

        Args:
            **kwargs (Unpack[DiscordPayload]): The payload the caller edited with.
        """
        self.edited.append(kwargs)

    def is_done(self) -> bool:
        """Whether the response has been used, which production branches on before answering.

        Reads `deferred` and `sent` only, so a recorded `edit_message` leaves it False; nextcord
        counts an edit as the response and would return True.

        Returns:
            True once the double has been deferred or sent to.
        """
        return self.deferred or bool(self.sent)


class FakeFollowup:
    """Interaction followup stub that records sends and edits."""

    def __init__(self) -> None:
        """Starts a followup double with nothing sent or edited."""
        self.sent: list[DiscordPayload] = []
        self.edited: list[DiscordPayload] = []

    async def send(self, **kwargs: Unpack[DiscordPayload]) -> FakeDiscordMessage:
        """Records the followup payload and hands back a message double.

        The `wait=True` call sites keep the message they get back and edit it later, so something
        has to come back. Each call mints a fresh double rather than reusing one, so a test that
        needs the sent message must hold on to what this returned.

        Args:
            **kwargs (Unpack[DiscordPayload]): The payload the caller sent.

        Returns:
            A new message double standing in for the followup that was sent.
        """
        self.sent.append(kwargs)
        return FakeDiscordMessage()

    async def edit_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records a followup message edit payload.

        Args:
            **kwargs (Unpack[DiscordPayload]): The payload the caller edited with, which carries
                the `message_id` naming which earlier followup is being edited.
        """
        self.edited.append(kwargs)


class FakeDiscordMessage:
    """Discord message stub that records mutations."""

    def __init__(self) -> None:
        """Starts a message double with fixed ids and empty mutation records.

        `id` and `channel.id` are fixed stand-ins for the paths that read them (scheduled cleanup
        keys on both) rather than values a test asserts on; an author, content or guild is grafted
        on per test by whichever production path needs one.
        """
        self.id = 1
        self.channel = SimpleNamespace(id=2)
        self.edits: list[DiscordPayload] = []
        self.reactions: list[str] = []
        self.removed: list[tuple[str, FakeUser]] = []
        self.replies: list[DiscordPayload] = []
        self.deleted = False
        self.suppressed = False

    async def edit(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an edit payload, mirroring `suppress` onto its own flag.

        The link-expansion cogs suppress the source message's own preview, and what a test cares
        about there is the final state rather than which edit in the list carried it.

        Args:
            **kwargs (Unpack[DiscordPayload]): The payload the caller edited with.
        """
        if "suppress" in kwargs:
            self.suppressed = bool(kwargs["suppress"])
        self.edits.append(kwargs)

    async def add_reaction(self, emoji: str) -> None:
        """Records an added reaction.

        Args:
            emoji (str): The reaction that was added.
        """
        self.reactions.append(emoji)

    async def remove_reaction(self, emoji: str, member: FakeUser) -> None:
        """Records a removed reaction without taking it out of `reactions`.

        A status chain swaps one emoji for the next, so `reactions` stays the full add order the
        run went through and `removed` is the separate record of what each swap took off.

        Args:
            emoji (str): The reaction that was removed.
            member (FakeUser): Whose reaction the removal was scoped to.
        """
        self.removed.append((emoji, member))

    async def reply(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records a message reply payload.

        Returns nothing, unlike `Message.reply`, so a path that keeps the reply it created needs
        a richer double than this one.

        Args:
            **kwargs (Unpack[DiscordPayload]): The payload the caller replied with.
        """
        self.replies.append(kwargs)

    async def delete(self) -> None:
        """Records the deletion as a flag, leaving every other recording readable afterwards."""
        self.deleted = True


class FakeInteraction:
    """Interaction stub shared by cog command and view tests."""

    def __init__(
        self,
        user: FakeUser | None = None,
        message: FakeDiscordMessage | object | None = None,
        filesize_limit: int = 25 * 1024 * 1024,
    ) -> None:
        """Builds an interaction double owning its own fresh response and followup recorders.

        The guild is a stand-in carrying `filesize_limit` alone, which is all `upload_limit_for`
        reads; lowering it is how a test drives a cog down the oversize hosted-URL branch instead
        of a native attachment.

        Args:
            user (FakeUser | None): The invoking user; None builds a default `FakeUser`.
            message (FakeDiscordMessage | object | None): What a view callback finds on
                `interaction.message`. `object` is in the union so the other fake message
                families the suite already has can be passed without conversion.
            filesize_limit (int): Upload ceiling in bytes exposed on the guild, defaulting to the
                25 MiB nextcord reports for an unboosted guild.
        """
        self.user = user or FakeUser()
        self.message = message
        self.guild = SimpleNamespace(filesize_limit=filesize_limit)
        self.response = FakeResponse()
        self.followup = FakeFollowup()
        self.edits: list[OriginalEditPayload] = []

    async def edit_original_message(self, **kwargs: Unpack[OriginalEditPayload]) -> None:
        """Records an edit to the deferred original response.

        Kept on the interaction rather than on the response double, mirroring where nextcord puts
        it: a cog that defers and then repaints its progress text lands every edit in this one
        list, so the last entry is what the user was left looking at.

        Args:
            **kwargs (Unpack[OriginalEditPayload]): The payload the caller edited with.
        """
        self.edits.append(kwargs)
