"""Pins the server-profile-before-global avatar ladder, and the write path that stores it.

`utils/avatars.py::guild_avatar_url` is the single place that ladder is written down, and every
caller caching `UserAccount.avatar_url` reads it rather than re-deriving the rule. Three tests
drive the helper directly against a fake guild whose `fetch_count` is observable, because what
is worth pinning is not only which URL comes back but what it cost: a cached member answers
with no REST call, a cache miss spends exactly one `fetch_member`, and a member Discord will
not hand over degrades to the global avatar instead of raising. The bot runs without the
`members` intent, so the cache-miss path is the ordinary one rather than an edge case, and
neither the fetch nor the `HTTPException` suppression it leans on fails loudly if it regresses:
the caller gets a plausible URL, or loses a reward, and nothing anywhere reports an error.

The last test pins the wiring instead of the ladder. `cli.DiscordBot.on_message` is the
per-message reward, the highest-volume caller, and it is checked by capturing what reaches
`credit_with_repayment`. A resolution that stopped being guild-aware there would quietly store
the author's global avatar on every message with every other assertion in the suite still green.

The doubles stay local instead of coming from `tests/helpers/discord_mocks.py`: the helper reads
only `id` / `display_avatar` / `guild_avatar` plus `Guild.get_member` / `fetch_member`, and a
guild that counts its own fetches is what the cost assertions need. They cross into production
signatures through `tests/helpers/casting.py`'s pure casts.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING

from discordbot import cli
from discordbot.utils.avatars import guild_avatar_url

from tests.helpers.casting import (
    as_guild,
    as_message,
    as_avatar_user,
    as_discord_bot,
    make_not_found,
)

if TYPE_CHECKING:
    import pytest


class FakeUser:
    """Minimal user-like object carrying only a global avatar, never a server profile."""

    def __init__(self, user_id: int = 1, avatar_url: str = "https://cdn.test/global.png") -> None:
        """Initializes the identity and the global avatar the fallback resolves to."""
        self.id = user_id
        self.name = "tester"
        self.display_avatar = SimpleNamespace(url=avatar_url)
        self.bot = False


class FakeMember(FakeUser):
    """Minimal member-like object returned by fake guild lookups, with a server profile."""

    def __init__(
        self,
        user_id: int = 1,
        avatar_url: str = "https://cdn.test/global.png",
        guild_avatar_url: str | None = "https://cdn.test/guild.png",
    ) -> None:
        """Initializes the global avatar plus the per-server override, absent when None."""
        super().__init__(user_id=user_id, avatar_url=avatar_url)
        self.guild_avatar = (
            SimpleNamespace(url=guild_avatar_url) if guild_avatar_url is not None else None
        )


class FakeGuild:
    """Minimal guild resolving members from cache or REST, and counting the REST calls."""

    def __init__(
        self, cached_member: FakeMember | None, fetched_member: FakeMember | None
    ) -> None:
        """Initializes the member each lookup path resolves to, either of which may be None."""
        self.cached_member = cached_member
        self.fetched_member = fetched_member
        self.fetch_count = 0

    def get_member(self, user_id: int) -> FakeMember | None:
        """Returns the configured cached member when its id matches, standing in for the cache."""
        if self.cached_member is not None and self.cached_member.id == user_id:
            return self.cached_member
        return None

    async def fetch_member(self, user_id: int) -> FakeMember:
        """Returns the configured fetched member, recording that the REST call was spent.

        Raises the `NotFound` Discord answers for a member who has left when none is
        configured. That is an `HTTPException`, which is exactly what the helper suppresses.
        """  # noqa: DOC501 -- ruff reads the raise as `make_not_found`, a builder, not a type
        self.fetch_count += 1
        if self.fetched_member is not None and self.fetched_member.id == user_id:
            return self.fetched_member
        raise make_not_found(message="member not found")


async def test_guild_avatar_url_prefers_cached_guild_avatar() -> None:
    """A cached member's server avatar wins, and costs no REST fetch."""
    guild = FakeGuild(
        cached_member=FakeMember(guild_avatar_url="https://cdn.test/cached.png"),
        fetched_member=None,
    )

    avatar_url = await guild_avatar_url(
        user=as_avatar_user(fake=FakeUser()), guild=as_guild(fake=guild)
    )

    assert avatar_url == "https://cdn.test/cached.png"
    assert guild.fetch_count == 0


async def test_guild_avatar_url_fetches_member_when_cache_misses() -> None:
    """A cache miss spends exactly one fetch to recover the server avatar."""
    guild = FakeGuild(
        cached_member=None,
        fetched_member=FakeMember(guild_avatar_url="https://cdn.test/fetched.png"),
    )

    avatar_url = await guild_avatar_url(
        user=as_avatar_user(fake=FakeUser()), guild=as_guild(fake=guild)
    )

    assert avatar_url == "https://cdn.test/fetched.png"
    assert guild.fetch_count == 1


async def test_guild_avatar_url_falls_back_to_global_avatar() -> None:
    """A member the fetch cannot resolve degrades to the global avatar instead of raising."""
    guild = FakeGuild(cached_member=None, fetched_member=None)

    avatar_url = await guild_avatar_url(
        user=as_avatar_user(fake=FakeUser(avatar_url="https://cdn.test/global.png")),
        guild=as_guild(fake=guild),
    )

    assert avatar_url == "https://cdn.test/global.png"
    assert guild.fetch_count == 1


async def test_message_reward_stores_guild_avatar(monkeypatch: "pytest.MonkeyPatch") -> None:
    """The per-message reward stores the author's server avatar, not their global one."""
    captured_avatar_url = ""

    async def fake_credit_with_repayment(
        user_id: int, name: str, avatar_url: str, amount: int
    ) -> SimpleNamespace:
        """Captures the avatar URL the reward hands to the economy facade.

        Returns:
            A stand-in for `CreditResult`; `on_message` discards it.
        """
        nonlocal captured_avatar_url
        del user_id, name, amount
        captured_avatar_url = avatar_url
        return SimpleNamespace(new_balance=0)

    async def noop_process_commands(message: SimpleNamespace) -> None:
        """Absorbs the `process_commands` call `on_message` always makes at the end."""
        del message

    monkeypatch.setattr(cli, "credit_with_repayment", fake_credit_with_repayment)
    author = FakeUser(user_id=7, avatar_url="https://cdn.test/global.png")
    author.bot = False
    message = SimpleNamespace(
        author=author,
        guild=FakeGuild(
            cached_member=FakeMember(
                user_id=7,
                avatar_url="https://cdn.test/global.png",
                guild_avatar_url="https://cdn.test/server.png",
            ),
            fetched_member=None,
        ),
    )
    # A bare sentinel for `bot.user`: it only has to compare unequal to the author, or
    # `on_message` drops the message as the bot's own before any reward runs.
    bot = SimpleNamespace(
        user=object(), process_commands=noop_process_commands, _message_reward_at={}
    )

    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))

    assert captured_avatar_url == "https://cdn.test/server.png"
