"""Tests for guild-aware Discord avatar selection."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from discordbot import cli
from discordbot.utils.avatars import AvatarUser, guild_avatar_url

from tests.helpers.casting import as_message, make_not_found

if TYPE_CHECKING:
    import pytest
    from nextcord import Guild


def _as_avatar_user(fake: object) -> AvatarUser:
    """Views a fake user as the AvatarUser protocol `guild_avatar_url` expects."""
    return cast("AvatarUser", fake)


def _as_guild(fake: object) -> "Guild | None":
    """Views a fake guild as the nextcord Guild `guild_avatar_url` expects."""
    return cast("Guild | None", fake)


class FakeUser:
    """Minimal user-like object with a global avatar."""

    def __init__(self, user_id: int = 1, avatar_url: str = "https://cdn.test/global.png") -> None:
        """Initializes a fake user identity."""
        self.id = user_id
        self.name = "tester"
        self.display_avatar = SimpleNamespace(url=avatar_url)
        self.bot = False


class FakeMember(FakeUser):
    """Minimal member-like object returned by fake guild lookups."""

    def __init__(
        self,
        user_id: int = 1,
        avatar_url: str = "https://cdn.test/global.png",
        guild_avatar_url: str | None = "https://cdn.test/guild.png",
    ) -> None:
        """Initializes global and optional guild avatars."""
        super().__init__(user_id=user_id, avatar_url=avatar_url)
        self.guild_avatar = (
            SimpleNamespace(url=guild_avatar_url) if guild_avatar_url is not None else None
        )


class FakeGuild:
    """Minimal guild that can return cached or fetched members."""

    def __init__(
        self, cached_member: FakeMember | None, fetched_member: FakeMember | None
    ) -> None:
        """Initializes member lookup fixtures."""
        self.cached_member = cached_member
        self.fetched_member = fetched_member
        self.fetch_count = 0

    def get_member(self, user_id: int) -> FakeMember | None:
        """Returns a cached member when one is configured."""
        if self.cached_member is not None and self.cached_member.id == user_id:
            return self.cached_member
        return None

    async def fetch_member(self, user_id: int) -> FakeMember:
        """Returns a fetched member when one is configured."""
        self.fetch_count += 1
        if self.fetched_member is not None and self.fetched_member.id == user_id:
            return self.fetched_member
        raise make_not_found(message="member not found")


async def test_guild_avatar_url_prefers_cached_guild_avatar() -> None:
    """Cached guild members provide the guild avatar without a REST fetch."""
    guild = FakeGuild(
        cached_member=FakeMember(guild_avatar_url="https://cdn.test/cached.png"),
        fetched_member=None,
    )

    avatar_url = await guild_avatar_url(
        user=_as_avatar_user(fake=FakeUser()), guild=_as_guild(fake=guild)
    )

    assert avatar_url == "https://cdn.test/cached.png"
    assert guild.fetch_count == 0


async def test_guild_avatar_url_fetches_member_when_cache_misses() -> None:
    """A fetch can recover the guild avatar when the event only has a user."""
    guild = FakeGuild(
        cached_member=None,
        fetched_member=FakeMember(guild_avatar_url="https://cdn.test/fetched.png"),
    )

    avatar_url = await guild_avatar_url(
        user=_as_avatar_user(fake=FakeUser()), guild=_as_guild(fake=guild)
    )

    assert avatar_url == "https://cdn.test/fetched.png"
    assert guild.fetch_count == 1


async def test_guild_avatar_url_falls_back_to_global_avatar() -> None:
    """Missing guild avatars and missing members fall back to the global avatar."""
    guild = FakeGuild(cached_member=None, fetched_member=None)

    avatar_url = await guild_avatar_url(
        user=_as_avatar_user(fake=FakeUser(avatar_url="https://cdn.test/global.png")),
        guild=_as_guild(fake=guild),
    )

    assert avatar_url == "https://cdn.test/global.png"
    assert guild.fetch_count == 1


async def test_message_reward_stores_guild_avatar(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Base message rewards pass the guild avatar into the economy DB facade."""
    captured_avatar_url = ""

    async def fake_credit_with_repayment(
        user_id: int, name: str, avatar_url: str, amount: int
    ) -> SimpleNamespace:
        """Records the avatar URL passed to the DB facade."""
        nonlocal captured_avatar_url
        del user_id, name, amount
        captured_avatar_url = avatar_url
        return SimpleNamespace(new_balance=0)

    async def noop_process_commands(message: SimpleNamespace) -> None:
        """Ignores command processing during the reward test."""
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
    bot = SimpleNamespace(
        user=object(), process_commands=noop_process_commands, _message_reward_at={}
    )

    await cli.DiscordBot.on_message(cast("cli.DiscordBot", bot), message=as_message(fake=message))

    assert captured_avatar_url == "https://cdn.test/server.png"
