"""Tests for guild-aware Discord avatar selection."""

from types import SimpleNamespace
from typing import TYPE_CHECKING

from nextcord import User, Member

from discordbot.utils.avatars import guild_avatar_url

from tests.helpers.casting import as_guild, as_avatar_user, make_not_found
from tests.helpers.discord_mocks import FakeUser

if TYPE_CHECKING:
    from nextcord.types.user import User as UserPayload


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
        self,
        cached_member: FakeMember | None,
        fetched_member: FakeMember | None,
        guild_id: int = 100,
    ) -> None:
        """Initializes member lookup fixtures."""
        self.cached_member = cached_member
        self.fetched_member = fetched_member
        self.id = guild_id
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


class _MemberState:
    """The slice of `nextcord.ConnectionState` that constructing a `Member` reads."""

    def store_user(self, data: "UserPayload") -> User:
        """Builds the member's user the way the real state does on a cache miss."""
        return User(state=self, data=data)  # ty: ignore[invalid-argument-type] -- the state slice a user reads


def _discord_member(user_id: int, guild: FakeGuild, guild_avatar: str) -> Member:
    """Builds a real `nextcord.Member`, since `guild_avatar_url` checks for one by type."""
    return Member(
        data={
            "user": {
                "id": str(user_id),
                "username": "tester",
                "discriminator": "0",
                "avatar": None,
            },
            "roles": [],
            "joined_at": "2020-01-01T00:00:00+00:00",
            "deaf": "false",
            "mute": "false",
            "flags": 0,
            "avatar": guild_avatar,
        },
        guild=guild,  # ty: ignore[invalid-argument-type] -- only its id is read, for the avatar URL
        state=_MemberState(),  # ty: ignore[invalid-argument-type] -- the state slice a member reads
    )


async def test_guild_avatar_url_reads_a_member_without_asking_the_guild() -> None:
    """A Member already carries its own guild avatar, so a stale guild cache is never read."""
    guild = FakeGuild(
        cached_member=FakeMember(user_id=7, guild_avatar_url="https://cdn.test/stale.png"),
        fetched_member=None,
    )
    member = _discord_member(user_id=7, guild=guild, guild_avatar="fresh")

    avatar_url = await guild_avatar_url(user=member, guild=as_guild(fake=guild))

    assert (
        avatar_url == "https://cdn.discordapp.com/guilds/100/users/7/avatars/fresh.png?size=1024"
    )


async def test_guild_avatar_url_prefers_cached_guild_avatar() -> None:
    """Cached guild members provide the guild avatar without a REST fetch."""
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
    """A fetch can recover the guild avatar when the event only has a user."""
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
    """Missing guild avatars and missing members fall back to the global avatar."""
    guild = FakeGuild(cached_member=None, fetched_member=None)

    avatar_url = await guild_avatar_url(
        user=as_avatar_user(fake=FakeUser(avatar_url="https://cdn.test/global.png")),
        guild=as_guild(fake=guild),
    )

    assert avatar_url == "https://cdn.test/global.png"
    assert guild.fetch_count == 1
