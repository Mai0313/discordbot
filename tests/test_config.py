"""Sanity checks for the typings.config module-level surface."""

from discordbot.typings import config


def test_fast_sync_guild_ids_contains_owner_guilds() -> None:
    """The pinned list ships with the owner's two working guilds."""
    assert 981592566208282634 in config.FAST_SYNC_GUILD_IDS
    assert 1143289646042853487 in config.FAST_SYNC_GUILD_IDS


def test_fast_sync_guild_ids_is_a_list_of_ints() -> None:
    """Slash-command decorators expect a plain ``list[int]`` for ``guild_ids``."""
    assert isinstance(config.FAST_SYNC_GUILD_IDS, list)
    assert all(isinstance(value, int) for value in config.FAST_SYNC_GUILD_IDS)
