"""Tests for the shared "is this message addressed to the bot" predicate."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from discordbot.utils.mentions import has_bot_mention, is_addressed_to_bot

from tests.helpers.casting import as_message
from tests.helpers.discord_mocks import FakeUser

if TYPE_CHECKING:
    from nextcord import ClientUser

BOT = cast("ClientUser", SimpleNamespace(id=999))


def test_has_bot_mention_matches_both_mention_forms() -> None:
    """Both `<@id>` and the legacy nickname form `<@!id>` count as a mention."""
    assert has_bot_mention(content="hey <@999> hi", bot_user=BOT) is True
    assert has_bot_mention(content="hey <@!999> hi", bot_user=BOT) is True


def test_has_bot_mention_ignores_another_user() -> None:
    """A mention of somebody else is not a mention of the bot."""
    assert has_bot_mention(content="hey <@111> hi", bot_user=BOT) is False


def test_has_bot_mention_without_bot_user() -> None:
    """Before the gateway connects there is no bot user, so nothing can match."""
    assert has_bot_mention(content="hey <@999>", bot_user=None) is False


def test_is_addressed_to_bot_treats_every_dm_as_addressed() -> None:
    """A DM reaches the reply pipeline without a mention, so it counts as addressed."""
    message = SimpleNamespace(guild=None, content="no mention here", author=FakeUser())
    assert is_addressed_to_bot(message=as_message(fake=message), bot_user=BOT) is True


def test_is_addressed_to_bot_needs_a_mention_in_a_guild() -> None:
    """In a guild only an explicit mention counts."""
    guild = SimpleNamespace(id=1)
    plain = SimpleNamespace(guild=guild, content="just a link", author=FakeUser())
    mentioned = SimpleNamespace(guild=guild, content="<@999> look", author=FakeUser())
    assert is_addressed_to_bot(message=as_message(fake=plain), bot_user=BOT) is False
    assert is_addressed_to_bot(message=as_message(fake=mentioned), bot_user=BOT) is True
