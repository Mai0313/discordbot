"""Tests for the shared bot-mention predicate."""

from types import SimpleNamespace

from discordbot.utils.mentions import has_bot_mention

BOT = SimpleNamespace(id=999)


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
