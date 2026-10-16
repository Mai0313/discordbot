"""Tests for the shared status-reaction helper."""

import pytest

from discordbot.utils.reactions import update_reaction

from tests.helpers.casting import as_message, make_forbidden
from tests.helpers.discord_mocks import FakeDiscordMessage
from tests.helpers.logfire_capture import capture_logs


async def test_a_refused_reaction_is_a_warn_with_its_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """A channel that refuses the reaction is logged with the ids, never swallowed silently."""
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    message = FakeDiscordMessage()

    async def refuse(emoji: str) -> None:
        """Refuses the reaction the way a channel overwrite does."""
        del emoji
        raise make_forbidden(message="Missing Permissions")

    monkeypatch.setattr(target=message, name="add_reaction", value=refuse)

    added = await update_reaction(message=as_message(fake=message), bot_user=None, emoji="✅")

    assert added == "✅"
    assert warns == [
        ("Discord refused a reaction", {"message_id": 1, "channel_id": 2, "emoji": "✅"})
    ]
