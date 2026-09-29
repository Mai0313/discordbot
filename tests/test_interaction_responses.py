"""Tests for the shared interaction send helpers and which of them expire."""

import pytest
import nextcord

from discordbot.utils import interaction_responses as interactions

from tests.helpers.casting import as_interaction
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage


@pytest.fixture
def scheduled_deletes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, str | None]]:
    """Records `(message, user_name)` for each message handed to the public cleanup."""
    scheduled: list[tuple[object, str | None]] = []

    def record(message: object, delay: float = 180, user_name: str | None = None) -> None:
        del delay
        scheduled.append((message, user_name))

    monkeypatch.setattr(target=interactions, name="schedule_public_message_delete", value=record)
    return scheduled


async def test_send_expiring_followup_waits_for_message_and_schedules_cleanup(
    monkeypatch: pytest.MonkeyPatch, scheduled_deletes: list[tuple[object, str | None]]
) -> None:
    """Public economy embeds must retrieve their message before cleanup."""
    interaction = FakeInteraction(user=FakeUser(name="alice"))
    posted = FakeDiscordMessage()
    sent: list[dict[str, object]] = []

    async def send_returning_posted(**kwargs: object) -> FakeDiscordMessage:
        """Records the send and answers with the one message the test can recognize."""
        sent.append(kwargs)
        return posted

    monkeypatch.setattr(target=interaction.followup, name="send", value=send_returning_posted)
    embed = nextcord.Embed(title="balance")

    await interactions.send_expiring_followup(
        interaction=as_interaction(fake=interaction), embed=embed
    )

    assert sent[0]["wait"] is True
    assert sent[0]["embed"] is embed
    assert "view" not in sent[0]
    assert scheduled_deletes == [(posted, "alice")]


async def test_send_private_followup_is_ephemeral_and_not_scheduled(
    scheduled_deletes: list[tuple[object, str | None]],
) -> None:
    """Personal economy embeds should not enter the public cleanup scheduler."""
    interaction = FakeInteraction(user=FakeUser(name="alice"))
    embed = nextcord.Embed(title="balance")

    await interactions.send_private_followup(
        interaction=as_interaction(fake=interaction), embed=embed
    )

    sent = interaction.followup.sent[0]
    assert sent["ephemeral"] is True
    assert sent["embed"] is embed
    assert scheduled_deletes == []
