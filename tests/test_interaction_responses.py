"""Tests for the shared interaction send helpers and which of them expire."""

from typing import NoReturn

import pytest
import nextcord

from discordbot.utils import interaction_responses as interactions

from tests.helpers.casting import as_interaction, make_not_found
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_logs


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


async def test_a_refused_ephemeral_notice_is_logged_with_who_it_was_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The notice is advisory, so a refusal raises nothing and logs whom it was meant for."""
    interaction = FakeInteraction(user=FakeUser(user_id=5), channel_id=6)

    async def refuse(**kwargs: object) -> NoReturn:
        """Fails the response the way Discord does once the token has expired."""
        del kwargs
        raise make_not_found(message="Unknown interaction")

    monkeypatch.setattr(target=interaction.response, name="send_message", value=refuse)
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")

    await interactions.send_ephemeral_notice(
        interaction=as_interaction(fake=interaction), content="notice", log_message="notice failed"
    )

    assert [
        (message, fields["user_id"], fields["channel_id"], fields["error_type"])
        for message, fields in warns
    ] == [("notice failed", 5, 6, "NotFound")]
