"""Tests for the plumbing the game tables share: the table edit and the final render."""

from __future__ import annotations

import pytest
from nextcord import Embed, HTTPException

from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.cogs.games.interactions import table_edit_kwargs, publish_final_table
from discordbot.cogs.games.blackjack_views import build_in_progress_embeds

from tests.helpers.games import card, blackjack_round
from tests.helpers.casting import (
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
    make_invalid_webhook_token,
)
from tests.helpers.discord_mocks import FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_levels
from tests.helpers.message_cleanup import record_scheduled_deletes


def test_blackjack_table_edit_payload_adds_width_spacer() -> None:
    """Blackjack table edits attach one transparent spacer and reference it from every embed."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]],
        dealer=[card(rank="K", suit="♣"), card(rank="9", suit="♦")],
    )
    seat_embeds = build_in_progress_embeds(round_state=round_state)

    payload = table_edit_kwargs(embeds=seat_embeds, view=None)

    assert payload["attachments"] == []
    assert payload["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    for embed in payload["embeds"]:
        assert embed.image.url == embed_spacer_url()


@pytest.mark.parametrize(
    argnames="failure",
    argvalues=[make_not_found(), RuntimeError("edit refused")],
    ids=["message_gone", "edit_refused"],
)
async def test_a_failed_final_render_still_schedules_the_table_for_deletion(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """Settlement is committed before the final render, so a failed render never raises.

    Raising would also skip the deletion behind it, leaving the table up until a restart sweeps it.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    message = FakeDiscordMessage()
    message.edit_failure = failure
    landed = await publish_final_table(
        message=as_message(fake=message),
        embeds=[Embed(title="settled")],
        user_name="alice",
        game_name="Dragon Gate",
        interaction=None,
        message_id=message.id,
    )

    assert landed is False
    assert (scheduled.messages, scheduled.user_names) == ([message], ["alice"])


@pytest.mark.parametrize(
    argnames=("failure", "report"),
    argvalues=[
        (make_not_found(message="Unknown Webhook"), ("info", False)),
        (make_invalid_webhook_token(), ("info", False)),
        (make_forbidden(message="Missing Access"), ("warn", False)),
        (make_server_error(), ("warn", True)),
    ],
    ids=["token_404", "token_401", "token_refused", "token_broke"],
)
async def test_a_final_render_the_press_cannot_make_goes_through_the_channel(
    monkeypatch: pytest.MonkeyPatch, failure: HTTPException, report: tuple[str, bool]
) -> None:
    """A press's 404 may be its token rather than the message, which only the channel can tell.

    So any failure of the press is recorded and the render retried through the channel.
    """
    record_scheduled_deletes(monkeypatch=monkeypatch)
    reports = capture_levels(monkeypatch=monkeypatch, levels=("info", "warn"))
    message = FakeDiscordMessage()
    press = FakeInteraction(message=message)
    press.edit_failure = failure

    landed = await publish_final_table(
        message=as_message(fake=message),
        embeds=[Embed(title="settled")],
        user_name="alice",
        game_name="Dragon Gate",
        interaction=as_interaction(fake=press),
        message_id=message.id,
    )

    assert landed is True
    assert message.edits[-1]["view"] is None
    assert [(level, "_exc_info" in fields) for level, _, fields in reports] == [report]
