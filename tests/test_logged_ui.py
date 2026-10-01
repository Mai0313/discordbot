"""A failing button, select or modal reaches `./data/logs` instead of nextcord's stderr print.

Each test drives nextcord's own dispatch (`_scheduled_task`), which is what catches the raise
and hands it to `on_error`, so a callback called directly would prove nothing here.
"""

from typing import Any, cast
import sqlite3

import pytest
import nextcord
from nextcord import ButtonStyle, Interaction
from nextcord.ui import Button
from nextcord.ext import commands
from sqlalchemy.exc import OperationalError

from discordbot.cogs.memory.views import MemoryPagesView
from discordbot.cogs.economy.views import CreditLoanDecisionView
from discordbot.cogs.games.interactions import GameView
from discordbot.cogs.games.dragon_gate_views import DragonGateView, DragonGateBetModal

from tests.helpers.casting import as_interaction, make_not_found
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_logs


async def test_a_failing_memory_page_press_is_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page turn whose edit Discord refused leaves a line naming the control and the user."""
    logged = capture_logs(monkeypatch=monkeypatch, level="error")
    view = MemoryPagesView(pages=["一", "二"], footer_text="footer", title="title")
    button = cast("Button[Any]", view.next_page)
    interaction = FakeInteraction(user=FakeUser(user_id=7), message=FakeDiscordMessage())
    error = make_not_found()

    async def refused(**_kwargs: object) -> None:
        """Stands in for Discord refusing the page edit."""
        raise error

    monkeypatch.setattr(target=interaction.response, name="edit_message", value=refused)
    await view._scheduled_task(item=button, interaction=as_interaction(fake=interaction))

    assert len(logged) == 1
    _message, fields = logged[0]
    assert fields["view"] == "MemoryPagesView"
    assert fields["custom_id"] == button.custom_id
    assert fields["user_id"] == 7
    assert fields["message_id"] == 1
    assert fields["_exc_info"] is error


async def test_a_failing_loan_decision_is_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancel whose ledger write failed after the defer is no longer silent anywhere."""
    logged = capture_logs(monkeypatch=monkeypatch, level="error")
    error = OperationalError("cancel", None, sqlite3.OperationalError("database is locked"))

    async def locked(**_kwargs: object) -> None:
        """Stands in for the ledger write losing its lock."""
        raise error

    monkeypatch.setattr("discordbot.cogs.economy.views.cancel_loan_proposal", locked)
    view = CreditLoanDecisionView(proposal_id=42, lender_id=2, creator_id=1)
    button = cast("Button[Any]", view.cancel)
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    await view._scheduled_task(item=button, interaction=as_interaction(fake=interaction))

    assert interaction.response.deferred
    assert len(logged) == 1
    _message, fields = logged[0]
    assert fields["view"] == "CreditLoanDecisionView"
    assert fields["custom_id"] == "credit:cancel"
    assert fields["_exc_info"] is error


async def test_a_failing_custom_bet_submit_is_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom bet that raised past the table's own catches is logged by the modal."""
    logged = capture_logs(monkeypatch=monkeypatch, level="error")
    error = RuntimeError("bet failed")

    class _Table:
        async def submit_custom_bet(self, **_kwargs: object) -> None:
            """Stands in for a table whose bet raised past its own catches."""
            raise error

    modal = DragonGateBetModal(view=cast("DragonGateView", _Table()), minimum=1, maximum=10)
    interaction = FakeInteraction(user=FakeUser(user_id=8))
    interaction.data = cast("Any", {"components": []})
    await modal._scheduled_task(interaction=as_interaction(fake=interaction))

    assert len(logged) == 1
    _message, fields = logged[0]
    assert fields["modal"] == "DragonGateBetModal"
    assert fields["user_id"] == 8
    assert fields["_exc_info"] is error


class _ProbeGameView(GameView):
    interaction_failure_log = "Probe game control failed"
    notice_failure_log = "Probe game notice failed"

    @nextcord.ui.button(label="probe", style=ButtonStyle.secondary)
    async def press(
        self, _button: Button["_ProbeGameView"], _interaction: Interaction[commands.Bot]
    ) -> None:
        """Raises the way a game control's unexpected failure does."""
        raise RuntimeError("probe")


async def test_a_game_view_keeps_its_own_failure_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """A game names the line its failing control logs, as it did before the shared base."""
    logged = capture_logs(monkeypatch=monkeypatch, level="error")
    view = _ProbeGameView()
    button = cast("Button[Any]", view.press)
    await view._scheduled_task(
        item=button, interaction=as_interaction(fake=FakeInteraction(user=FakeUser(user_id=9)))
    )

    assert [message for message, _fields in logged] == ["Probe game control failed"]
    assert logged[0][1]["item_label"] == "probe"
