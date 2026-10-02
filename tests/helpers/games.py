"""Builders the games tests share: cards, seats, lobbies, settlement, and a view's controls."""

from typing import Any

import pytest
from nextcord import Interaction
from nextcord.ui import View, Button
from nextcord.ext import commands

from discordbot.typings.games import (
    Card,
    GameParticipant,
    BlackjackPlayerSettlement,
    RefreshParticipantsResult,
)
from discordbot.cogs.games.lobby import PrepareParticipant
from discordbot.cogs.games.blackjack import BlackjackRound
from discordbot.cogs.games.settlement import settle_blackjack_player


def card(rank: str, suit: str = "♠") -> Card:
    """Builds one card; the suit has a default because every rule here reads the rank alone."""
    return Card(rank=rank, suit=suit)


def seat(
    user_id: int = 1, display_name: str = "Alice", bet: int = 100, balance_at_start: int = 1_000
) -> GameParticipant:
    """Builds a seated player whose account name is the display name lower-cased."""
    return GameParticipant(
        user_id=user_id,
        account_name=display_name.lower(),
        display_name=display_name,
        bet=bet,
        balance_at_start=balance_at_start,
        is_allin=False,
    )


def joins_as(participant: GameParticipant) -> PrepareParticipant:
    """Builds a lobby join hook that seats `participant` whoever presses 加入."""

    async def prepare_participant(interaction: Interaction[commands.Bot]) -> GameParticipant:
        del interaction
        return participant

    return prepare_participant


async def everyone_stays(participants: list[GameParticipant]) -> RefreshParticipantsResult:
    """Lobby start hook that leaves every participant seated."""
    return RefreshParticipantsResult(participants=participants)


def lobby_button(view: View, label: str) -> Button[Any]:
    """Returns the lobby control labelled `label`, failing the test when it is absent."""
    for child in view.children:
        if isinstance(child, Button) and child.label == label:
            return child
    raise AssertionError(f"no lobby button {label!r}")


class ScheduledDeletes:
    """What a view handed to the public-message cleanup, recorded in place of a deletion."""

    def __init__(self) -> None:
        """Initializes one list per argument, in call order."""
        self.messages: list[object] = []
        self.user_names: list[str | None] = []
        self.interactions: list[object | None] = []

    def __call__(
        self,
        message: object,
        delay: float = 180,
        user_name: str | None = None,
        interaction: object | None = None,
    ) -> None:
        """Records one scheduled deletion."""
        del delay
        self.messages.append(message)
        self.user_names.append(user_name)
        self.interactions.append(interaction)


def record_scheduled_deletes(monkeypatch: pytest.MonkeyPatch) -> ScheduledDeletes:
    """Replaces the public-message cleanup the games reach with a recorder, for this test.

    Every module that imports the scheduler holds its own reference, so each is patched.
    """
    scheduled = ScheduledDeletes()
    for module in (
        "discordbot.cogs.games.interactions",
        "discordbot.cogs.games.lobby",
        "discordbot.utils.interaction_responses",
    ):
        monkeypatch.setattr(f"{module}.schedule_public_message_delete", scheduled)
    return scheduled


async def settle_only_seat(round_state: BlackjackRound) -> BlackjackPlayerSettlement:
    """Settles a one-seat round's player against the ledger and returns the settlement."""
    return await settle_blackjack_player(round_state=round_state, player=round_state.players[0])


def component_ids(view: View) -> set[str]:
    """Returns the custom ids of the controls a view currently shows.

    Game views remove a control that cannot be used rather than disabling it, so which ids
    are attached is the state a player sees.
    """
    return set(component_rows(view=view))


def component_rows(view: View) -> dict[str, int | None]:
    """Returns `{custom_id: row}` for the controls a view currently shows."""
    rows: dict[str, int | None] = {}
    for child in view.children:
        custom_id = getattr(child, "custom_id", None)
        if isinstance(custom_id, str):
            rows[custom_id] = child.row
    return rows


def attached_button(view: View, custom_id: str) -> Button[Any]:
    """Returns the attached button with this custom id, failing the test when it is absent."""
    for child in view.children:
        if isinstance(child, Button) and child.custom_id == custom_id:
            return child
    raise AssertionError(f"no attached button {custom_id!r}; attached: {component_ids(view=view)}")
