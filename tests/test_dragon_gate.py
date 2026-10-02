"""Tests for 射龍門 rules and interaction views."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING, Any, TypeVar, NoReturn, cast
import sqlite3
import contextlib

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography
import pytest
from nextcord import Embed, HTTPException
from sqlalchemy.exc import OperationalError

from discordbot.typings.games import GameParticipant
from discordbot.typings.economy import (
    JackpotSnapshot,
    JackpotSettlementResult,
    JackpotSettlementRequest,
    JackpotSettlementBatchResult,
)
from discordbot.utils.discord_embeds import embed_text_length
from discordbot.cogs.games.dragon_gate import (
    ANTE,
    GAME_ID,
    DragonGateRound,
    DragonGateDirection,
    DragonGateTurnResult,
    DragonGatePlayerResult,
    DragonGateParticipantUnknownError,
    DragonGatePairChoiceUnavailableError,
    card_value,
    has_open_gate,
)
from discordbot.cogs.games.interactions import publish_final_table
from discordbot.services.economy.database import (
    get_jackpot_snapshot,
    apply_jackpot_settlement,
    apply_jackpot_settlement_batch,
)
from discordbot.cogs.games.dragon_gate_views import (
    DRAGON_GATE_VISIBLE_PLAYER_LINES,
    DRAGON_GATE_VISIBLE_HISTORY_LINES,
    DragonGateView,
    DragonGateBetModal,
    DragonGateLobbyView,
    build_dragon_gate_final_embed,
    build_dragon_gate_lobby_embed,
    build_dragon_gate_history_embed,
    build_dragon_gate_in_progress_embed,
)
from discordbot.services.economy.presentation import amount_code

from tests.helpers.games import (
    card,
    seat,
    joins_as,
    lobby_button,
    component_ids,
    component_rows,
    everyone_stays,
    attached_button,
    attached_select,
)
from tests.helpers.casting import (
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
    make_invalid_webhook_token,
)
from tests.helpers.economy import seed_balance, get_jackpot_pool
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_levels
from tests.helpers.message_cleanup import record_scheduled_deletes
from tests.helpers.economy_invariants import assert_wallet_consistent

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from _typeshed import SupportsLenAndGetItem


T = TypeVar("T")


_RIGGED_FILLER: tuple[str, ...] = ("2", "♠") * 32


class RiggedRandom(Random):
    """Random subclass that returns a fixed rank/suit sequence (padded with filler)."""

    def __init__(self, choices: Sequence[str]) -> None:
        """Initializes the deterministic choice stream with safe filler values."""
        super().__init__(x=0)
        # Pad with safe filler so the rules engine can keep dealing extra turns
        # after the asserted hand finishes; the view layer is responsible for
        # finalising on jackpot exhaustion, not the rules.
        padded = tuple(choices) + _RIGGED_FILLER
        self._scripted_choices: Iterator[str] = iter(padded)

    def choice(self, seq: SupportsLenAndGetItem[T]) -> T:
        """Returns the next scripted choice and verifies it belongs to the input."""
        value = next(self._scripted_choices)
        assert value in [seq[index] for index in range(len(seq))]
        return cast("T", value)


def _participant(user_id: int, display_name: str, balance: int = 1_000_000) -> GameParticipant:
    """Builds a 射龍門 seat, which always stakes the ante."""
    return seat(user_id=user_id, display_name=display_name, bet=ANTE, balance_at_start=balance)


async def _funded(user_id: int, display_name: str, balance: int = 1_000_000) -> GameParticipant:
    """Builds a 射龍門 seat whose wallet in the isolated ledger holds what it sat down with."""
    participant = _participant(user_id=user_id, display_name=display_name, balance=balance)
    await seed_balance(user_id=user_id, name=participant.account_name, amount=balance)
    return participant


def _record_jackpot_settlements(monkeypatch: pytest.MonkeyPatch) -> list[JackpotSettlementRequest]:
    """Settles every ante and bet against the isolated ledger, recording each request in order.

    Also stands in for the table cleanup, so no deletion is ever scheduled.
    """
    requests: list[JackpotSettlementRequest] = []

    async def settle(  # noqa: PLR0913 -- mirrors apply_jackpot_settlement
        player_id: int,
        player_account_name: str,
        player_delta: int,
        game_id: str,
        player_avatar_url: str = "",
        expected_jackpot_generation: int | None = None,
    ) -> JackpotSettlementResult:
        """Records the bet's request, then settles it."""
        requests.append(
            JackpotSettlementRequest(
                player_id=player_id,
                player_account_name=player_account_name,
                player_avatar_url=player_avatar_url,
                player_delta=player_delta,
                expected_jackpot_generation=expected_jackpot_generation,
            )
        )
        return await apply_jackpot_settlement(
            player_id=player_id,
            player_account_name=player_account_name,
            player_delta=player_delta,
            game_id=game_id,
            player_avatar_url=player_avatar_url,
            expected_jackpot_generation=expected_jackpot_generation,
        )

    async def settle_batch(
        game_id: str, settlements: Sequence[JackpotSettlementRequest]
    ) -> JackpotSettlementBatchResult:
        """Records the antes' requests, then settles them as one batch."""
        requests.extend(settlements)
        return await apply_jackpot_settlement_batch(game_id=game_id, settlements=settlements)

    monkeypatch.setattr("discordbot.cogs.games.dragon_gate_views.apply_jackpot_settlement", settle)
    monkeypatch.setattr("discordbot.cogs.games.lobby.apply_jackpot_settlement_batch", settle_batch)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    return requests


def test_card_value_uses_ace_low_and_faces_above_ten() -> None:
    """射龍門 compares A as 1 and J/Q/K as 11/12/13."""
    assert card_value(card=card(rank="A")) == 1
    assert card_value(card=card(rank="J")) == 11
    assert card_value(card=card(rank="Q")) == 12
    assert card_value(card=card(rank="K")) == 13


def test_adjacent_non_pair_pillars_are_redealt_without_counting_turn() -> None:
    """Adjacent non-pair pillars have no gate and are skipped before betting."""
    assert has_open_gate(pillars=[card(rank="4"), card(rank="3", suit="♥")]) is False
    assert has_open_gate(pillars=[card(rank="7"), card(rank="7", suit="♥")]) is True

    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("4", "♠", "3", "♥", "5", "♣", "9", "♦", "7", "♠")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )

    assert round_state.turn_number == 1
    assert round_state.active_turn is not None
    assert [card.rank for card in round_state.active_turn.pillars] == ["5", "9"]

    result = round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)
    assert result.outcome == "gate_win"
    assert result.delta == 10_000


def test_gate_win_returns_positive_delta() -> None:
    """A third card between the pillars wins one bet from the pot."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )
    result = round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)

    assert result.outcome == "gate_win"
    assert result.delta == 10_000
    assert round_state.player_delta(user_id=1) == 10_000


def test_outside_card_returns_negative_one_bet() -> None:
    """A third card outside the gate loses one bet."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "K", "♣", "A", "♦", "K", "♥")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )
    result = round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)

    assert result.outcome == "outside_lose"
    assert result.delta == -10_000
    assert round_state.player_delta(user_id=1) == -10_000


def test_pillar_hit_returns_negative_double_bet() -> None:
    """A third card equal to either pillar loses two bets."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "9", "♣", "A", "♦", "K", "♥")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )
    result = round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)

    assert result.outcome == "pillar_hit"
    assert result.delta == -20_000
    assert round_state.player_delta(user_id=1) == -20_000


def test_pair_gate_requires_high_or_low_choice() -> None:
    """Same-point pillars require a higher/lower choice before betting."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "8", "♣")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )

    with pytest.raises(expected_exception=ValueError, match="direction"):
        round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)

    round_state.choose_pair_direction(user_id=1, direction="higher")
    result = round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)
    assert result.outcome == "pair_win"
    assert result.delta == 10_000


def test_pair_pillar_hit_returns_triple_loss() -> None:
    """A same-point third card on a same-point gate loses three bets."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "7", "♣", "A", "♦", "K", "♥")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )
    round_state.choose_pair_direction(user_id=1, direction="lower")
    result = round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)

    assert result.outcome == "pair_pillar_hit"
    assert result.delta == -30_000
    assert round_state.player_delta(user_id=1) == -30_000


@pytest.mark.parametrize(
    argnames=("direction", "third"), argvalues=[("higher", "3"), ("lower", "9")]
)
def test_a_pair_card_on_the_side_not_called_loses_one_bet(
    direction: DragonGateDirection, third: str
) -> None:
    """A third card off the pillar pays only on the side the player called."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", third, "♣")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )
    round_state.choose_pair_direction(user_id=1, direction=direction)
    result = round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)

    assert (result.outcome, result.delta) == ("pair_lose", -10_000)
    assert round_state.player_delta(user_id=1) == -10_000


@pytest.mark.parametrize(
    argnames=("rank", "expected"), argvalues=[("A", "higher"), ("K", "lower")], ids=["ace", "king"]
)
def test_an_ace_or_king_pair_is_dealt_with_the_only_guess_that_can_win(
    rank: str, expected: DragonGateDirection
) -> None:
    """Nothing ranks below an ace or above a king, so the one winnable guess comes preset."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=(rank, "♠", rank, "♥", "7", "♣")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )

    assert round_state.needs_pair_choice() is False
    assert round_state.active_turn is not None
    assert round_state.active_turn.direction == expected

    result = round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)
    assert (result.outcome, result.delta, result.direction) == ("pair_win", 10_000, expected)


@pytest.mark.parametrize(
    argnames=("rank", "losing", "expected"),
    argvalues=[("A", "lower", "higher"), ("K", "higher", "lower")],
    ids=["ace", "king"],
)
def test_an_ace_or_king_pair_refuses_the_guess_that_cannot_win(
    rank: str, losing: DragonGateDirection, expected: DragonGateDirection
) -> None:
    """The guess that can only lose is never accepted on an ace or king pair."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=(rank, "♠", rank, "♥")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )

    with pytest.raises(expected_exception=DragonGatePairChoiceUnavailableError):
        round_state.choose_pair_direction(user_id=1, direction=losing)
    assert round_state.active_turn is not None
    assert round_state.active_turn.direction == expected


def test_a_gate_that_is_not_a_pair_offers_no_guess() -> None:
    """Only a pair takes a high/low guess, even when a pillar is an ace."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("A", "♠", "5", "♥")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )

    assert round_state.active_turn is not None
    assert round_state.active_turn.direction is None
    with pytest.raises(expected_exception=DragonGatePairChoiceUnavailableError):
        round_state.choose_pair_direction(user_id=1, direction="higher")


def test_turns_rotate_through_active_seats() -> None:
    """The next active player is dealt a fresh gate after a bet resolves."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "K", "♣", "4", "♦", "Q", "♣")),
        participants=[
            _participant(user_id=1, display_name="Alice"),
            _participant(user_id=2, display_name="Bob"),
        ],
    )

    round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)

    assert round_state.finished is False
    assert round_state.active_turn is not None
    assert round_state.active_turn.participant.user_id == 2
    assert [card.rank for card in round_state.active_turn.pillars] == ["4", "Q"]


def test_withdraw_advances_to_next_player_and_records_delta() -> None:
    """Withdrawing the active player skips to the next non-withdrawn seat."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "4", "♦", "Q", "♣")),
        participants=[
            _participant(user_id=1, display_name="Alice"),
            _participant(user_id=2, display_name="Bob"),
        ],
    )

    leftover = round_state.withdraw(user_id=1)

    assert leftover == 0
    assert round_state.finished is False
    assert round_state.active_turn is not None
    assert round_state.active_turn.participant.user_id == 2


def test_withdraw_finishes_round_when_last_player_leaves() -> None:
    """The round flips finished after the final active player leaves."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )

    round_state.withdraw(user_id=1)

    assert round_state.finished is True
    assert round_state.active_turn is None


def test_withdraw_rejects_non_participant() -> None:
    """Withdrawing someone not at the table is a programmer error."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        participants=[_participant(user_id=1, display_name="Alice")],
    )

    with pytest.raises(expected_exception=DragonGateParticipantUnknownError):
        round_state.withdraw(user_id=999)


def test_dragon_gate_embeds_show_lobby_progress_and_final_state() -> None:
    """Each embed carries what a player reads off it: who sits, whose turn, how the table ended."""
    owner = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    lobby = build_dragon_gate_lobby_embed(
        owner=owner, participants=[owner, bob], jackpot=100_000, status="ready"
    )
    seated = "\n".join(field.value or "" for field in lobby.fields)
    assert "Alice" in seated
    assert "Bob" in seated
    assert amount_code(amount=100_000, compact=True) in [field.value for field in lobby.fields]
    assert lobby.description == "ready"

    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )
    progress = build_dragon_gate_in_progress_embed(round_state=round_state, jackpot=110_000)
    assert isinstance(progress.description, str)
    assert "11萬" in progress.description
    assert "輪到 Alice" in progress.description

    assert round_state.place_bet(user_id=1, amount=10_000, jackpot=110_000).outcome == "gate_win"
    results = [
        DragonGatePlayerResult(
            participant=owner,
            delta=round_state.player_delta(user_id=1),
            final_balance=950_000,
            withdrawn=False,
        )
    ]
    final = build_dragon_gate_final_embed(
        round_state=round_state, results=results, jackpot=109_900, reason="彩金池清空"
    )
    # A lone player's title is their own signed net, and the description says why it ended.
    assert isinstance(final.title, str)
    assert amount_code(amount=10_000, signed=True, compact=True) in final.title
    assert isinstance(final.description, str)
    assert "彩金池清空" in final.description


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


async def test_dragon_gate_controls_hide_unavailable_actions() -> None:
    """Active controls are removed instead of left visible but disabled."""
    owner = _participant(user_id=1, display_name="Alice")

    normal_round = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")), participants=[owner]
    )
    normal_view = DragonGateView(
        round_state=normal_round,
        owner=owner,
        jackpot_snapshot=100_000,
        final_balances={1: 1_000_000},
    )
    normal_view.sync_controls()
    assert component_ids(view=normal_view) == {"dg:bet", "dg:leave"}
    assert component_rows(view=normal_view) == {"dg:leave": 0, "dg:bet": 2}
    assert attached_select(view=normal_view, custom_id="dg:bet").disabled is False

    pair_round = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "8", "♣")), participants=[owner]
    )
    pair_view = DragonGateView(
        round_state=pair_round,
        owner=owner,
        jackpot_snapshot=100_000,
        final_balances={1: 1_000_000},
    )
    pair_view.sync_controls()
    assert component_ids(view=pair_view) == {"dg:higher", "dg:lower", "dg:leave"}
    assert component_rows(view=pair_view) == {"dg:higher": 1, "dg:lower": 1, "dg:leave": 0}

    pair_round.choose_pair_direction(user_id=1, direction="higher")
    pair_view.sync_controls()
    assert component_ids(view=pair_view) == {"dg:bet", "dg:leave"}
    assert component_rows(view=pair_view) == {"dg:leave": 0, "dg:bet": 2}
    assert attached_select(view=pair_view, custom_id="dg:bet").disabled is False


@pytest.mark.parametrize(
    argnames=("rank", "label"), argvalues=[("A", "⬆️ 猜大"), ("K", "⬇️ 猜小")], ids=["ace", "king"]
)
async def test_an_ace_or_king_pair_table_goes_straight_to_the_bet(rank: str, label: str) -> None:
    """With one winnable guess there is nothing to choose: no guess buttons, no choice prompt."""
    owner = _participant(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=(rank, "♠", rank, "♥")), participants=[owner]
    )
    view = DragonGateView(
        round_state=round_state,
        owner=owner,
        jackpot_snapshot=100_000,
        final_balances={1: 1_000_000},
    )
    view.sync_controls()

    assert component_ids(view=view) == {"dg:bet", "dg:leave"}
    description = build_dragon_gate_in_progress_embed(
        round_state=round_state, jackpot=100_000
    ).description
    assert isinstance(description, str)
    assert label in description
    assert "請先按" not in description


async def test_dragon_gate_lobby_join_leave_and_owner_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lobby buttons mutate participants, and the owner's start charges the ante."""
    owner = await _funded(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    message = FakeDiscordMessage()
    settlements = _record_jackpot_settlements(monkeypatch=monkeypatch)
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=pool_before,
    )
    view.message = as_message(fake=message)

    join_button = lobby_button(view=view, label="加入")
    await join_button.callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    assert view.participants == [owner, bob]
    join_embed = message.edits[-1]["embed"]
    assert isinstance(join_embed, Embed)
    assert isinstance(join_embed.description, str)

    leave_button = lobby_button(view=view, label="離開")
    await leave_button.callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    assert view.participants == [owner]

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1), message=message)
    await lobby_button(view=view, label="開始").callback(as_interaction(fake=owner_interaction))
    assert isinstance(message.edits[-1]["view"], DragonGateView)
    assert settlements == [
        JackpotSettlementRequest(
            player_id=owner.user_id,
            player_account_name=owner.account_name,
            player_avatar_url=owner.avatar_url,
            player_delta=-ANTE,
            require_full_debit=True,
        )
    ]
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before + ANTE


async def test_dragon_gate_lobby_ante_rejection_keeps_lobby_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ante settlement rejects a non-owner, the lobby stays startable."""
    owner = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    message = FakeDiscordMessage()

    async def rejected_ante_batch(
        game_id: str, settlements: Sequence[JackpotSettlementRequest]
    ) -> JackpotSettlementBatchResult:
        """Rejects Bob's ante without mutating the table."""
        assert game_id == GAME_ID
        assert all(settlement.require_full_debit for settlement in settlements)
        return JackpotSettlementBatchResult(
            player_balances={},
            applied_player_deltas={},
            jackpot_balance=100_000,
            rejected_player_ids=(2,),
        )

    monkeypatch.setattr(
        "discordbot.cogs.games.lobby.apply_jackpot_settlement_batch", rejected_ante_batch
    )
    record_scheduled_deletes(monkeypatch=monkeypatch)

    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=100_000,
    )
    view.message = as_message(fake=message)

    join_button = lobby_button(view=view, label="加入")
    await join_button.callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    start_button = lobby_button(view=view, label="開始")
    await start_button.callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=1), message=message))
    )

    assert view.participants == [owner]
    assert view._started is False
    assert isinstance(message.edits[-1]["view"], DragonGateLobbyView)
    embed = message.edits[-1]["embed"]
    assert isinstance(embed, Embed)
    assert embed.description == "餘額不足已移出: Bob"


@pytest.mark.parametrize(
    argnames="failure",
    argvalues=[
        make_forbidden(message="Missing Access"),
        make_not_found(message="Unknown Message"),
        make_server_error(),
    ],
    ids=["refused", "message_gone", "discord_failing"],
)
async def test_a_start_whose_table_never_lands_returns_the_antes_and_reopens_the_lobby(
    monkeypatch: pytest.MonkeyPatch, failure: HTTPException
) -> None:
    """The antes are charged before the table edit, so a table that never appears hands them back.

    Left marked started, the lobby would refuse every press and skip its own timeout cleanup.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    batches: list[list[JackpotSettlementRequest]] = []

    async def recording_batch(
        game_id: str, settlements: Sequence[JackpotSettlementRequest]
    ) -> JackpotSettlementBatchResult:
        """Records each batch and applies it to the isolated ledger."""
        batches.append(list(settlements))
        return await apply_jackpot_settlement_batch(game_id=game_id, settlements=settlements)

    monkeypatch.setattr(
        "discordbot.cogs.games.lobby.apply_jackpot_settlement_batch", recording_batch
    )
    owner = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    for participant in (owner, bob):
        await seed_balance(
            user_id=participant.user_id, name=participant.account_name, amount=1_000
        )
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    message = FakeDiscordMessage()
    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=pool_before,
    )
    view.message = as_message(fake=message)
    join_button = lobby_button(view=view, label="加入")
    await join_button.callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )

    owner_start = FakeInteraction(user=FakeUser(user_id=1), message=message)
    owner_start.edit_failure = failure
    start_button = lobby_button(view=view, label="開始")
    # Only a refusal is answered in place; any other failure still reaches the view's on_error.
    with contextlib.suppress(HTTPException):
        await start_button.callback(as_interaction(fake=owner_start))

    await assert_wallet_consistent(user_id=1, expected_balance=1_000)
    await assert_wallet_consistent(user_id=2, expected_balance=1_000)
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before
    assert len(batches) == 2, "the antes go back in one transaction, as they were charged"
    # order-contract: the refund is awaited after the ante batch it reverses.
    assert {request.player_id: request.player_delta for request in batches[-1]} == {
        1: ANTE,
        2: ANTE,
    }
    assert not view.is_finished()
    await view.on_timeout()
    assert scheduled.messages == [message]


@pytest.mark.parametrize(
    argnames=("failing_step", "error"),
    argvalues=[
        (
            "discordbot.cogs.games.lobby.apply_jackpot_settlement_batch",
            OperationalError("ante", None, sqlite3.OperationalError("database is locked")),
        ),
        (
            "discordbot.cogs.games.dragon_gate_views.DragonGateView.in_progress_embeds",
            ValueError("table"),
        ),
    ],
    ids=["ante_charge_fails", "table_build_fails"],
)
async def test_a_start_that_raises_before_its_table_is_up_charges_nothing_and_reopens_the_lobby(
    monkeypatch: pytest.MonkeyPatch, failing_step: str, error: Exception
) -> None:
    """Whichever step of the start raises, the antes end where they were and the lobby reopens.

    Left marked started, the lobby would refuse every press and skip its own timeout cleanup.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    def failing(*_args: object, **_kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(failing_step, failing)
    owner = await _funded(user_id=1, display_name="Alice", balance=1_000)
    bob = await _funded(user_id=2, display_name="Bob", balance=1_000)
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    message = FakeDiscordMessage()
    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=pool_before,
    )
    view.message = as_message(fake=message)
    await lobby_button(view=view, label="加入").callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    # Called directly, the press skips the view's on_error, so the raise reaches the test.
    with pytest.raises(type(error)):
        await lobby_button(view=view, label="開始").callback(
            as_interaction(fake=FakeInteraction(user=FakeUser(user_id=1), message=message))
        )

    await assert_wallet_consistent(user_id=1, expected_balance=1_000)
    await assert_wallet_consistent(user_id=2, expected_balance=1_000)
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before
    assert not view.is_finished()
    await view.on_timeout()
    assert scheduled.messages == [message]


async def test_a_lobby_in_a_channel_the_bot_was_shut_out_of_still_opens_and_plays_its_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every press edits the message it sits on through its own token, not through the channel.

    The lobby went up on the slash command's token, so it shows in a channel the server shut the
    bot out of afterwards, where every channel edit is refused. A start edited through the
    channel charges the antes for a table that never appears.
    """
    owner = await _funded(user_id=1, display_name="Alice")
    bob = await _funded(user_id=2, display_name="Bob")
    settlements = _record_jackpot_settlements(monkeypatch=monkeypatch)

    message = FakeDiscordMessage()
    message.edit_failure = make_forbidden(message="Missing Access")
    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=await get_jackpot_pool(game_id=GAME_ID),
    )
    view.message = as_message(fake=message)

    await lobby_button(view=view, label="加入").callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    assert isinstance(message.edits[-1]["view"], DragonGateLobbyView)

    owner_start = FakeInteraction(user=FakeUser(user_id=1), message=message)
    await lobby_button(view=view, label="開始").callback(as_interaction(fake=owner_start))
    table = message.edits[-1]["view"]
    assert isinstance(table, DragonGateView)
    assert owner_start.followup.sent == []
    assert len(settlements) == 2, "the antes were charged once and never handed back"
    assert {request.player_id: request.player_delta for request in settlements} == {
        1: -ANTE,
        2: -ANTE,
    }

    await table._handle_bet_choice(
        choice="min",
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        ),
    )
    assert len(settlements) == 3
    assert len(message.edits) == 3
    assert message.edits[-1]["view"] is table

    # Bob's gate is the filler's pair of twos, so he calls it before he may bet.
    await attached_button(view=table, custom_id="dg:higher").callback(
        as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=2), message=message, custom_id="dg:higher")
        )
    )
    assert len(message.edits) == 4
    await attached_button(view=table, custom_id="dg:leave").callback(
        as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:leave")
        )
    )
    assert len(message.edits) == 5, "Alice left and the table stayed open for Bob"
    assert message.edits[-1]["view"] is table

    await attached_button(view=table, custom_id="dg:leave").callback(
        as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=2), message=message, custom_id="dg:leave")
        )
    )
    assert message.edits[-1]["view"] is None, "the last leave settled the table"
    assert table.is_finished(), "the settled table stopped taking presses"


async def test_dragon_gate_view_pair_choice_bet_settles_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bet calls apply_jackpot_settlement and updates the live snapshot."""
    owner = await _funded(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "8", "♣")), participants=[owner]
    )
    settlements = _record_jackpot_settlements(monkeypatch=monkeypatch)
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    message = FakeDiscordMessage()
    view = DragonGateView(
        round_state=round_state,
        owner=owner,
        jackpot_snapshot=pool_before,
        final_balances={1: 1_000_000},
    )
    view.message = as_message(fake=message)
    view.sync_controls()
    assert component_ids(view=view) == {"dg:higher", "dg:lower", "dg:leave"}
    assert attached_button(view=view, custom_id="dg:higher").disabled is False

    choose_higher = attached_button(view=view, custom_id="dg:higher")
    await choose_higher.callback(
        as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:higher")
        )
    )
    assert round_state.active_turn is not None
    assert round_state.active_turn.direction == "higher"
    assert component_ids(view=view) == {"dg:bet", "dg:leave"}
    assert attached_select(view=view, custom_id="dg:bet").disabled is False

    await view._handle_bet_choice(
        choice="min",
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        ),
    )

    # 7-pair, higher, third = 8 → pair_win at +bet (MIN_BET = 20)
    assert settlements[-1].player_delta == 20
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before - 20
    assert view._jackpot_snapshot == pool_before - 20


async def test_dragon_gate_view_max_bet_is_bounded_by_player_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A low-balance player's max bet is capped at their balance, not the whole pool."""
    owner = await _funded(user_id=1, display_name="Alice", balance=100)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )
    settlements = _record_jackpot_settlements(monkeypatch=monkeypatch)
    pool = await get_jackpot_pool(game_id=GAME_ID)
    assert pool > 100

    message = FakeDiscordMessage()
    view = DragonGateView(
        round_state=round_state, owner=owner, jackpot_snapshot=pool, final_balances={1: 100}
    )
    view.message = as_message(fake=message)
    view.sync_controls()

    # The pool is bounded down to the player's 100 balance.
    assert view._active_max_bet() == 100

    await view._handle_bet_choice(
        choice="max",
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        ),
    )

    # Gate win pays only the balance-bounded 100, closing the free-option.
    assert settlements[-1].player_delta == 100
    assert round_state.player_delta(user_id=1) == 100


async def test_dragon_gate_view_sub_min_balance_cannot_bet_above_wallet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A player whose balance is below the minimum bet cannot win above wallet risk."""
    owner = await _funded(user_id=1, display_name="Alice", balance=15)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )
    settlements = _record_jackpot_settlements(monkeypatch=monkeypatch)

    message = FakeDiscordMessage()
    view = DragonGateView(
        round_state=round_state,
        owner=owner,
        jackpot_snapshot=await get_jackpot_pool(game_id=GAME_ID),
        final_balances={1: 15},
    )
    view.message = as_message(fake=message)
    view.sync_controls()

    # Balance 15 is below the 20 minimum, so betting is unavailable instead of
    # being floored back above the player's wallet.
    assert view._active_max_bet() == 15
    assert component_ids(view=view) == {"dg:leave"}

    interaction = FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
    await view._handle_bet_choice(choice="min", interaction=as_interaction(fake=interaction))
    assert settlements == []
    assert interaction.followup.sent[-1]["content"] == "餘額不足以下注，請先離桌"


async def test_dragon_gate_view_pool_emptied_replenishes_and_finalises_without_clawback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draining the pool replenishes it and skips the 逆贏不拿 refund."""
    owner = await _funded(user_id=1, display_name="Alice", balance=500_000)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )
    settlements = _record_jackpot_settlements(monkeypatch=monkeypatch)
    # A fresh ledger's pool holds its seed, which is what a drained pool is topped back up to.
    seed = await get_jackpot_snapshot(game_id=GAME_ID)

    message = FakeDiscordMessage()
    # The channel refuses every edit, so the closing render lands only through the press.
    message.edit_failure = make_forbidden(message="Missing Access")
    view = DragonGateView(
        round_state=round_state,
        owner=owner,
        jackpot_snapshot=seed.balance,
        final_balances={1: 500_000},
    )
    view.message = as_message(fake=message)
    view.sync_controls()

    await view._handle_bet_choice(
        choice="max",
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        ),
    )

    # gate_win for the full pot → pool replenished, table finalised, no refund follow-up
    assert [request.player_delta for request in settlements] == [seed.balance]
    assert await get_jackpot_snapshot(game_id=GAME_ID) == JackpotSnapshot(
        balance=seed.balance, generation=seed.generation + 1
    )
    await assert_wallet_consistent(user_id=1, expected_balance=500_000 + seed.balance)
    assert view._settled is True
    # The pool state above is the invariant, and the closing render has to say WHY the table
    # ended: the reason rides the final embed's description, and an emptied pool that was
    # topped back up reads differently from one that simply emptied.
    embeds = message.edits[-1]["embeds"]
    assert isinstance(embeds, list)
    assert isinstance(embeds[0], Embed)
    assert "系統已自動補池" in str(embeds[0].description)


async def test_dragon_gate_view_uses_capped_jackpot_settlement_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale view snapshot is replaced by the DB-applied jackpot delta."""
    owner = _participant(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )

    async def capped_settlement(**kwargs: Any) -> JackpotSettlementResult:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Returns a lower applied delta than the rules snapshot requested."""
        assert kwargs["expected_jackpot_generation"] == 2
        return JackpotSettlementResult(
            player_balance=507_000,
            jackpot_balance=100_000,
            jackpot_generation=3,
            applied_player_delta=7_000,
            jackpot_depleted=True,
        )

    monkeypatch.setattr(
        "discordbot.cogs.games.dragon_gate_views.apply_jackpot_settlement", capped_settlement
    )

    async def fake_get_balance(user_id: int) -> int:
        """Returns the owner's wallet balance for the live bet bound check."""
        del user_id
        return 500_000

    monkeypatch.setattr("discordbot.cogs.games.dragon_gate_views.get_balance", fake_get_balance)
    record_scheduled_deletes(monkeypatch=monkeypatch)

    message = FakeDiscordMessage()
    view = DragonGateView(
        round_state=round_state,
        owner=owner,
        jackpot_snapshot=10_000,
        jackpot_generation=2,
        final_balances={1: 500_000},
    )
    view.message = as_message(fake=message)
    view.sync_controls()

    await view._handle_bet_choice(
        choice="max",
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        ),
    )

    assert round_state.player_delta(user_id=1) == 7_000
    assert view._settled is True
    embeds = message.edits[-1]["embeds"]
    assert isinstance(embeds, list)
    final_embed = embeds[1]
    assert isinstance(final_embed, Embed)
    assert isinstance(final_embed.description, str)
    assert "+7,000" in final_embed.description
    assert "+10,000" not in final_embed.description
    assert view._jackpot_generation == 3


async def test_dragon_gate_view_single_player_zero_balance_finalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A player whose Dragon Gate loss clamps to zero is withdrawn and finalizes."""
    owner = await _funded(user_id=1, display_name="Alice", balance=30)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "3", "♣")), participants=[owner]
    )
    _record_jackpot_settlements(monkeypatch=monkeypatch)
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    message = FakeDiscordMessage()
    # The channel refuses every edit, so the closing render lands only through the press.
    message.edit_failure = make_forbidden(message="Missing Access")
    view = DragonGateView(
        round_state=round_state, owner=owner, jackpot_snapshot=pool_before, final_balances={1: 30}
    )
    view.message = as_message(fake=message)
    view.sync_controls()

    await view._handle_bet_choice(
        choice="min",
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        ),
    )

    # MIN_BET 20 pillar hit (-40) clamps to the 30 balance, busting the player.
    await assert_wallet_consistent(user_id=1, expected_balance=0)
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before + 30
    assert round_state.player_delta(user_id=1) == -30
    assert round_state.is_active(user_id=1) is False
    assert round_state.finished is True
    assert view._settled is True
    # The end-state above is the invariant; what the render has to carry is the clamped delta,
    # which the history embed below holds.
    embeds = message.edits[-1]["embeds"]
    assert isinstance(embeds, list)
    history_embed = embeds[-1]
    assert isinstance(history_embed, Embed)
    assert isinstance(history_embed.description, str)
    assert "-30" in history_embed.description
    assert "-40" not in history_embed.description


async def test_dragon_gate_view_zero_balance_withdraws_only_that_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In multiplayer, a zero-balance loser leaves while the next player continues."""
    alice = await _funded(user_id=1, display_name="Alice", balance=30)
    bob = await _funded(user_id=2, display_name="Bob", balance=100_000)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "3", "♣")), participants=[alice, bob]
    )
    _record_jackpot_settlements(monkeypatch=monkeypatch)
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    message = FakeDiscordMessage()
    view = DragonGateView(
        round_state=round_state,
        owner=alice,
        jackpot_snapshot=pool_before,
        final_balances={1: 30, 2: 100_000},
    )
    view.message = as_message(fake=message)
    view.sync_controls()

    await view._handle_bet_choice(
        choice="min",
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        ),
    )

    # MIN_BET 20 pillar hit (-40) clamps to the 30 balance, busting only Alice.
    await assert_wallet_consistent(user_id=1, expected_balance=0)
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before + 30
    assert round_state.player_delta(user_id=1) == -30
    assert round_state.is_active(user_id=1) is False
    assert round_state.is_active(user_id=2) is True
    assert round_state.finished is False
    assert view._settled is False
    assert round_state.active_turn is not None
    assert round_state.active_turn.participant.user_id == 2


async def test_dragon_gate_view_leave_refunds_running_winnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaving with a positive running delta refunds the surplus into the pool."""
    alice = await _funded(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣", "4", "♦", "Q", "♣")),
        participants=[alice, bob],
    )
    settlements = _record_jackpot_settlements(monkeypatch=monkeypatch)
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    message = FakeDiscordMessage()
    view = DragonGateView(
        round_state=round_state,
        owner=alice,
        jackpot_snapshot=pool_before,
        final_balances={1: 1_000_000, 2: 1_000_000},
    )
    view.message = as_message(fake=message)
    view.sync_controls()

    await view._handle_bet_choice(
        choice="min",
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        ),
    )
    assert round_state.player_delta(user_id=1) == 20

    leave_button = attached_button(view=view, custom_id="dg:leave")
    await leave_button.callback(
        as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:leave")
        )
    )

    # Bet settled +20 into Alice. Leave refunds 20 back into the pool.
    # order-contract: the leave hands back winnings the bet already settled.
    assert [request.player_delta for request in settlements] == [20, -20]
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before
    assert view._refunded_to_pool[1] == 20
    assert round_state.is_active(user_id=1) is False
    assert round_state.active_turn is not None
    assert round_state.active_turn.participant.user_id == 2


async def test_dragon_gate_view_bet_uses_live_wallet_not_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bet is validated against the live wallet, not a stale in-table balance cache."""
    owner = _participant(user_id=1, display_name="Alice", balance=1_000)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )
    # Live wallet dropped to 100 (player spent elsewhere mid-round); the in-table
    # cache still shows the post-ante 1,000.
    await seed_balance(user_id=1, name=owner.account_name, amount=100)
    settlements = _record_jackpot_settlements(monkeypatch=monkeypatch)

    message = FakeDiscordMessage()
    view = DragonGateView(
        round_state=round_state,
        owner=owner,
        jackpot_snapshot=await get_jackpot_pool(game_id=GAME_ID),
        final_balances={1: 1_000},
    )
    view.message = as_message(fake=message)
    view.sync_controls()

    # 500 is under the stale 1,000 cache but over the live 100 balance, so it is rejected.
    await view.submit_custom_bet(
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message)
        ),
        raw_amount="500",
    )

    assert settlements == []
    assert round_state.player_delta(user_id=1) == 0


async def test_dragon_gate_view_leave_without_winnings_does_not_refund(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaving while down or even does not push points back into the pool."""
    alice = await _funded(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "K", "♣", "4", "♦", "Q", "♣")),
        participants=[alice, bob],
    )
    settlements = _record_jackpot_settlements(monkeypatch=monkeypatch)

    message = FakeDiscordMessage()
    view = DragonGateView(
        round_state=round_state,
        owner=alice,
        jackpot_snapshot=await get_jackpot_pool(game_id=GAME_ID),
        final_balances={1: 1_000_000, 2: 1_000_000},
    )
    view.message = as_message(fake=message)
    view.sync_controls()

    await view._handle_bet_choice(
        choice="min",
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        ),
    )
    assert round_state.player_delta(user_id=1) == -20

    leave_button = attached_button(view=view, custom_id="dg:leave")
    await leave_button.callback(
        as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:leave")
        )
    )

    # Single bet settled -20; leave path does not append another settlement.
    assert [request.player_delta for request in settlements] == [-20]
    assert 1 not in view._refunded_to_pool


async def test_dragon_gate_view_rejects_non_active_and_invalid_custom_bet() -> None:
    """Only the active player can bet; the leave button is open to all seated."""
    alice = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")), participants=[alice, bob]
    )
    view = DragonGateView(
        round_state=round_state,
        owner=alice,
        jackpot_snapshot=100_000,
        final_balances={1: 1_000_000, 2: 1_000_000},
    )

    non_active = FakeInteraction(
        user=FakeUser(user_id=2), message=FakeDiscordMessage(), custom_id="dg:bet"
    )
    assert await view.interaction_check(interaction=as_interaction(fake=non_active)) is False
    assert non_active.followup.sent == [{"content": "現在輪到 Alice", "ephemeral": True}]

    leave_ok = FakeInteraction(
        user=FakeUser(user_id=2), message=FakeDiscordMessage(), custom_id="dg:leave"
    )
    assert await view.interaction_check(interaction=as_interaction(fake=leave_ok)) is True

    invalid = FakeInteraction(user=FakeUser(user_id=1), message=FakeDiscordMessage())
    await view.submit_custom_bet(
        interaction=as_interaction(fake=invalid), raw_amount="not a number"
    )
    assert invalid.followup.sent == [{"content": "下注金額要是整數", "ephemeral": True}]


async def _refusing_table(gate: tuple[str, str], bob: bool, finished: bool) -> DragonGateView:
    """Builds Alice's table on a 100,000 pool whose first gate is `gate`, dealt to her.

    `bob` seats Bob after her. `finished` withdraws every seat, leaving a round that is over
    while its view has not settled.
    """
    participants = [await _funded(user_id=1, display_name="Alice")]
    if bob:
        participants.append(await _funded(user_id=2, display_name="Bob"))
    low, high = gate
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=(low, "♠", high, "♥")), participants=participants
    )
    if finished:
        for participant in participants:
            round_state.withdraw(user_id=participant.user_id)
    return DragonGateView(
        round_state=round_state,
        owner=participants[0],
        jackpot_snapshot=100_000,
        final_balances={participant.user_id: 1_000_000 for participant in participants},
    )


@pytest.mark.parametrize(
    argnames=("gate", "bob", "finished", "expected"),
    argvalues=[
        (("7", "7"), False, True, "這桌已經不能操作了"),
        (("7", "7"), True, False, "現在輪到 Alice"),
        (("3", "9"), False, False, "這手不需要猜大小"),
    ],
    ids=["table-over", "not-their-turn", "not-a-pair"],
)
async def test_a_refused_pair_guess_says_why(
    gate: tuple[str, str], bob: bool, finished: bool, expected: str
) -> None:
    """A guess the round will not take answers with the notice for the rule it broke.

    With Bob seated, he is the one pressing; otherwise Alice is.
    """
    view = await _refusing_table(gate=gate, bob=bob, finished=finished)
    press = FakeInteraction(
        user=FakeUser(user_id=2 if bob else 1), message=FakeDiscordMessage(), custom_id="dg:higher"
    )

    await view._choose_direction(interaction=as_interaction(fake=press), direction="higher")

    assert press.followup.sent == [{"content": expected, "ephemeral": True}]


@pytest.mark.parametrize(
    argnames=("gate", "bob", "finished", "raw_amount", "expected"),
    argvalues=[
        (("7", "7"), False, False, "20", "同點門柱要先猜大或猜小"),
        (("3", "9"), False, True, "20", "這桌已經不能操作了"),
        (("3", "9"), True, False, "20", "現在輪到 Alice"),
        (("3", "9"), False, False, "5", "下注金額需介於 20 虛擬歡樂豆 到 10萬 虛擬歡樂豆"),
    ],
    ids=["guess-first", "table-over", "not-their-turn", "out-of-range"],
)
async def test_a_refused_bet_says_why(
    gate: tuple[str, str], bob: bool, finished: bool, raw_amount: str, expected: str
) -> None:
    """A bet the round will not take answers with the notice for the rule it broke.

    With Bob seated, he is the one betting; otherwise Alice is.
    """
    view = await _refusing_table(gate=gate, bob=bob, finished=finished)
    press = FakeInteraction(user=FakeUser(user_id=2 if bob else 1), message=FakeDiscordMessage())

    await view.submit_custom_bet(interaction=as_interaction(fake=press), raw_amount=raw_amount)

    assert press.followup.sent == [{"content": expected, "ephemeral": True}]


async def test_a_press_on_a_round_with_no_turn_left_is_refused() -> None:
    """A round that is over before its view settles turns every turn-bound control away."""
    view = await _refusing_table(gate=("3", "9"), bob=False, finished=True)
    press = FakeInteraction(
        user=FakeUser(user_id=1), message=FakeDiscordMessage(), custom_id="dg:bet"
    )

    assert await view.interaction_check(interaction=as_interaction(fake=press)) is False
    assert press.followup.sent == [{"content": "這桌已經不能操作了", "ephemeral": True}]


async def test_dragon_gate_custom_bet_modal_allows_formatted_maximum() -> None:
    """Custom bet input length matches the comma-stripping parser."""
    owner = _participant(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")), participants=[owner]
    )
    view = DragonGateView(
        round_state=round_state,
        owner=owner,
        jackpot_snapshot=1_000_000,
        final_balances={1: 1_000_000},
    )
    modal = DragonGateBetModal(view=view, minimum=10_000, maximum=1_000_000)

    assert modal.amount.max_length == len("1,000,000")
    assert isinstance(modal.amount.placeholder, str)
    assert modal.amount.placeholder


async def test_dragon_gate_view_timeout_refunds_remaining_winners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout refunds positive running deltas back into the jackpot."""
    alice = await _funded(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[alice]
    )
    settlements = _record_jackpot_settlements(monkeypatch=monkeypatch)
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    message = FakeDiscordMessage()
    view = DragonGateView(
        round_state=round_state,
        owner=alice,
        jackpot_snapshot=pool_before,
        final_balances={1: 1_000_000},
    )
    view.message = as_message(fake=message)
    view.sync_controls()

    await view._handle_bet_choice(
        choice="min",
        interaction=as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        ),
    )
    assert round_state.player_delta(user_id=1) == 20

    await view.on_timeout()

    # order-contract: the timeout hands back winnings the bet already settled.
    assert [request.player_delta for request in settlements] == [20, -20]
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before
    assert view._refunded_to_pool[1] == 20
    embeds = message.edits[-1]["embeds"]
    assert isinstance(embeds, list)
    assert all(isinstance(embed, Embed) for embed in embeds)


@pytest.mark.parametrize(argnames="expired", argvalues=[False, True], ids=["live", "expired"])
@pytest.mark.parametrize(argnames="last", argvalues=["start", "bet", "direction", "leave"])
async def test_a_dragon_gate_table_left_to_time_out_in_a_shut_out_channel_closes_through_its_last_press(
    monkeypatch: pytest.MonkeyPatch, last: str, expired: bool
) -> None:
    """A timeout has no press of its own, so it renders and deletes through the last one.

    Whichever control that press was, the start included. Once its token has expired only the
    channel is left, which refuses the render.
    """
    owner = await _funded(user_id=1, display_name="Alice")
    bob = await _funded(user_id=2, display_name="Bob")
    _record_jackpot_settlements(monkeypatch=monkeypatch)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    message = FakeDiscordMessage()
    message.edit_failure = make_forbidden(message="Missing Access")
    lobby = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=await get_jackpot_pool(game_id=GAME_ID),
    )
    lobby.message = as_message(fake=message)
    await lobby_button(view=lobby, label="加入").callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    press = FakeInteraction(user=FakeUser(user_id=1), message=message)
    await lobby_button(view=lobby, label="開始").callback(as_interaction(fake=press))
    table = message.edits[-1]["view"]
    assert isinstance(table, DragonGateView)
    if last != "start":
        press = FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
        await table._handle_bet_choice(choice="min", interaction=as_interaction(fake=press))
    # Bob's gate is the filler's pair of twos, so he calls it before he may bet.
    if last == "direction":
        press = FakeInteraction(user=FakeUser(user_id=2), message=message, custom_id="dg:higher")
        await attached_button(view=table, custom_id="dg:higher").callback(
            as_interaction(fake=press)
        )
    elif last == "leave":
        press = FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:leave")
        await attached_button(view=table, custom_id="dg:leave").callback(
            as_interaction(fake=press)
        )
    assert press.edits[-1]["view"] is table, "the table is still open"
    press.expired = expired

    await table.on_timeout()

    assert (press.edits[-1]["view"] is None) is not expired, "the settled table landed via it"
    assert scheduled.interactions == [press], "the delete rides the same press"


def test_dragon_gate_history_embed_uses_account_name_for_code_block() -> None:
    """History code blocks use stable account names instead of long display names."""
    participant = GameParticipant(
        user_id=1,
        account_name="alice",
        display_name="Alice With A Very Long Server Nickname",
        bet=ANTE,
        balance_at_start=100_000,
        is_allin=False,
    )
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[participant]
    )
    result = round_state.place_bet(user_id=1, amount=10_000, jackpot=100_000)

    embed = build_dragon_gate_history_embed(history=[result], round_state=round_state)

    assert embed is not None
    assert isinstance(embed.description, str)
    assert "alice" in embed.description
    assert "Alice With A Very Long Server Nickname" not in embed.description


def test_dragon_gate_history_embed_stays_inside_discord_at_its_worst() -> None:
    """A long round must not grow the history past what Discord will render.

    Past the limit the table stops updating while the round carries on, and silently, since what
    fails is the edit rather than anything a player does.

    Both limits, because the one that binds first is not the obvious one: a description gets
    4096, but `_finalize_locked` sends this embed beside the final one and Discord counts 6000
    across a message's embeds, which the history is by far the larger half of.

    Every input is at its widest rather than at whatever a convenient deal produced — the gate
    that renders longest, names at Discord's 32-character maximum, every seat withdrawn so each
    scoreboard row carries its suffix, and amounts at the widest the compact formatter emits.
    A worst case assembled from whatever is to hand reads several lines' worth narrower, which is
    headroom that is not there.
    """
    longest_name = "w" * 32
    participants = [
        _participant(user_id=index, display_name=longest_name, balance=10**15)
        for index in range(1, DRAGON_GATE_VISIBLE_PLAYER_LINES + 1)
    ]
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=participants
    )
    round_state.withdrawn_user_ids = {participant.user_id for participant in participants}
    for participant in participants:
        round_state.player_deltas[participant.user_id] = -(10**15)
    widest_turn = DragonGateTurnResult(
        turn_number=99_999,
        participant=participants[0],
        pillars=[card(rank="10"), card(rank="10", suit="♥")],
        third_card=card(rank="10", suit="♦"),
        bet=10**15,
        outcome="pair_pillar_hit",
        delta=-(10**15),
    )
    # Distinct turn numbers of the same width, so which turns survive the cap can be read back.
    history = [
        widest_turn.model_copy(update={"turn_number": 90_000 + index})
        for index in range(DRAGON_GATE_VISIBLE_HISTORY_LINES * 3)
    ]
    results = [
        DragonGatePlayerResult(
            participant=participant,
            delta=-(10**15),
            final_balance=10**15,
            withdrawn=True,
            refunded_to_pool=10**15,
        )
        for participant in participants
    ]

    embed = build_dragon_gate_history_embed(history=history, round_state=round_state)
    final_embed = build_dragon_gate_final_embed(
        round_state=round_state, results=results, jackpot=10**15, reason="round over"
    )

    assert embed is not None
    assert isinstance(embed.description, str)
    assert len(embed.description) <= 4096, (
        f"the history embed renders {len(embed.description)} characters at its worst, past "
        f"Discord's 4096 per description; lower DRAGON_GATE_VISIBLE_HISTORY_LINES"
    )
    settled_message = embed_text_length(embed=final_embed) + embed_text_length(embed=embed)
    assert settled_message <= 6000, (
        f"the settled table renders {settled_message} characters across its two embeds, past "
        f"Discord's 6000 per message; lower DRAGON_GATE_VISIBLE_HISTORY_LINES"
    )
    # Dropped turns are counted rather than vanishing, and the newest are the ones kept.
    hidden = len(history) - DRAGON_GATE_VISIBLE_HISTORY_LINES
    assert f"(前 {hidden} 手省略)" in embed.description
    shown = [
        turn.turn_number for turn in history if f"第 {turn.turn_number} 手" in embed.description
    ]
    assert shown == [turn.turn_number for turn in history[hidden:]]
