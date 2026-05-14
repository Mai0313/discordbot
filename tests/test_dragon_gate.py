"""Tests for 射龍門 rules and interaction views."""

from __future__ import annotations

from types import SimpleNamespace
from random import Random
from typing import TYPE_CHECKING, TypeVar, cast

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography
import pytest
from nextcord import Embed

from discordbot.cogs._games import lobby, dragon_gate_views
from discordbot.typings.games import Card, GameParticipant, DragonGatePlayerResult
from discordbot.cogs._games.dragon_gate import (
    ANTE,
    GAME_ID,
    DragonGateRound,
    DragonGateParticipantUnknownError,
    card_value,
    has_open_gate,
)
from discordbot.cogs._games.dragon_gate_views import (
    DragonGateView,
    DragonGateBetModal,
    DragonGateLobbyView,
    build_dragon_gate_final_embed,
    build_dragon_gate_lobby_embed,
    build_dragon_gate_in_progress_embed,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from _typeshed import SupportsLenAndGetItem

T = TypeVar("T")


class MessageStub:
    """Minimal Discord message stub that records edits."""

    def __init__(self) -> None:
        self.edits: list[dict[str, object]] = []

    async def edit(self, **kwargs: object) -> None:
        """Records a message edit payload."""
        self.edits.append(kwargs)


class ResponseStub:
    """Minimal interaction response stub."""

    def __init__(self) -> None:
        self.deferred = False
        self.sent: list[dict[str, object]] = []
        self.modals: list[object] = []

    async def defer(self) -> None:
        """Records that the interaction was deferred."""
        self.deferred = True

    async def send_message(self, **kwargs: object) -> None:
        """Records an ephemeral or public interaction message."""
        self.sent.append(kwargs)

    def is_done(self) -> bool:
        """Returns whether the interaction response has already been used."""
        return self.deferred or bool(self.sent) or bool(self.modals)

    async def send_modal(self, modal: object) -> None:
        """Records a modal launch."""
        self.modals.append(modal)


class FollowupStub:
    """Minimal interaction followup stub."""

    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send(self, **kwargs: object) -> MessageStub:
        """Records followup sends and returns a fake message."""
        self.sent.append(kwargs)
        return MessageStub()


class InteractionStub:
    """Minimal interaction stub for view callbacks."""

    def __init__(
        self, user_id: int = 1, message: MessageStub | None = None, custom_id: str = ""
    ) -> None:
        self.user = SimpleNamespace(
            id=user_id,
            name=f"user{user_id}",
            display_name=f"User {user_id}",
            display_avatar=SimpleNamespace(url=f"https://example.test/{user_id}.png"),
        )
        self.message = message
        self.response = ResponseStub()
        self.followup = FollowupStub()
        self.data: dict[str, object] = {"custom_id": custom_id}


class DealerStub:
    """Deterministic dealer stub for 射龍門 view tests."""

    def __init__(self) -> None:
        self.table_settle_calls: list[dict[str, object]] = []
        self.taunt_calls: list[dict[str, object]] = []

    async def taunt_bet(self, **kwargs: object) -> str:
        """Returns a deterministic opening line."""
        self.taunt_calls.append(kwargs)
        return "taunt"

    async def table_settle(self, **kwargs: object) -> str:
        """Returns a deterministic settlement line and records the call."""
        self.table_settle_calls.append(kwargs)
        return "settled"


_RIGGED_FILLER: tuple[str, ...] = ("2", "♠") * 32


class RiggedRandom(Random):
    """Random subclass that returns a fixed rank/suit sequence (padded with filler)."""

    def __init__(self, choices: Sequence[str]) -> None:
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


class JackpotState:
    """In-memory simulator for ``apply_jackpot_settlement`` used in view tests.

    Each ``settle`` call mutates the simulated player balance and jackpot
    snapshot, lets tests assert the running effect of multiple settlements
    without spinning up a real database.
    """

    def __init__(
        self,
        initial_jackpot: int = 100_000,
        initial_balance: int = 100_000,
        replenish_seed: int = 100_000,
    ) -> None:
        self.jackpot = initial_jackpot
        self.balances: dict[int, int] = {}
        self._initial_balance = initial_balance
        self._replenish_seed = replenish_seed
        self.calls: list[dict[str, object]] = []

    async def settle(
        self,
        player_id: int,
        player_account_name: str,
        player_delta: int,
        game_id: str,
        player_avatar_url: str = "",
    ) -> tuple[int, int]:
        """Mocks ``apply_jackpot_settlement`` and tracks the call chain."""
        assert game_id == GAME_ID
        self.balances.setdefault(player_id, self._initial_balance)
        self.balances[player_id] += player_delta
        self.jackpot -= player_delta
        if self._replenish_seed > 0 and self.jackpot <= 0:
            self.jackpot = self._replenish_seed
        self.calls.append({
            "player_id": player_id,
            "player_account_name": player_account_name,
            "player_delta": player_delta,
            "player_avatar_url": player_avatar_url,
        })
        return self.balances[player_id], self.jackpot


def _participant(user_id: int, display_name: str, balance: int = 1_000_000) -> GameParticipant:
    return GameParticipant(
        user_id=user_id,
        account_name=display_name.lower(),
        display_name=display_name,
        bet=ANTE,
        balance_at_start=balance,
        is_allin=False,
    )


def _install_jackpot_mock(monkeypatch: pytest.MonkeyPatch, state: JackpotState) -> None:
    monkeypatch.setattr(
        target=dragon_gate_views, name="apply_jackpot_settlement", value=state.settle
    )
    monkeypatch.setattr(target=lobby, name="apply_jackpot_settlement", value=state.settle)

    async def fake_get_balance(user_id: int) -> int:
        return state.balances.get(user_id, 0)

    monkeypatch.setattr(target=dragon_gate_views, name="get_balance", value=fake_get_balance)
    monkeypatch.setattr(
        target=dragon_gate_views, name="schedule_game_message_delete", value=lambda message: None
    )
    monkeypatch.setattr(
        target=lobby, name="schedule_game_message_delete", value=lambda message: None
    )


def test_card_value_uses_ace_low_and_faces_above_ten() -> None:
    """射龍門 compares A as 1 and J/Q/K as 11/12/13."""
    assert card_value(card=Card(rank="A", suit="♠")) == 1
    assert card_value(card=Card(rank="J", suit="♠")) == 11
    assert card_value(card=Card(rank="Q", suit="♠")) == 12
    assert card_value(card=Card(rank="K", suit="♠")) == 13


def test_adjacent_non_pair_pillars_are_redealt_without_counting_turn() -> None:
    """Adjacent non-pair pillars have no gate and are skipped before betting."""
    assert has_open_gate(pillars=[Card(rank="4", suit="♠"), Card(rank="3", suit="♥")]) is False
    assert has_open_gate(pillars=[Card(rank="7", suit="♠"), Card(rank="7", suit="♥")]) is True

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
    """Embed builders produce well-formed lobby / progress / final embeds."""
    owner = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    lobby = build_dragon_gate_lobby_embed(
        owner=owner, participants=[owner, bob], jackpot=100_000, status="ready"
    )
    assert isinstance(lobby, Embed)
    assert isinstance(lobby.title, str)
    assert lobby.title
    assert lobby.fields
    assert all(isinstance(field.value, str) and field.value for field in lobby.fields)

    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )
    progress = build_dragon_gate_in_progress_embed(round_state=round_state, jackpot=110_000)
    assert isinstance(progress, Embed)
    assert isinstance(progress.title, str)
    assert progress.title
    assert isinstance(progress.description, str)
    assert "110,000" in progress.description

    round_state.place_bet(user_id=1, amount=10_000, jackpot=110_000)
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
    assert isinstance(final, Embed)
    assert isinstance(final.title, str)
    assert final.title


async def test_dragon_gate_lobby_join_leave_and_owner_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lobby buttons mutate participants and only the owner starts the table."""
    owner = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    message = MessageStub()

    state = JackpotState()
    _install_jackpot_mock(monkeypatch=monkeypatch, state=state)

    async def prepare_participant(interaction: InteractionStub) -> GameParticipant | None:
        assert interaction.user.id == 2
        return bob

    async def refresh_participants(
        participants: list[GameParticipant],
    ) -> tuple[list[GameParticipant], list[str]]:
        return participants, []

    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        dealer=DealerStub(),
        dealer_name="Dealer",
        dealer_avatar_url="",
        prepare_participant=prepare_participant,
        refresh_participants=refresh_participants,
        initial_jackpot=state.jackpot,
    )
    view.message = message

    join_button = next(child for child in view.children if getattr(child, "label", "") == "加入")
    await join_button.callback(InteractionStub(user_id=2, message=message))
    assert view.participants == [owner, bob]
    join_embed = message.edits[-1]["embed"]
    assert isinstance(join_embed, Embed)
    assert isinstance(join_embed.description, str)

    leave_button = next(child for child in view.children if getattr(child, "label", "") == "離開")
    await leave_button.callback(InteractionStub(user_id=2, message=message))
    assert view.participants == [owner]

    start_button = next(child for child in view.children if getattr(child, "label", "") == "開始")
    other_interaction = InteractionStub(user_id=2, message=message)
    await start_button.callback(other_interaction)
    assert other_interaction.response.sent

    owner_interaction = InteractionStub(user_id=1, message=message)
    await start_button.callback(owner_interaction)
    assert isinstance(message.edits[-1]["view"], DragonGateView)
    assert state.calls == [
        {
            "player_id": owner.user_id,
            "player_account_name": owner.account_name,
            "player_delta": -ANTE,
            "player_avatar_url": owner.avatar_url,
        }
    ]
    assert state.jackpot == 100_000 + ANTE


async def test_dragon_gate_view_pair_choice_bet_settles_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bet calls apply_jackpot_settlement and updates the live snapshot."""
    owner = _participant(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "8", "♣")), participants=[owner]
    )

    state = JackpotState(initial_jackpot=100_000, initial_balance=1_000_000)
    _install_jackpot_mock(monkeypatch=monkeypatch, state=state)

    message = MessageStub()
    dealer = DealerStub()
    view = DragonGateView(
        dealer=dealer,
        round_state=round_state,
        owner=owner,
        dealer_name="Dealer",
        dealer_line="taunt",
        jackpot_snapshot=state.jackpot,
        final_balances={1: 1_000_000},
    )
    view.message = message
    view.sync_controls()
    assert view._button(custom_id="dg:higher").disabled is False
    assert view._select(custom_id="dg:bet").disabled is True

    choose_higher = view._button(custom_id="dg:higher")
    await choose_higher.callback(
        InteractionStub(user_id=1, message=message, custom_id="dg:higher")
    )
    assert round_state.active_turn is not None
    assert round_state.active_turn.direction == "higher"
    assert view._select(custom_id="dg:bet").disabled is False

    await view._handle_bet_choice(
        choice="min", interaction=InteractionStub(user_id=1, message=message, custom_id="dg:bet")
    )

    # 7-pair, higher, third = 8 → pair_win at +bet
    assert state.calls[-1]["player_delta"] == 10_000
    assert state.jackpot == 100_000 - 10_000
    assert view._jackpot_snapshot == state.jackpot


async def test_dragon_gate_view_pool_emptied_replenishes_and_finalises_without_clawback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draining the pool replenishes it and skips the 逆贏不拿 refund."""
    owner = _participant(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )

    state = JackpotState(initial_jackpot=10_000, initial_balance=500_000)
    _install_jackpot_mock(monkeypatch=monkeypatch, state=state)

    message = MessageStub()
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=owner,
        dealer_name="Dealer",
        dealer_line="taunt",
        jackpot_snapshot=state.jackpot,
        final_balances={1: 500_000},
    )
    view.message = message
    view.sync_controls()

    await view._handle_bet_choice(
        choice="max", interaction=InteractionStub(user_id=1, message=message, custom_id="dg:bet")
    )

    # gate_win for the full pot → pool replenished, table finalised, no refund follow-up
    assert state.jackpot == 100_000
    assert len(state.calls) == 1
    assert state.calls[0]["player_delta"] == 10_000
    assert view._settled is True
    embeds = message.edits[-1]["embeds"]
    assert isinstance(embeds, list)
    assert isinstance(embeds[1], Embed)
    assert isinstance(embeds[1].description, str)
    assert "系統已自動補池" in embeds[1].description


async def test_dragon_gate_view_leave_refunds_running_winnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaving with a positive running delta refunds the surplus into the pool."""
    alice = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣", "4", "♦", "Q", "♣")),
        participants=[alice, bob],
    )

    state = JackpotState(initial_jackpot=100_000, initial_balance=1_000_000)
    _install_jackpot_mock(monkeypatch=monkeypatch, state=state)

    message = MessageStub()
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=alice,
        dealer_name="Dealer",
        dealer_line="taunt",
        jackpot_snapshot=state.jackpot,
        final_balances={1: 1_000_000, 2: 1_000_000},
    )
    view.message = message
    view.sync_controls()

    await view._handle_bet_choice(
        choice="min", interaction=InteractionStub(user_id=1, message=message, custom_id="dg:bet")
    )
    assert round_state.player_delta(user_id=1) == 10_000

    leave_button = view._button(custom_id="dg:leave")
    await leave_button.callback(InteractionStub(user_id=1, message=message, custom_id="dg:leave"))

    # Bet settled +10k into Alice. Leave refunds 10k back into the pool.
    assert [call["player_delta"] for call in state.calls] == [10_000, -10_000]
    assert state.jackpot == 100_000
    assert view._refunded_to_pool[1] == 10_000
    assert round_state.is_active(user_id=1) is False
    assert round_state.active_turn is not None
    assert round_state.active_turn.participant.user_id == 2


async def test_dragon_gate_view_leave_without_winnings_does_not_refund(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaving while down or even does not push points back into the pool."""
    alice = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "K", "♣", "4", "♦", "Q", "♣")),
        participants=[alice, bob],
    )

    state = JackpotState(initial_jackpot=100_000, initial_balance=1_000_000)
    _install_jackpot_mock(monkeypatch=monkeypatch, state=state)

    message = MessageStub()
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=alice,
        dealer_name="Dealer",
        dealer_line="taunt",
        jackpot_snapshot=state.jackpot,
        final_balances={1: 1_000_000, 2: 1_000_000},
    )
    view.message = message
    view.sync_controls()

    await view._handle_bet_choice(
        choice="min", interaction=InteractionStub(user_id=1, message=message, custom_id="dg:bet")
    )
    assert round_state.player_delta(user_id=1) == -10_000

    leave_button = view._button(custom_id="dg:leave")
    await leave_button.callback(InteractionStub(user_id=1, message=message, custom_id="dg:leave"))

    # Single bet settled -10k; leave path does not append another settlement.
    assert [call["player_delta"] for call in state.calls] == [-10_000]
    assert 1 not in view._refunded_to_pool


async def test_dragon_gate_view_rejects_non_active_and_invalid_custom_bet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the active player can bet; the leave button is open to all seated."""
    alice = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")), participants=[alice, bob]
    )
    state = JackpotState()
    _install_jackpot_mock(monkeypatch=monkeypatch, state=state)
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=alice,
        dealer_name="Dealer",
        dealer_line="taunt",
        jackpot_snapshot=state.jackpot,
        final_balances={1: 1_000_000, 2: 1_000_000},
    )

    non_active = InteractionStub(user_id=2, message=MessageStub(), custom_id="dg:bet")
    assert await view.interaction_check(interaction=non_active) is False
    assert non_active.response.sent

    leave_ok = InteractionStub(user_id=2, message=MessageStub(), custom_id="dg:leave")
    assert await view.interaction_check(interaction=leave_ok) is True

    invalid = InteractionStub(user_id=1, message=MessageStub())
    await view.submit_custom_bet(interaction=invalid, raw_amount="not a number")
    assert invalid.response.sent


async def test_dragon_gate_custom_bet_modal_allows_formatted_maximum() -> None:
    """Custom bet input length matches the comma-stripping parser."""
    owner = _participant(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")), participants=[owner]
    )
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=owner,
        dealer_name="Dealer",
        dealer_line="taunt",
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
    alice = _participant(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[alice]
    )
    state = JackpotState(initial_jackpot=100_000, initial_balance=1_000_000)
    _install_jackpot_mock(monkeypatch=monkeypatch, state=state)

    message = MessageStub()
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=alice,
        dealer_name="Dealer",
        dealer_line="taunt",
        jackpot_snapshot=state.jackpot,
        final_balances={1: 1_000_000},
    )
    view.message = message
    view.sync_controls()

    await view._handle_bet_choice(
        choice="min", interaction=InteractionStub(user_id=1, message=message, custom_id="dg:bet")
    )
    assert round_state.player_delta(user_id=1) == 10_000

    await view.on_timeout()

    assert [call["player_delta"] for call in state.calls] == [10_000, -10_000]
    assert state.jackpot == 100_000
    assert view._refunded_to_pool[1] == 10_000
    embeds = message.edits[-1]["embeds"]
    assert isinstance(embeds, list)
    assert all(isinstance(embed, Embed) for embed in embeds)
