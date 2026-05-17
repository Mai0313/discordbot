"""Tests for 射龍門 rules and interaction views."""

from __future__ import annotations

from types import SimpleNamespace
from random import Random
from typing import TYPE_CHECKING, Any, TypeVar, cast

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography
import pytest
from nextcord import Embed

from discordbot.typings.games import (
    Card,
    GameParticipant,
    DragonGatePlayerResult,
    RefreshParticipantsResult,
)
from discordbot.typings.economy import (
    JackpotSettlementResult,
    JackpotSettlementRequest,
    JackpotSettlementBatchResult,
)
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
    build_dragon_gate_history_embed,
    build_dragon_gate_in_progress_embed,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from _typeshed import SupportsLenAndGetItem

T = TypeVar("T")


class MessageStub:
    """Minimal Discord message stub that records edits."""

    def __init__(self) -> None:
        """Initializes the recorded edit payloads."""
        self.edits: list[dict[str, Any]] = []

    async def edit(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Records a message edit payload."""
        self.edits.append(kwargs)


class ResponseStub:
    """Minimal interaction response stub."""

    def __init__(self) -> None:
        """Initializes interaction response state records."""
        self.deferred = False
        self.sent: list[dict[str, Any]] = []
        self.modals: list[DragonGateBetModal] = []

    async def defer(self) -> None:
        """Records that the interaction was deferred."""
        self.deferred = True

    async def send_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Records an ephemeral or public interaction message."""
        self.sent.append(kwargs)

    def is_done(self) -> bool:
        """Returns whether the interaction response has already been used."""
        return self.deferred or bool(self.sent) or bool(self.modals)

    async def send_modal(self, modal: DragonGateBetModal) -> None:
        """Records a modal launch."""
        self.modals.append(modal)


class FollowupStub:
    """Minimal interaction followup stub."""

    def __init__(self) -> None:
        """Initializes recorded followup sends."""
        self.sent: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> MessageStub:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Records followup sends and returns a fake message."""
        self.sent.append(kwargs)
        return MessageStub()


class InteractionStub:
    """Minimal interaction stub for view callbacks."""

    def __init__(
        self, user_id: int = 1, message: MessageStub | None = None, custom_id: str = ""
    ) -> None:
        """Initializes a callback interaction with user and component data."""
        self.user = SimpleNamespace(
            id=user_id,
            name=f"user{user_id}",
            display_name=f"User {user_id}",
            display_avatar=SimpleNamespace(url=f"https://example.test/{user_id}.png"),
        )
        self.message = message
        self.response = ResponseStub()
        self.followup = FollowupStub()
        self.data: dict[str, Any] = {"custom_id": custom_id}


class DealerStub:
    """Deterministic dealer stub for 射龍門 view tests."""

    def __init__(self) -> None:
        """Initializes dealer call records."""
        self.table_settle_calls: list[dict[str, Any]] = []
        self.taunt_calls: list[dict[str, Any]] = []

    async def taunt_bet(self, **kwargs: Any) -> str:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Returns a deterministic opening line."""
        self.taunt_calls.append(kwargs)
        return "taunt"

    async def table_settle(self, **kwargs: Any) -> str:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Returns a deterministic settlement line and records the call."""
        self.table_settle_calls.append(kwargs)
        return "settled"


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


class JackpotState:
    """In-memory simulator for jackpot settlement helpers used in view tests.

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
        """Initializes simulated player balances and jackpot state."""
        self.jackpot = initial_jackpot
        self.generation = 0
        self.balances: dict[int, int] = {}
        self._initial_balance = initial_balance
        self._replenish_seed = replenish_seed
        self.calls: list[dict[str, Any]] = []

    async def settle(  # noqa: PLR0913 -- mirrors apply_jackpot_settlement for monkeypatching
        self,
        player_id: int,
        player_account_name: str,
        player_delta: int,
        game_id: str,
        player_avatar_url: str = "",
        expected_jackpot_generation: int | None = None,
    ) -> JackpotSettlementResult:
        """Mocks ``apply_jackpot_settlement`` and tracks the call chain."""
        assert game_id == GAME_ID
        self.balances.setdefault(player_id, self._initial_balance)
        starting_balance = self.balances[player_id]
        if (
            player_delta > 0
            and expected_jackpot_generation is not None
            and expected_jackpot_generation != self.generation
        ):
            applied_delta = 0
        elif player_delta < 0:
            self.balances[player_id] = max(starting_balance + player_delta, 0)
            applied_delta = self.balances[player_id] - starting_balance
        else:
            self.balances[player_id] += player_delta
            applied_delta = self.balances[player_id] - starting_balance
        self.jackpot -= applied_delta
        depleted = self._replenish_seed > 0 and self.jackpot <= 0
        if depleted:
            self.jackpot = self._replenish_seed
            self.generation += 1
        self.calls.append({
            "player_id": player_id,
            "player_account_name": player_account_name,
            "player_delta": player_delta,
            "player_avatar_url": player_avatar_url,
            "expected_jackpot_generation": expected_jackpot_generation,
        })
        return JackpotSettlementResult(
            player_balance=self.balances[player_id],
            jackpot_balance=self.jackpot,
            jackpot_generation=self.generation,
            applied_player_delta=applied_delta,
            jackpot_depleted=depleted,
        )

    async def settle_batch(
        self, game_id: str, settlements: Sequence[JackpotSettlementRequest]
    ) -> JackpotSettlementBatchResult:
        """Mocks ``apply_jackpot_settlement_batch`` with the same state model."""
        player_balances: dict[int, int] = {}
        applied_player_deltas: dict[int, int] = {}
        for settlement in settlements:
            result = await self.settle(
                player_id=settlement.player_id,
                player_account_name=settlement.player_account_name,
                player_delta=settlement.player_delta,
                game_id=game_id,
                player_avatar_url=settlement.player_avatar_url,
                expected_jackpot_generation=settlement.expected_jackpot_generation,
            )
            player_balances[settlement.player_id] = result.player_balance
            applied_player_deltas[settlement.player_id] = result.applied_player_delta
            self.jackpot = result.jackpot_balance
        return JackpotSettlementBatchResult(
            player_balances=player_balances,
            applied_player_deltas=applied_player_deltas,
            jackpot_balance=self.jackpot,
            jackpot_generation=self.generation,
        )


def _participant(user_id: int, display_name: str, balance: int = 1_000_000) -> GameParticipant:
    """Builds a prepared 射龍門 participant for view tests."""
    return GameParticipant(
        user_id=user_id,
        account_name=display_name.lower(),
        display_name=display_name,
        bet=ANTE,
        balance_at_start=balance,
        is_allin=False,
    )


def _install_jackpot_mock(monkeypatch: pytest.MonkeyPatch, state: JackpotState) -> None:
    """Patches jackpot database calls to use an in-memory state model."""
    monkeypatch.setattr(
        "discordbot.cogs._games.dragon_gate_views.apply_jackpot_settlement", state.settle
    )
    monkeypatch.setattr(
        "discordbot.cogs._games.lobby.apply_jackpot_settlement_batch", state.settle_batch
    )

    async def fake_get_balance(user_id: int) -> int:
        """Returns the simulated final balance for a player."""
        return state.balances.get(user_id, 0)

    monkeypatch.setattr("discordbot.cogs._games.dragon_gate_views.get_balance", fake_get_balance)
    monkeypatch.setattr(
        "discordbot.cogs._games.dragon_gate_views.schedule_game_message_delete",
        lambda message: None,
    )
    monkeypatch.setattr(
        "discordbot.cogs._games.lobby.schedule_game_message_delete", lambda message: None
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
        """Returns Bob when the join interaction is accepted."""
        assert interaction.user.id == 2
        return bob

    async def refresh_participants(
        participants: list[GameParticipant],
    ) -> RefreshParticipantsResult:
        """Leaves all participants seated for lobby start."""
        return RefreshParticipantsResult(participants=participants)

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
            "expected_jackpot_generation": None,
        }
    ]
    assert state.jackpot == 100_000 + ANTE


async def test_dragon_gate_lobby_ante_rejection_keeps_lobby_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ante settlement rejects a non-owner, the lobby stays startable."""
    owner = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    message = MessageStub()

    async def prepare_participant(interaction: InteractionStub) -> GameParticipant | None:
        """Returns Bob when the join interaction is accepted."""
        assert interaction.user.id == 2
        return bob

    async def refresh_participants(
        participants: list[GameParticipant],
    ) -> RefreshParticipantsResult:
        """Leaves all participants seated for lobby start."""
        return RefreshParticipantsResult(participants=participants)

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
        "discordbot.cogs._games.lobby.apply_jackpot_settlement_batch", rejected_ante_batch
    )
    monkeypatch.setattr(
        "discordbot.cogs._games.lobby.schedule_game_message_delete", lambda message: None
    )

    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        dealer=DealerStub(),
        dealer_name="Dealer",
        dealer_avatar_url="",
        prepare_participant=prepare_participant,
        refresh_participants=refresh_participants,
        initial_jackpot=100_000,
    )
    view.message = message

    join_button = next(child for child in view.children if getattr(child, "label", "") == "加入")
    await join_button.callback(InteractionStub(user_id=2, message=message))
    start_button = next(child for child in view.children if getattr(child, "label", "") == "開始")
    await start_button.callback(InteractionStub(user_id=1, message=message))

    assert view.participants == [owner]
    assert view._started is False
    assert isinstance(message.edits[-1]["view"], DragonGateLobbyView)
    embed = message.edits[-1]["embed"]
    assert isinstance(embed, Embed)
    assert embed.description == "餘額不足已移出: Bob"


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
        "discordbot.cogs._games.dragon_gate_views.apply_jackpot_settlement", capped_settlement
    )
    monkeypatch.setattr(
        "discordbot.cogs._games.dragon_gate_views.schedule_game_message_delete",
        lambda message: None,
    )

    message = MessageStub()
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=owner,
        dealer_name="Dealer",
        dealer_line="taunt",
        jackpot_snapshot=10_000,
        jackpot_generation=2,
        final_balances={1: 500_000},
    )
    view.message = message
    view.sync_controls()

    await view._handle_bet_choice(
        choice="max", interaction=InteractionStub(user_id=1, message=message, custom_id="dg:bet")
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
    owner = _participant(user_id=1, display_name="Alice", balance=8_000)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "3", "♣")), participants=[owner]
    )

    state = JackpotState(initial_jackpot=100_000, initial_balance=8_000)
    _install_jackpot_mock(monkeypatch=monkeypatch, state=state)

    message = MessageStub()
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=owner,
        dealer_name="Dealer",
        dealer_line="taunt",
        jackpot_snapshot=state.jackpot,
        final_balances={1: 8_000},
    )
    view.message = message
    view.sync_controls()

    await view._handle_bet_choice(
        choice="min", interaction=InteractionStub(user_id=1, message=message, custom_id="dg:bet")
    )

    assert state.balances[1] == 0
    assert state.jackpot == 108_000
    assert round_state.player_delta(user_id=1) == -8_000
    assert round_state.is_active(user_id=1) is False
    assert round_state.finished is True
    assert view._settled is True
    embeds = message.edits[-1]["embeds"]
    assert isinstance(embeds, list)
    assert isinstance(embeds[1], Embed)
    assert isinstance(embeds[1].description, str)
    assert "所有玩家已離桌或餘額歸零" in embeds[1].description
    history_embed = embeds[-1]
    assert isinstance(history_embed, Embed)
    assert isinstance(history_embed.description, str)
    assert "-8,000" in history_embed.description
    assert "-20,000" not in history_embed.description


async def test_dragon_gate_view_zero_balance_withdraws_only_that_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In multiplayer, a zero-balance loser leaves while the next player continues."""
    alice = _participant(user_id=1, display_name="Alice", balance=8_000)
    bob = _participant(user_id=2, display_name="Bob", balance=100_000)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "3", "♣")), participants=[alice, bob]
    )

    state = JackpotState(initial_jackpot=100_000, initial_balance=100_000)
    state.balances[1] = 8_000
    state.balances[2] = 100_000
    _install_jackpot_mock(monkeypatch=monkeypatch, state=state)

    message = MessageStub()
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=alice,
        dealer_name="Dealer",
        dealer_line="taunt",
        jackpot_snapshot=state.jackpot,
        final_balances={1: 8_000, 2: 100_000},
    )
    view.message = message
    view.sync_controls()

    await view._handle_bet_choice(
        choice="min", interaction=InteractionStub(user_id=1, message=message, custom_id="dg:bet")
    )

    assert state.balances[1] == 0
    assert state.jackpot == 108_000
    assert round_state.player_delta(user_id=1) == -8_000
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
