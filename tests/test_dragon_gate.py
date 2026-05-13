"""Tests for 射龍門 rules and interaction views."""

from __future__ import annotations

from types import SimpleNamespace
from random import Random
from typing import TYPE_CHECKING, TypeVar, cast

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography
import pytest
from nextcord import Embed

from discordbot.cogs._games import dragon_gate_views
from discordbot.typings.games import Card, GameParticipant, WagerSettlement
from discordbot.cogs._games.dragon_gate import DragonGateRound, card_value
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

    def __init__(self, user_id: int = 1, message: MessageStub | None = None) -> None:
        self.user = SimpleNamespace(
            id=user_id,
            name=f"user{user_id}",
            display_name=f"User {user_id}",
            display_avatar=SimpleNamespace(url=f"https://example.test/{user_id}.png"),
        )
        self.message = message
        self.response = ResponseStub()
        self.followup = FollowupStub()


class DealerStub:
    """Deterministic dealer stub for 射龍門 view tests."""

    def __init__(self) -> None:
        self.table_settle_calls: list[dict[str, object]] = []

    async def taunt_bet(self, **_kwargs: object) -> str:
        """Returns a deterministic opening line."""
        return "taunt"

    async def table_settle(self, **kwargs: object) -> str:
        """Returns a deterministic settlement line and records the call."""
        self.table_settle_calls.append(kwargs)
        return "settled"


class RiggedRandom(Random):
    """Random subclass that returns a fixed rank/suit sequence."""

    def __init__(self, choices: Sequence[str]) -> None:
        super().__init__(x=0)
        self._scripted_choices: Iterator[str] = iter(choices)

    def choice(self, seq: SupportsLenAndGetItem[T]) -> T:
        """Returns the next scripted choice and verifies it belongs to the input."""
        value = next(self._scripted_choices)
        assert value in [seq[index] for index in range(len(seq))]
        return cast("T", value)


def _participant(user_id: int, display_name: str, ante: int = 100) -> GameParticipant:
    return GameParticipant(
        user_id=user_id,
        account_name=display_name.lower(),
        display_name=display_name,
        bet=ante,
        balance_at_start=1_000,
        is_allin=False,
    )


def test_card_value_uses_ace_low_and_faces_above_ten() -> None:
    """射龍門 compares A as 1 and J/Q/K as 11/12/13."""
    assert card_value(card=Card(rank="A", suit="♠")) == 1
    assert card_value(card=Card(rank="J", suit="♠")) == 11
    assert card_value(card=Card(rank="Q", suit="♠")) == 12
    assert card_value(card=Card(rank="K", suit="♠")) == 13


def test_gate_win_takes_bet_from_pot() -> None:
    """A third card between the pillars wins one bet from the pot."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")),
        participants=[_participant(user_id=1, display_name="Alice")],
        ante=100,
    )
    result = round_state.place_bet(user_id=1, amount=100)

    assert result.outcome == "gate_win"
    assert result.delta == 100
    assert result.pot_after == 0
    assert round_state.player_delta(user_id=1) == 0
    assert round_state.finished is True


def test_outside_card_loses_one_bet_into_pot() -> None:
    """A third card outside the gate pays one bet into the pot."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "K", "♣", "A", "♦", "K", "♥")),
        participants=[_participant(user_id=1, display_name="Alice")],
        ante=100,
    )
    result = round_state.place_bet(user_id=1, amount=100)

    assert result.outcome == "outside_lose"
    assert result.delta == -100
    assert result.pot_after == 200
    assert round_state.player_delta(user_id=1) == -200


def test_pillar_hit_loses_double_bet_into_pot() -> None:
    """A third card equal to either pillar pays two bets into the pot."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "9", "♣", "A", "♦", "K", "♥")),
        participants=[_participant(user_id=1, display_name="Alice")],
        ante=100,
    )
    result = round_state.place_bet(user_id=1, amount=100)

    assert result.outcome == "pillar_hit"
    assert result.delta == -200
    assert result.pot_after == 300
    assert round_state.player_delta(user_id=1) == -300


def test_pair_gate_requires_high_or_low_choice() -> None:
    """Same-point pillars require a higher/lower choice before betting."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "8", "♣")),
        participants=[_participant(user_id=1, display_name="Alice")],
        ante=100,
    )

    with pytest.raises(expected_exception=ValueError, match="direction"):
        round_state.place_bet(user_id=1, amount=100)

    round_state.choose_pair_direction(user_id=1, direction="higher")
    result = round_state.place_bet(user_id=1, amount=100)
    assert result.outcome == "pair_win"
    assert result.delta == 100


def test_pair_pillar_hit_loses_triple_bet() -> None:
    """A same-point third card on a same-point gate pays three bets."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "7", "♣", "A", "♦", "K", "♥")),
        participants=[_participant(user_id=1, display_name="Alice")],
        ante=100,
    )
    round_state.choose_pair_direction(user_id=1, direction="lower")
    result = round_state.place_bet(user_id=1, amount=100)

    assert result.outcome == "pair_pillar_hit"
    assert result.delta == -300
    assert result.pot_after == 400
    assert round_state.player_delta(user_id=1) == -400


def test_turns_rotate_until_pot_is_empty() -> None:
    """The next active player is dealt a fresh gate when the pot remains."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "K", "♣", "4", "♦", "Q", "♣")),
        participants=[
            _participant(user_id=1, display_name="Alice"),
            _participant(user_id=2, display_name="Bob"),
        ],
        ante=100,
    )

    round_state.place_bet(user_id=1, amount=100)

    assert round_state.finished is False
    assert round_state.active_turn is not None
    assert round_state.active_turn.participant.user_id == 2
    assert [card.rank for card in round_state.active_turn.pillars] == ["4", "Q"]


def test_dragon_gate_embeds_show_lobby_progress_and_final_state() -> None:
    """Embed builders expose pot, gate, last result, and settlements."""
    owner = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    lobby = build_dragon_gate_lobby_embed(
        owner=owner, participants=[owner, bob], ante=100, status="ready"
    )
    assert lobby.title == "♦️ 射龍門 | Lobby"
    assert "Alice 房主" in lobby.fields[0].value

    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner], ante=100
    )
    progress = build_dragon_gate_in_progress_embed(round_state=round_state, dealer_line="hello")
    assert progress.fields[0].name == "彩金池"
    assert "門柱" in progress.fields[1].value

    round_state.place_bet(user_id=1, amount=100)
    results = [
        dragon_gate_views.DragonGatePlayerResult(
            participant=owner,
            settlement=WagerSettlement(delta=0, payout=0, new_balance=900, house_balance=0),
        )
    ]
    final = build_dragon_gate_final_embed(
        round_state=round_state, results=results, dealer_line="done", reason="彩金池清空"
    )
    assert final.title == "♦️ 射龍門 | 結算"
    assert final.fields[0].value == "彩金池清空"
    assert "射進龍門" in final.fields[1].value


async def test_dragon_gate_lobby_join_leave_and_owner_start() -> None:
    """Lobby buttons mutate participants and only the owner starts the table."""
    owner = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    message = MessageStub()

    async def prepare_participant(
        interaction: InteractionStub, ante: int
    ) -> GameParticipant | None:
        assert ante == 100
        assert interaction.user.id == 2
        return bob

    async def refresh_participants(
        participants: list[GameParticipant], ante: int
    ) -> tuple[list[GameParticipant], list[str]]:
        assert ante == 100
        return participants, []

    view = DragonGateLobbyView(
        owner=owner,
        ante=100,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        dealer=DealerStub(),
        dealer_id=99,
        dealer_name="Dealer",
        dealer_avatar_url="",
        prepare_participant=prepare_participant,
        refresh_participants=refresh_participants,
    )
    view.message = message

    join_button = next(child for child in view.children if getattr(child, "label", "") == "加入")
    await join_button.callback(InteractionStub(user_id=2, message=message))
    assert view.participants == [owner, bob]
    assert message.edits[-1]["embed"].description == "Bob 已加入"

    leave_button = next(child for child in view.children if getattr(child, "label", "") == "離開")
    await leave_button.callback(InteractionStub(user_id=2, message=message))
    assert view.participants == [owner]
    assert message.edits[-1]["embed"].description == "Bob 已離開"

    start_button = next(child for child in view.children if getattr(child, "label", "") == "開始")
    other_interaction = InteractionStub(user_id=2, message=message)
    await start_button.callback(other_interaction)
    assert other_interaction.followup.sent[0]["content"] == "只有房主可以開始"

    owner_interaction = InteractionStub(user_id=1, message=message)
    await start_button.callback(owner_interaction)
    assert isinstance(message.edits[-1]["view"], DragonGateView)


async def test_dragon_gate_view_pair_choice_bet_and_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active table buttons choose high/low, bet, settle, and disable controls."""
    owner = _participant(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "8", "♣")), participants=[owner], ante=100
    )
    deleted: list[MessageStub] = []

    async def fake_settle_dragon_gate_player(**kwargs: object) -> WagerSettlement:
        delta = kwargs["delta"]
        assert isinstance(delta, int)
        return WagerSettlement(
            delta=delta, payout=max(delta, 0), new_balance=1_000 + delta, house_balance=-delta
        )

    def record_delete(message: MessageStub) -> None:
        deleted.append(message)

    monkeypatch.setattr(
        target=dragon_gate_views,
        name="settle_dragon_gate_player",
        value=fake_settle_dragon_gate_player,
    )
    monkeypatch.setattr(
        target=dragon_gate_views, name="schedule_game_message_delete", value=record_delete
    )

    message = MessageStub()
    dealer = DealerStub()
    view = DragonGateView(
        dealer=dealer,
        round_state=round_state,
        owner=owner,
        dealer_id=99,
        dealer_name="Dealer",
        dealer_line="taunt",
    )
    view.message = message
    view.sync_controls()
    assert view._button(custom_id="dg:higher").disabled is False
    assert view._button(custom_id="dg:min").disabled is True

    choose_higher = view._button(custom_id="dg:higher")
    await choose_higher.callback(InteractionStub(user_id=1, message=message))
    assert round_state.active_turn is not None
    assert round_state.active_turn.direction == "higher"
    assert view._button(custom_id="dg:min").disabled is False

    bet_minimum = view._button(custom_id="dg:min")
    await bet_minimum.callback(InteractionStub(user_id=1, message=message))

    assert round_state.finished is True
    assert dealer.table_settle_calls[0]["game"] == "dragon_gate"
    assert deleted == [message]
    assert all(getattr(child, "disabled", False) for child in view.children)


async def test_dragon_gate_view_rejects_non_active_and_invalid_custom_bet() -> None:
    """Only the active player can act, and custom bet input must be an integer."""
    alice = _participant(user_id=1, display_name="Alice")
    bob = _participant(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")), participants=[alice, bob], ante=100
    )
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=alice,
        dealer_id=99,
        dealer_name="Dealer",
        dealer_line="taunt",
    )

    non_active = InteractionStub(user_id=2, message=MessageStub())
    assert await view.interaction_check(interaction=non_active) is False
    assert non_active.response.sent[0]["content"] == "現在輪到 Alice"

    invalid = InteractionStub(user_id=1, message=MessageStub())
    await view.submit_custom_bet(interaction=invalid, raw_amount="not a number")
    assert invalid.response.sent[0]["content"] == "下注金額要是整數"

    stale_turn = InteractionStub(user_id=2, message=MessageStub())
    await view.submit_custom_bet(interaction=stale_turn, raw_amount="100")
    assert stale_turn.followup.sent[0]["content"] == "現在輪到 Alice"

    pair_round = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥")), participants=[alice], ante=100
    )
    pair_view = DragonGateView(
        dealer=DealerStub(),
        round_state=pair_round,
        owner=alice,
        dealer_id=99,
        dealer_name="Dealer",
        dealer_line="taunt",
    )
    pair_stale_modal = InteractionStub(user_id=1, message=MessageStub())
    await pair_view.submit_custom_bet(interaction=pair_stale_modal, raw_amount="100")
    assert pair_stale_modal.followup.sent[0]["content"] == "同點門柱要先猜大或猜小"

    modal = DragonGateBetModal(view=view, minimum=100, maximum=200)
    assert modal.title == "自訂下注"


async def test_dragon_gate_custom_bet_modal_allows_formatted_maximum() -> None:
    """Custom bet input length matches the comma-stripping parser."""
    owner = _participant(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")), participants=[owner], ante=100
    )
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=owner,
        dealer_id=99,
        dealer_name="Dealer",
        dealer_line="taunt",
    )
    modal = DragonGateBetModal(view=view, minimum=100, maximum=1_000_000)

    assert modal.amount.max_length == len("1,000,000")
    assert modal.amount.placeholder == "100 到 1,000,000"


async def test_dragon_gate_view_timeout_settles_remaining_pot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout finalizes the table and leaves unresolved pot in the house ledger."""
    owner = _participant(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")), participants=[owner], ante=100
    )
    settled_deltas: list[int] = []

    async def fake_settle_dragon_gate_player(**kwargs: object) -> WagerSettlement:
        delta = kwargs["delta"]
        assert isinstance(delta, int)
        settled_deltas.append(delta)
        return WagerSettlement(
            delta=delta, payout=max(delta, 0), new_balance=1_000 + delta, house_balance=-delta
        )

    monkeypatch.setattr(
        target=dragon_gate_views,
        name="settle_dragon_gate_player",
        value=fake_settle_dragon_gate_player,
    )
    monkeypatch.setattr(
        target=dragon_gate_views, name="schedule_game_message_delete", value=lambda message: None
    )

    message = MessageStub()
    view = DragonGateView(
        dealer=DealerStub(),
        round_state=round_state,
        owner=owner,
        dealer_id=99,
        dealer_name="Dealer",
        dealer_line="taunt",
    )
    view.message = message

    await view.on_timeout()

    assert settled_deltas == [-100]
    embed = message.edits[-1]["embed"]
    assert isinstance(embed, Embed)
    assert embed.fields[0].value == "逾時未操作, 剩餘彩金歸莊家"
