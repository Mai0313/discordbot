"""Tests for settling a Blackjack table: finalizing, timeouts, failures and the final embeds."""

from typing import Any, Literal, cast
import asyncio
import sqlite3

import pytest
from nextcord.ui import Button
from sqlalchemy.exc import OperationalError

from discordbot.cogs.games import blackjack_views
from discordbot.typings.games import (
    BlackjackPlayerResult,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
)
from discordbot.cogs.games.shoe import BlackjackShoeStore
from discordbot.typings.economy import RoundSettlementResult
from discordbot.cogs.games.blackjack import Card, BlackjackRound
from discordbot.services.economy.database import (
    get_balance,
    get_casino_ledger,
    apply_blackjack_settlement,
)
from discordbot.cogs.games.blackjack_views import BlackjackView, build_final_embeds

from tests.helpers.games import (
    card,
    seat,
    blackjack_view,
    attached_button,
    blackjack_round,
    settle_only_seat,
)
from tests.helpers.casting import (
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
)
from tests.helpers.economy import seed_balance
from tests.helpers.discord_mocks import FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_logs, capture_levels
from tests.helpers.message_cleanup import ScheduledDeletes


async def test_finalize_persists_remaining_shoe_to_the_store(
    monkeypatch: pytest.MonkeyPatch, scheduled_cleanups: ScheduledDeletes
) -> None:
    """Settling a round writes the round's remaining shoe back into the channel store."""
    store = BlackjackShoeStore()
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="9", suit="♥")]],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    round_state.shoe = [
        card(rank="7"),
        card(rank="8", suit="♥"),
        card(rank="2", suit="♦"),
        card(rank="3", suit="♣"),
    ]
    view = BlackjackView(round_state=round_state, owner=seat(), shoe_store=store, channel_id=42)

    async def _refuse_settlement(**_kwargs: object) -> None:
        raise RuntimeError("settlement stubbed out")

    # The shoe save is what is under test, so no seat needs a ledger to settle against.
    monkeypatch.setattr(blackjack_views, "settle_blackjack_player", _refuse_settlement)

    await view.finalize(message=as_message(fake=FakeDiscordMessage()), interaction=None)

    # The store holds a decoupled copy of the round's remaining shoe.
    assert store.shoes.get(42) == round_state.shoe
    assert store.shoes.get(42) is not round_state.shoe


async def test_history_persistence_uses_the_dealer_hand_captured_at_settlement(
    monkeypatch: pytest.MonkeyPatch, scheduled_cleanups: ScheduledDeletes
) -> None:
    """The round history records the dealer hand as it stood when the round settled.

    The write runs in a background task after `finalize` returns, so it has to be handed a copy:
    anything that touches the live round's dealer list afterwards would otherwise reach the row.
    """
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="9", suit="♥")]],
        dealer=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
        seats=[seat(bet=50, balance_at_start=100)],
        finished=True,
    )
    view = blackjack_view(round_state=round_state)
    recorded: dict[str, object] = {}

    async def record_blackjack_history(**kwargs: object) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(blackjack_views, "record_blackjack_history", record_blackjack_history)

    await view.finalize(message=as_message(fake=FakeDiscordMessage()), interaction=None)
    round_state.dealer.append(card(rank="K", suit="♣"))
    await view.wait_for_background_tasks()

    assert recorded["dealer_cards"] == [card(rank="10", suit="♣"), card(rank="8", suit="♦")]
    assert recorded["dealer_total"] == 18


async def test_blackjack_view_finalizes_once_when_called_concurrently(
    scheduled_cleanups: ScheduledDeletes,
) -> None:
    """Concurrent finalization attempts must not pay out one Blackjack hand twice."""
    await seed_balance(user_id=1, name="alice", amount=100)

    message = FakeDiscordMessage()
    view = blackjack_view(
        round_state=blackjack_round(
            hands=[[card(rank="10"), card(rank="Q", suit="♥")]],
            dealer=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
            seats=[seat(bet=50, balance_at_start=100)],
            finished=True,
        )
    )

    await asyncio.gather(
        view.finalize(message=as_message(fake=message), interaction=None),
        view.finalize(message=as_message(fake=message), interaction=None),
    )

    assert await get_balance(user_id=1) == 150
    ledger = await get_casino_ledger()
    assert ledger.balance == -50
    assert "embeds" not in message.edits[0]
    await view.wait_for_background_tasks()
    assert len(message.edits) == 2
    assert message.edits[1]["view"] is None
    assert scheduled_cleanups.messages == [message]


async def test_blackjack_view_timeout_auto_stands_and_settles(
    scheduled_cleanups: ScheduledDeletes,
) -> None:
    """A player who walks away is treated as standing and the wager resolves."""
    await seed_balance(user_id=1, name="alice", amount=100)

    message = FakeDiscordMessage()
    view = blackjack_view(
        round_state=blackjack_round(
            hands=[[card(rank="10"), card(rank="8", suit="♥")]],
            dealer=[card(rank="10", suit="♣"), card(rank="Q", suit="♦")],
            seats=[seat(bet=50, balance_at_start=100)],
        )
    )
    view.message = as_message(fake=message)

    await view.on_timeout()

    assert view.round_state.finished is True
    assert await get_balance(user_id=1) == 50
    ledger = await get_casino_ledger()
    assert ledger.balance == 50
    assert "embeds" not in message.edits[0]
    await view.wait_for_background_tasks()
    assert len(message.edits) == 2
    assert message.edits[1]["view"] is None
    assert scheduled_cleanups.messages == [message]


async def test_a_seat_whose_settlement_fails_leaves_the_rest_of_the_table_to_settle(
    monkeypatch: pytest.MonkeyPatch, scheduled_cleanups: ScheduledDeletes
) -> None:
    """A seat whose write fails is left unsettled, while the seats after it still settle.

    The table still shows its result and is deleted on schedule. The failed seat shows no payout
    and gets no history row, since nothing was booked for it.
    """
    seats = [
        seat(user_id=user_id, display_name=name)
        for user_id, name in ((1, "Alice"), (2, "Bob"), (3, "Carol"))
    ]
    for participant in seats:
        await seed_balance(
            user_id=participant.user_id, name=participant.account_name, amount=1_000
        )
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="9", suit="♥")] for _ in seats],
        dealer=[card(rank="10", suit="♣"), card(rank="7", suit="♦")],
        seats=seats,
        finished=True,
    )

    async def refuse_bob(**kwargs: Any) -> RoundSettlementResult:  # noqa: ANN401 -- forwarded as-is
        if kwargs["player_id"] == 2:
            raise OperationalError("settle", None, sqlite3.OperationalError("database is locked"))
        return await apply_blackjack_settlement(**kwargs)

    monkeypatch.setattr("discordbot.cogs.games.settlement.apply_blackjack_settlement", refuse_bob)
    recorded: list[BlackjackPlayerResult] = []

    async def record_blackjack_history(
        results: list[BlackjackPlayerResult], **_kwargs: object
    ) -> None:
        recorded.extend(results)

    monkeypatch.setattr(blackjack_views, "record_blackjack_history", record_blackjack_history)
    errors = capture_logs(monkeypatch=monkeypatch, level="error")
    message = FakeDiscordMessage()
    view = blackjack_view(round_state=round_state)

    await view.finalize(message=as_message(fake=message), interaction=None)
    await view.wait_for_background_tasks()

    assert [await get_balance(user_id=user_id) for user_id in (1, 2, 3)] == [1_100, 1_000, 1_100]
    _, alice, bob, carol = message.edits[-1]["embeds"]
    assert [embed.footer.text for embed in (alice, bob, carol)] == [
        "已結算",
        "結算失敗, 這局不算",
        "已結算",
    ]
    assert "餘額" not in cast("str", bob.description)
    assert scheduled_cleanups.messages == [message]
    assert {result.participant.user_id for result in recorded} == {1, 3}
    assert [(fields["user_id"], fields["error_type"]) for _, fields in errors] == [
        (2, "OperationalError")
    ]


@pytest.mark.parametrize(
    ("failure", "level", "traceback"),
    [
        (make_forbidden(message="Missing Access"), "warn", False),
        (make_not_found(message="Unknown Message"), "info", False),
        (make_server_error(), "warn", True),
    ],
    ids=["refused", "message_gone", "broke"],
)
async def test_a_table_edit_on_the_way_to_settling_that_fails_is_reported(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, level: str, traceback: bool
) -> None:
    """The disable and both peek stages only show the round moving, so a failure only logs."""
    monkeypatch.setattr(blackjack_views, "PEEK_REVEAL_DELAY_SECONDS", 0)
    reports = capture_levels(monkeypatch=monkeypatch, levels=("info", "warn"))
    message = FakeDiscordMessage()
    message.edit_failure = failure
    view = blackjack_view(
        round_state=blackjack_round(
            hands=[[card(rank="10"), card(rank="8", suit="♥")]],
            dealer=[card(rank="A", suit="♣"), card(rank="K", suit="♦")],
        )
    )

    await view._safe_edit_view_locked(message=as_message(fake=message), interaction=None)
    await view._animate_peek_locked(message=as_message(fake=message), interaction=None)

    # order-contract: the three edits are awaited one after another, and each reports the same.
    assert [(name, "_exc_info" in fields) for name, _, fields in reports] == [
        (level, traceback)
    ] * 3


async def test_blackjack_view_dealer_plays_h17_rule(scheduled_cleanups: ScheduledDeletes) -> None:
    """Dealer plays deterministically under H17 (hits below 17, stands on hard 17+)."""
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]],
        dealer=[card(rank="10", suit="♣"), card(rank="3", suit="♦")],
        seats=[seat(bet=50, balance_at_start=100)],
    )
    round_state.shoe = [card(rank="5", suit="♣")]

    message = FakeDiscordMessage()
    view = blackjack_view(round_state=round_state)

    await view.finalize(message=as_message(fake=message), interaction=None)

    assert [str(card) for card in view.round_state.dealer] == ["10♣", "3♦", "5♣"]
    assert view.round_state.dealer_played is True
    assert await get_balance(user_id=1) == 50
    ledger = await get_casino_ledger()
    assert ledger.balance == 50
    assert "embeds" not in message.edits[0]
    final_embeds = message.edits[1]["embeds"]
    description = cast("str", final_embeds[0].description)
    assert "-# 動作: 規則: 13 hit 抽 5♣ → 18；規則: 18 stand" in description.splitlines()
    await view.wait_for_background_tasks()
    assert scheduled_cleanups.messages == [message]


async def test_a_five_card_twenty_one_still_waits_for_the_dealer_to_play(
    scheduled_cleanups: ScheduledDeletes,
) -> None:
    """過五關 wins whatever the dealer holds, except at 21, where the dealer still plays.

    The dealer's 16 draws to 21, so the hand pushes and only the five-card bonus is paid.
    """
    await seed_balance(user_id=1, name="alice", amount=1_000)
    round_state = blackjack_round(
        hands=[
            [
                card(rank="2"),
                card(rank="3", suit="♥"),
                card(rank="4", suit="♣"),
                card(rank="5", suit="♦"),
                card(rank="7"),
            ]
        ],
        dealer=[card(rank="10", suit="♣"), card(rank="6", suit="♦")],
        seats=[seat(bet=100, balance_at_start=1_000)],
    )
    round_state.shoe = [card(rank="5", suit="♣")]
    view = blackjack_view(round_state=round_state)

    await view.finalize(message=as_message(fake=FakeDiscordMessage()), interaction=None)
    await view.wait_for_background_tasks()

    assert [str(card) for card in round_state.dealer] == ["10♣", "6♦", "5♣"]
    assert await get_balance(user_id=1) == 1_100


async def test_blackjack_view_locks_actions_while_finalizing(
    monkeypatch: pytest.MonkeyPatch, scheduled_cleanups: ScheduledDeletes
) -> None:
    """A late Hit cannot mutate a hand that is already finalizing from Stand."""
    settlement_started = asyncio.Event()
    continue_settlement = asyncio.Event()

    async def delayed_settle_blackjack_player(**_kwargs: object) -> BlackjackPlayerSettlement:
        """Blocks settlement until the test releases the finalization lock."""
        settlement_started.set()
        await continue_settlement.wait()
        return BlackjackPlayerSettlement(
            outcome="win",
            delta=50,
            new_balance=150,
            casino_balance=-50,
            base_delta=50,
            hands=[
                BlackjackHandSettlement(
                    cards=[card(rank="10"), card(rank="Q", suit="♥")],
                    bet=50,
                    outcome="win",
                    delta=50,
                )
            ],
        )

    monkeypatch.setattr(
        blackjack_views, "settle_blackjack_player", delayed_settle_blackjack_player
    )

    message = FakeDiscordMessage()
    view = blackjack_view(
        round_state=blackjack_round(
            hands=[[card(rank="10"), card(rank="Q", suit="♥")]],
            dealer=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
            seats=[seat(bet=50, balance_at_start=50)],
        )
    )

    hit_button = attached_button(view=view, custom_id="bj:hit")
    stand_button = attached_button(view=view, custom_id="bj:stand")
    stand_task = asyncio.create_task(
        coro=stand_button.callback(as_interaction(fake=FakeInteraction(message=message)))
    )
    await settlement_started.wait()

    assert view.is_finished(), "the view stopped taking presses before it settled"
    assert len(message.edits) == 1
    in_flight_view = cast("BlackjackView", message.edits[0]["view"])
    assert all(child.disabled for child in in_flight_view.children if isinstance(child, Button))

    hit_task = asyncio.create_task(
        coro=hit_button.callback(as_interaction(fake=FakeInteraction(message=message)))
    )
    await asyncio.sleep(delay=0)

    assert len(view.round_state.players[0].hands[0].cards) == 2
    continue_settlement.set()
    await asyncio.gather(stand_task, hit_task)

    assert len(view.round_state.players[0].hands[0].cards) == 2
    assert "embeds" not in message.edits[0]
    await view.wait_for_background_tasks()
    assert len(message.edits) == 2
    assert message.edits[1]["view"] is None
    assert scheduled_cleanups.messages == [message]


class _HeldEditInteraction(FakeInteraction):
    """Holds its first edit open until the test releases it, like a slow Discord round trip."""

    def __init__(self, message: FakeDiscordMessage) -> None:
        """Initializes the hold and release signals beside the recorded edits."""
        super().__init__(message=message)
        self.holding = asyncio.Event()
        self.release = asyncio.Event()

    async def edit_original_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- Discord kwargs
        """Blocks the first edit until released, then records it."""
        if not self.holding.is_set():
            self.holding.set()
            await self.release.wait()
        await super().edit_original_message(**kwargs)


class _ContendedLock(asyncio.Lock):
    """Signals the moment a second caller has to wait for it."""

    def __init__(self) -> None:
        """Initializes the contention signal."""
        super().__init__()
        self.contended = asyncio.Event()

    async def acquire(self) -> Literal[True]:
        """Records contention before waiting for the lock like any caller."""
        if self.locked():
            self.contended.set()
        return await super().acquire()


async def test_a_timeout_waits_for_the_action_in_flight(
    scheduled_cleanups: ScheduledDeletes,
) -> None:
    """A timeout that lands mid-action settles after it, so the settled table is what stays up.

    The Hit is held inside its table edit when the timeout fires. Settling underneath it would
    let that held edit land last, putting the pre-settlement table and its buttons back over
    the final one.
    """
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="2", suit="♥")]],
        dealer=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
        seats=[seat(bet=50, balance_at_start=100)],
    )
    round_state.shoe = [card(rank="5", suit="♣")]
    message = FakeDiscordMessage()
    view = blackjack_view(round_state=round_state)
    view.message = as_message(fake=message)
    lock = _ContendedLock()
    view._round_lock = lock
    hit_press = _HeldEditInteraction(message=message)

    hit = asyncio.create_task(
        coro=attached_button(view=view, custom_id="bj:hit").callback(
            as_interaction(fake=hit_press)
        )
    )
    await hit_press.holding.wait()
    timeout = asyncio.create_task(coro=view.on_timeout())
    contended = asyncio.create_task(coro=lock.contended.wait())
    await asyncio.wait({timeout, contended}, return_when=asyncio.FIRST_COMPLETED)
    hit_press.release.set()
    await asyncio.gather(hit, timeout)
    contended.cancel()
    await asyncio.gather(contended, return_exceptions=True)
    await view.wait_for_background_tasks()

    assert [str(card) for card in round_state.players[0].hands[0].cards] == ["10♠", "2♥", "5♣"]
    assert await get_balance(user_id=1) == 50
    assert message.edits[-1]["view"] is None
    assert scheduled_cleanups.messages == [message]


# Final table embeds --------------------------------------------------------


def _five_card_round(last_card: str, dealer_cards: list[Card]) -> BlackjackRound:
    """Builds a settled one-seat round whose hand is 2-3-4-5 plus `last_card`."""
    round_state = blackjack_round(
        hands=[
            [
                card(rank="2"),
                card(rank="3", suit="♥"),
                card(rank="4", suit="♣"),
                card(rank="5", suit="♦"),
                card(rank=last_card),
            ]
        ],
        dealer=dealer_cards,
        seats=[seat(bet=10_000, balance_at_start=100_000)],
        finished=True,
    )
    return round_state


async def test_blackjack_final_embed_shows_five_card_bonus_metadata() -> None:
    """Final Blackjack embeds display five-card outcome and bonus metadata."""
    await seed_balance(user_id=1, name="alice", amount=100_000)
    round_state = _five_card_round(
        last_card="7", dealer_cards=[card(rank="10", suit="♣"), card(rank="9", suit="♦")]
    )
    settlement = await settle_only_seat(round_state=round_state)

    embeds = build_final_embeds(
        round_state=round_state,
        results=[
            BlackjackPlayerResult(
                participant=round_state.players[0].participant, settlement=settlement
            )
        ],
    )

    description = cast("str", embeds[1].description)
    assert "## ✨ 過五關 · 21" in description
    assert "過五關 bonus `+1萬`" in description


async def test_blackjack_final_embed_shows_five_card_win_without_bonus_metadata() -> None:
    """Final Blackjack embeds display non-21 five-card wins without bonus metadata."""
    await seed_balance(user_id=1, name="alice", amount=100_000)
    round_state = _five_card_round(
        last_card="6",
        dealer_cards=[
            card(rank="7", suit="♣"),
            card(rank="7", suit="♦"),
            card(rank="7", suit="♥"),
        ],
    )
    settlement = await settle_only_seat(round_state=round_state)

    embeds = build_final_embeds(
        round_state=round_state,
        results=[
            BlackjackPlayerResult(
                participant=round_state.players[0].participant, settlement=settlement
            )
        ],
    )

    description = cast("str", embeds[1].description)
    assert "## 🎉 過五關 · 20" in description
    assert "過五關 bonus" not in description


async def test_blackjack_final_embed_uses_aggregate_insurance_push_title() -> None:
    """Insurance break-even should present as aggregate push in the final title."""
    await seed_balance(user_id=1, name="alice", amount=300)
    round_state = blackjack_round(
        hands=[[card(rank="9"), card(rank="8", suit="♥")]],
        dealer=[card(rank="K", suit="♣"), card(rank="A", suit="♦")],
        seats=[seat(bet=100, balance_at_start=100)],
        finished=True,
    )
    player = round_state.players[0]
    player.insurance_bet = 50
    player.insurance_resolved = True
    round_state.peeked_blackjack = True

    settlement = await settle_only_seat(round_state=round_state)
    embeds = build_final_embeds(
        round_state=round_state,
        results=[BlackjackPlayerResult(participant=player.participant, settlement=settlement)],
    )

    description = cast("str", embeds[1].description)
    assert "## 😢 你輸了 · 17 < 21" in description
    assert "保險 `50` → 中獎 `+100`" in description
