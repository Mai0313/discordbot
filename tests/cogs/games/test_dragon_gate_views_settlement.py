"""Tests for 射龍門 bets, leaves and timeouts settling against the jackpot."""

from __future__ import annotations

from typing import Any
import sqlite3

import pytest
from nextcord import Embed
from sqlalchemy.exc import OperationalError

from discordbot.typings.economy import JackpotSnapshot, JackpotSettlementResult
from discordbot.cogs.games.dragon_gate import GAME_ID, DragonGateRound
from discordbot.services.economy.database import get_jackpot_snapshot, apply_jackpot_settlement
from discordbot.cogs.games.dragon_gate_views import DragonGateView

from tests.helpers.games import component_ids, attached_button, attached_select
from tests.helpers.casting import as_message, as_interaction, make_forbidden
from tests.helpers.economy import seed_balance, get_jackpot_pool
from tests.helpers.dragon_gate import (
    RiggedRandom,
    dragon_gate_seat,
    funded_dragon_gate_seat,
    record_jackpot_settlements,
)
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_logs
from tests.helpers.message_cleanup import record_scheduled_deletes
from tests.helpers.economy_invariants import assert_wallet_consistent


async def test_dragon_gate_view_pair_choice_bet_settles_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bet calls apply_jackpot_settlement and updates the live snapshot."""
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "8", "♣")), participants=[owner]
    )
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)
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
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice", balance=100)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)
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
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice", balance=15)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)

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
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice", balance=500_000)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)
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
    owner = dragon_gate_seat(user_id=1, display_name="Alice")
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
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice", balance=30)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "3", "♣")), participants=[owner]
    )
    record_jackpot_settlements(monkeypatch=monkeypatch)
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
    alice = await funded_dragon_gate_seat(user_id=1, display_name="Alice", balance=30)
    bob = await funded_dragon_gate_seat(user_id=2, display_name="Bob", balance=100_000)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "3", "♣")), participants=[alice, bob]
    )
    record_jackpot_settlements(monkeypatch=monkeypatch)
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
    alice = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    bob = dragon_gate_seat(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣", "4", "♦", "Q", "♣")),
        participants=[alice, bob],
    )
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)
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
    owner = dragon_gate_seat(user_id=1, display_name="Alice", balance=1_000)
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[owner]
    )
    # Live wallet dropped to 100 (player spent elsewhere mid-round); the in-table
    # cache still shows the post-ante 1,000.
    await seed_balance(user_id=1, name=owner.account_name, amount=100)
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)

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
    alice = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    bob = dragon_gate_seat(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "K", "♣", "4", "♦", "Q", "♣")),
        participants=[alice, bob],
    )
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)

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


@pytest.mark.parametrize(
    argnames=("third_cards", "paid_bets", "written"),
    argvalues=[(("7", "♣"), 0, []), (("K", "♣", "4", "♦", "Q", "♣", "7", "♣"), 1, [20, -20])],
    ids=["unpaid-win", "unpaid-loss-then-paid-win"],
)
async def test_a_bet_whose_settlement_raises_counts_nowhere_at_the_table(
    monkeypatch: pytest.MonkeyPatch,
    third_cards: tuple[str, ...],
    paid_bets: int,
    written: list[int],
) -> None:
    """A bet the ledger never wrote leaves no trace, so leaving takes back only paid winnings.

    Counted, an unpaid win is taken from the wallet on leave, and an unpaid loss hides a later
    paid win from that clawback.
    """
    alice = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", *third_cards)), participants=[alice]
    )
    record_scheduled_deletes(monkeypatch=monkeypatch)
    failures = iter([
        OperationalError("bet", None, sqlite3.OperationalError("database is locked"))
    ])
    settled: list[int] = []

    async def settle(**kwargs: Any) -> JackpotSettlementResult:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Refuses the first write as a locked database would, then settles every later one."""
        failure = next(failures, None)
        if failure is not None:
            raise failure
        settled.append(kwargs["player_delta"])
        return await apply_jackpot_settlement(**kwargs)

    monkeypatch.setattr("discordbot.cogs.games.dragon_gate_views.apply_jackpot_settlement", settle)
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

    refused = FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
    # Called directly, the press skips the view's on_error, so the raise reaches the test.
    with pytest.raises(OperationalError):
        await view._handle_bet_choice(choice="min", interaction=as_interaction(fake=refused))

    assert refused.followup.sent == [{"content": "下注失敗, 這注不算", "ephemeral": True}]
    assert round_state.player_delta(user_id=1) == 0
    assert round_state.last_result is None
    assert round_state.turn_number == 1, "the same gate takes the next bet"
    assert view._history == []

    for _ in range(paid_bets):
        await view._handle_bet_choice(
            choice="min",
            interaction=as_interaction(
                fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:bet")
            ),
        )
    await attached_button(view=view, custom_id="dg:leave").callback(
        as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:leave")
        )
    )

    # order-contract: the leave hands back winnings a bet already wrote.
    assert settled == written
    await assert_wallet_consistent(user_id=1, expected_balance=1_000_000)
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before


async def test_a_leave_whose_refund_raises_keeps_the_player_seated_until_one_lands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leave takes the player off the table only once its 逆贏不拿 refund is written.

    Withdrawn first, a player whose refund raised walked off with the winnings: a second press
    was turned away, and nothing refunds a withdrawn seat later. A press queued behind the one
    that landed must not refund twice.
    """
    alice = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    bob = dragon_gate_seat(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[alice, bob]
    )
    record_scheduled_deletes(monkeypatch=monkeypatch)
    failures = iter([
        OperationalError("refund", None, sqlite3.OperationalError("database is locked"))
    ])
    settled: list[int] = []

    async def settle(**kwargs: Any) -> JackpotSettlementResult:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Refuses the first refund as a locked database would, then settles every write."""
        failure = next(failures, None) if kwargs["player_delta"] < 0 else None
        if failure is not None:
            raise failure
        settled.append(kwargs["player_delta"])
        return await apply_jackpot_settlement(**kwargs)

    monkeypatch.setattr("discordbot.cogs.games.dragon_gate_views.apply_jackpot_settlement", settle)
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

    refused = FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:leave")
    # Called directly, the press skips the view's on_error, so the raise reaches the test.
    with pytest.raises(OperationalError):
        await leave_button.callback(as_interaction(fake=refused))

    assert refused.followup.sent == [{"content": "離桌失敗, 請再按一次離桌", "ephemeral": True}]
    assert round_state.is_active(user_id=1)
    assert view._refunded_to_pool == {}

    await leave_button.callback(
        as_interaction(
            fake=FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:leave")
        )
    )
    queued = FakeInteraction(user=FakeUser(user_id=1), message=message, custom_id="dg:leave")
    # Called directly, the press skips the view's interaction check, as one queued on the round
    # lock behind the press that landed has already passed it.
    await leave_button.callback(as_interaction(fake=queued))

    assert queued.followup.sent == [{"content": "你不在這桌", "ephemeral": True}]
    assert round_state.is_active(user_id=1) is False
    # order-contract: the leave hands back winnings the bet already wrote.
    assert settled == [20, -20]
    assert view._refunded_to_pool == {1: 20}
    await assert_wallet_consistent(user_id=1, expected_balance=1_000_000)
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before


async def test_dragon_gate_view_timeout_refunds_remaining_winners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout refunds positive running deltas back into the jackpot."""
    alice = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")), participants=[alice]
    )
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)
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


async def test_a_timeout_whose_refund_raises_for_one_seat_still_settles_the_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refund that raises on timeout costs that seat's refund alone, never the table's close.

    Raising out of the timeout skipped every later seat's refund and the close itself, so the
    table showed no result and stayed up until a restart. The seat whose refund never landed
    keeps its winnings, and the final table claims no refund for it.
    """
    alice = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    bob = await funded_dragon_gate_seat(user_id=2, display_name="Bob")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣", "4", "♦", "Q", "♣", "7", "♦")),
        participants=[alice, bob],
    )
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    errors = capture_logs(monkeypatch=monkeypatch, level="error")

    async def settle(**kwargs: Any) -> JackpotSettlementResult:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Refuses Alice's refund as a locked database would, then settles every other write."""
        if kwargs["player_id"] == 1 and kwargs["player_delta"] < 0:
            raise OperationalError("refund", None, sqlite3.OperationalError("database is locked"))
        return await apply_jackpot_settlement(**kwargs)

    monkeypatch.setattr("discordbot.cogs.games.dragon_gate_views.apply_jackpot_settlement", settle)
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
    for user_id in (1, 2):
        await view._handle_bet_choice(
            choice="min",
            interaction=as_interaction(
                fake=FakeInteraction(
                    user=FakeUser(user_id=user_id), message=message, custom_id="dg:bet"
                )
            ),
        )
    assert [round_state.player_delta(user_id=user_id) for user_id in (1, 2)] == [20, 20]

    await view.on_timeout()

    assert view._settled
    assert message.edits[-1]["view"] is None, "the final table landed"
    assert scheduled.messages == [message]
    assert view._refunded_to_pool == {2: 20}
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before - 20
    assert [fields["user_id"] for _, fields in errors] == [1]
    final = message.edits[-1]["embeds"]
    assert isinstance(final, list)
    assert isinstance(final[0].description, str)
    assert final[0].description.count("逆贏退回") == 1
