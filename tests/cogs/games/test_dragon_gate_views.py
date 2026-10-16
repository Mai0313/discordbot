"""Tests for the 射龍門 table view: its embeds, controls, presses and refusals."""

from __future__ import annotations

import pytest

from discordbot.typings.games import GameParticipant
from discordbot.utils.discord_embeds import embed_text_length
from discordbot.cogs.games.dragon_gate import (
    ANTE,
    GAME_ID,
    DragonGateRound,
    DragonGateTurnResult,
    DragonGatePlayerResult,
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
    joins_as,
    lobby_button,
    component_ids,
    component_rows,
    everyone_stays,
    attached_button,
    attached_select,
)
from tests.helpers.casting import as_message, as_interaction, make_forbidden
from tests.helpers.economy import get_jackpot_pool
from tests.helpers.dragon_gate import (
    RiggedRandom,
    dragon_gate_seat,
    funded_dragon_gate_seat,
    record_jackpot_settlements,
)
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.message_cleanup import record_scheduled_deletes


def test_dragon_gate_embeds_show_lobby_progress_and_final_state() -> None:
    """Each embed carries what a player reads off it: who sits, whose turn, how the table ended."""
    owner = dragon_gate_seat(user_id=1, display_name="Alice")
    bob = dragon_gate_seat(user_id=2, display_name="Bob")
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

    result = round_state.resolve_bet(user_id=1, amount=10_000, jackpot=110_000)
    assert result.outcome == "gate_win"
    round_state.record_result(result=result)
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


async def test_dragon_gate_controls_hide_unavailable_actions() -> None:
    """Active controls are removed instead of left visible but disabled."""
    owner = dragon_gate_seat(user_id=1, display_name="Alice")

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
    owner = dragon_gate_seat(user_id=1, display_name="Alice")
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


async def test_a_lobby_in_a_channel_the_bot_was_shut_out_of_still_opens_and_plays_its_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every press edits the message it sits on through its own token, not through the channel.

    The lobby went up on the slash command's token, so it shows in a channel the server shut the
    bot out of afterwards, where every channel edit is refused. A start edited through the
    channel charges the antes for a table that never appears.
    """
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    bob = await funded_dragon_gate_seat(user_id=2, display_name="Bob")
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)

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


async def test_dragon_gate_view_rejects_non_active_and_invalid_custom_bet() -> None:
    """Only the active player can bet; the leave button is open to all seated."""
    alice = dragon_gate_seat(user_id=1, display_name="Alice")
    bob = dragon_gate_seat(user_id=2, display_name="Bob")
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
    participants = [await funded_dragon_gate_seat(user_id=1, display_name="Alice")]
    if bob:
        participants.append(await funded_dragon_gate_seat(user_id=2, display_name="Bob"))
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


@pytest.mark.parametrize(argnames="control", argvalues=["custom", "min", "higher", "leave"])
async def test_an_action_that_reaches_a_settled_table_is_told_it_is_over(
    monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    """An action landing after the table settled hears what a press on it does.

    The custom bet form never passes the table's check, and a press that passed it before the
    table settled can reach the table only after; neither may end in silence.
    """
    alice = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")), participants=[alice]
    )
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)
    message = FakeDiscordMessage()
    view = DragonGateView(
        round_state=round_state,
        owner=alice,
        jackpot_snapshot=100_000,
        final_balances={1: 1_000_000},
    )
    view.message = as_message(fake=message)
    await view.on_timeout()
    assert view._settled
    press = FakeInteraction(user=FakeUser(user_id=1), message=message)
    interaction = as_interaction(fake=press)

    if control == "custom":
        await view.submit_custom_bet(interaction=interaction, raw_amount="20000")
    elif control == "min":
        await view._handle_bet_choice(choice="min", interaction=interaction)
    elif control == "higher":
        await view._choose_direction(interaction=interaction, direction="higher")
    else:
        await view._handle_leave(interaction=interaction)

    assert press.followup.sent == [{"content": "這桌已經結束, 等下一桌吧", "ephemeral": True}]
    assert settlements == []


async def test_dragon_gate_custom_bet_modal_allows_formatted_maximum() -> None:
    """Custom bet input length matches the comma-stripping parser."""
    owner = dragon_gate_seat(user_id=1, display_name="Alice")
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


@pytest.mark.parametrize(argnames="expired", argvalues=[False, True], ids=["live", "expired"])
@pytest.mark.parametrize(argnames="last", argvalues=["start", "bet", "direction", "leave"])
async def test_a_dragon_gate_table_left_to_time_out_in_a_shut_out_channel_closes_through_its_last_press(
    monkeypatch: pytest.MonkeyPatch, last: str, expired: bool
) -> None:
    """A timeout has no press of its own, so it renders and deletes through the last one.

    Whichever control that press was, the start included. Once its token has expired only the
    channel is left, which refuses the render.
    """
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    bob = await funded_dragon_gate_seat(user_id=2, display_name="Bob")
    record_jackpot_settlements(monkeypatch=monkeypatch)
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
    result = round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)

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
        dragon_gate_seat(user_id=index, display_name=longest_name, balance=10**15)
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
