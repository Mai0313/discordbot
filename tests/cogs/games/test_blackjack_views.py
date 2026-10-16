"""Tests for the Blackjack table view: its controls, rendering, presses, bot turns and start.

Controls are presence-based, so the button tests assert which custom_ids are
attached rather than which are disabled. The bot's decisions come from the EV
engine, so they are asserted exactly.
"""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random
from typing import NoReturn, cast
import contextlib

import pytest
from nextcord import HTTPException
from nextcord.ui import Button

from discordbot.cogs.games import blackjack_views
from discordbot.typings.games import BotAction, GameParticipant
from discordbot.cogs.games.database import fetch_recent_blackjack_rounds
from discordbot.cogs.games.blackjack import Card, BlackjackRound, BlackjackPlayerHand
from discordbot.services.economy.database import get_balance
from discordbot.cogs.games.blackjack_views import BlackjackView, build_in_progress_embeds

from tests.helpers.games import (
    card,
    seat,
    lobby_button,
    component_ids,
    blackjack_hand,
    blackjack_view,
    component_rows,
    attached_button,
    blackjack_round,
    scripted_blackjack_lobby,
)
from tests.helpers.casting import (
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
)
from tests.helpers.economy import seed_balance
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_logs, capture_levels
from tests.helpers.message_cleanup import ScheduledDeletes, record_scheduled_deletes


def _button_states(view: BlackjackView) -> dict[str, bool]:
    """Returns `{custom_id: disabled}` for every button in the view."""
    states: dict[str, bool] = {}
    for child in view.children:
        cid = getattr(child, "custom_id", None)
        if cid is not None and isinstance(child, Button):
            states[cid] = bool(child.disabled)
    return states


async def test_action_controls_show_on_their_rows_and_leave_once_not_allowed() -> None:
    """A fresh pair shows all five actions on their rows; a Hit removes the first-action ones.

    Which actions a hand allows is the round's rule; the view's part is that an action it no
    longer allows leaves the view rather than staying behind disabled.
    """
    round_state = blackjack_round(
        hands=[[card(rank="8"), card(rank="8", suit="♥")]],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = blackjack_view(round_state=round_state)
    view.sync_buttons()

    assert component_rows(view=view) == {
        "bj:hit": 0,
        "bj:stand": 0,
        "bj:double": 1,
        "bj:split": 1,
        "bj:surrender": 1,
    }
    assert all(disabled is False for disabled in _button_states(view=view).values())

    round_state.shoe = [card(rank="2")]
    round_state.hit(user_id=1)
    view.sync_buttons()

    assert component_ids(view=view) == {"bj:hit", "bj:stand"}
    assert all(disabled is False for disabled in _button_states(view=view).values())


async def test_insurance_phase_hides_action_buttons_and_shows_insurance() -> None:
    """During insurance only insure_yes / insure_no are interactive."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="5", suit="♥")]],
        dealer=[card(rank="A", suit="♣"), card(rank="9", suit="♦")],
    )
    round_state.phase = "insurance"
    view = blackjack_view(round_state=round_state)
    view.sync_buttons()

    states = _button_states(view=view)
    assert component_ids(view=view) == {"bj:insure_yes", "bj:insure_no"}
    assert states["bj:insure_yes"] is False
    assert states["bj:insure_no"] is False
    assert component_rows(view=view) == {"bj:insure_yes": 1, "bj:insure_no": 1}


async def test_settled_phase_removes_every_button() -> None:
    """After settlement no controls remain attached to the view."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="9", suit="♥")]],
        dealer=[card(rank="K", suit="♣"), card(rank="7", suit="♦")],
    )
    round_state.phase = "settled"
    view = blackjack_view(round_state=round_state)
    view.sync_buttons()

    assert component_ids(view=view) == set()


async def test_sync_buttons_drops_insurance_controls_outside_insurance() -> None:
    """Insurance buttons join the view for the insurance phase alone and leave it after."""
    round_state = blackjack_round(
        hands=[[card(rank="8"), card(rank="8", suit="♥")]],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = blackjack_view(round_state=round_state)
    view.sync_buttons()

    ids = component_ids(view=view)
    assert "bj:insure_yes" not in ids
    assert "bj:insure_no" not in ids

    round_state.phase = "insurance"
    view.sync_buttons()

    ids = component_ids(view=view)
    assert "bj:insure_yes" in ids
    assert "bj:insure_no" in ids

    round_state.phase = "player_actions"
    view.sync_buttons()

    ids = component_ids(view=view)
    assert "bj:insure_yes" not in ids
    assert "bj:insure_no" not in ids


async def test_build_in_progress_embeds_force_show_hole_reveals_dealer_total() -> None:
    """`force_show_hole=True` flips the dealer hole card face-up for peek reveal."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]],
        dealer=[card(rank="A", suit="♣"), card(rank="K", suit="♦")],
    )

    embeds = build_in_progress_embeds(round_state=round_state, force_show_hole=True)
    dealer_embed = embeds[0]

    assert isinstance(dealer_embed.description, str)
    assert "A♣" in dealer_embed.description
    assert "K♦" in dealer_embed.description
    assert "🂠" not in dealer_embed.description


def test_blackjack_in_progress_dealer_seat_hides_hole_card() -> None:
    """The dealer seat embed shows one hidden card marker plus the visible up-card."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]],
        dealer=[card(rank="8", suit="♣"), card(rank="K", suit="♦")],
    )

    embeds = build_in_progress_embeds(round_state=round_state)
    dealer_embed = embeds[0]

    assert isinstance(dealer_embed.description, str)
    assert "🂠" in dealer_embed.description
    assert "K♦" in dealer_embed.description
    assert "8♣" not in dealer_embed.description


def test_blackjack_in_progress_dealer_seat_single_card_is_visible() -> None:
    """A one-card dealer fallback should not render as a hidden hole card."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]], dealer=[card(rank="8", suit="♣")]
    )

    embeds = build_in_progress_embeds(round_state=round_state)
    dealer_embed = embeds[0]

    assert isinstance(dealer_embed.description, str)
    assert "8♣" in dealer_embed.description
    assert "🂠" not in dealer_embed.description


async def test_interaction_check_sends_ephemeral_notice_when_settled() -> None:
    """After the round is settled, clicks get an ephemeral notice rather than silent ignore."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]],
        dealer=[card(rank="K", suit="♣"), card(rank="9", suit="♦")],
    )
    view = blackjack_view(round_state=round_state)
    view._settled = True
    press = FakeInteraction()

    allowed = await view.interaction_check(interaction=as_interaction(fake=press))

    assert allowed is False
    assert press.followup.sent == [{"content": "這局已經結束, 等下一局吧", "ephemeral": True}]


@pytest.mark.parametrize(
    argnames="custom_id",
    argvalues=[
        "bj:hit",
        "bj:stand",
        "bj:double",
        "bj:split",
        "bj:surrender",
        "bj:insure_yes",
        "bj:insure_no",
    ],
)
async def test_a_press_that_reaches_a_settled_round_is_told_it_is_over(
    custom_id: str, scheduled_cleanups: ScheduledDeletes
) -> None:
    """A press checked while the round was open but run after it settled hears it is over.

    nextcord checks a press and runs its callback as two steps, so a press can pass the check
    before the timeout settles the round and only reach the round after.
    """
    await seed_balance(user_id=1, name="alice", amount=1_000)
    insurance = custom_id.startswith("bj:insure")
    round_state = blackjack_round(
        hands=[[card(rank="8"), card(rank="8", suit="♥")]],
        dealer=[card(rank="A" if insurance else "5", suit="♣"), card(rank="6", suit="♦")],
    )
    if insurance:
        round_state.phase = "insurance"
    message = FakeDiscordMessage()
    view = blackjack_view(round_state=round_state)
    view.message = as_message(fake=message)
    button = attached_button(view=view, custom_id=custom_id)
    await view.on_timeout()
    await view.wait_for_background_tasks()
    settled_balance = await get_balance(user_id=1)
    press = FakeInteraction(message=message)

    await button.callback(as_interaction(fake=press))

    assert press.followup.sent == [{"content": "這局已經結束, 等下一局吧", "ephemeral": True}]
    assert press.edits == []
    assert await get_balance(user_id=1) == settled_balance
    assert scheduled_cleanups.messages == [message]


async def test_a_seat_that_can_never_insure_is_not_sent_to_refresh() -> None:
    """A 1-point seat's half-bet rounds to zero, and no newer table will change that.

    The insurance buttons belong to the table rather than to a seat — every undecided player
    sees the same pair — so the seat that cannot use them only finds out by pressing, and what
    it is told then is the whole of the feature for it.
    """
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]],
        dealer=[card(rank="A", suit="♣"), card(rank="9", suit="♦")],
        seats=[seat(bet=1)],
    )
    round_state.phase = "insurance"
    view = blackjack_view(round_state=round_state)
    message = FakeDiscordMessage()
    press = FakeInteraction(message=message)

    decided = await view._take_insurance_locked(
        interaction=as_interaction(fake=press), message=as_message(fake=message), user_id=1
    )

    assert decided is False
    assert round_state.players[0].insurance_bet == 0
    assert press.followup.sent == [
        {"content": "你的下注太小，一半不到 1 點，這局沒有保險可買", "ephemeral": True}
    ]


async def test_bot_dispatcher_skips_when_active_player_is_human() -> None:
    """If the active seat belongs to a human, the bot dispatcher returns immediately."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = blackjack_view(round_state=round_state)
    view.bot_user_id = 999
    message = FakeDiscordMessage()
    press = FakeInteraction(message=message)
    await view._maybe_play_bot_turn_locked(
        message=as_message(fake=message), interaction=as_interaction(fake=press)
    )
    # The human's hand is untouched because the bot never acts on a human seat.
    assert press.edits == []
    assert len(round_state.players[0].hands[0].cards) == 2


async def test_bot_dispatcher_breaks_when_action_does_not_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-op bot dispatch exits instead of spinning on the same turn."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = blackjack_view(round_state=round_state)
    view.bot_user_id = 1
    calls = 0

    async def no_op_dispatch(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(view, "_dispatch_bot_action_locked", no_op_dispatch)
    message = FakeDiscordMessage()

    await view._maybe_play_bot_turn_locked(
        message=as_message(fake=message),
        interaction=as_interaction(fake=FakeInteraction(message=message)),
    )

    assert calls == 1


async def test_a_bot_that_has_decided_insurance_waits_for_the_other_seats() -> None:
    """Once the bot's own insurance is settled, the table's decision belongs to the humans.

    The phase stays open until every seat decides, so the bot must not take another turn there:
    it would be refused by the round and still re-render the table on every retry.
    """
    round_state = blackjack_round(
        hands=[
            [card(rank="10"), card(rank="7", suit="♥")],
            [card(rank="9", suit="♣"), card(rank="8", suit="♦")],
        ],
        dealer=[card(rank="A", suit="♣"), card(rank="9", suit="♦")],
        seats=[seat(user_id=1, display_name="Bot"), seat(user_id=2, display_name="Bob")],
    )
    round_state.phase = "insurance"
    round_state.players[0].insurance_resolved = True
    view = blackjack_view(round_state=round_state)
    view.bot_user_id = 1
    message = FakeDiscordMessage()

    await view.maybe_play_bot_turn(
        message=as_message(fake=message),
        interaction=as_interaction(fake=FakeInteraction(message=message)),
    )

    assert message.edits == []
    assert round_state.phase == "insurance"
    assert round_state.players[1].insurance_resolved is False


async def test_a_bot_call_closing_insurance_without_a_natural_keeps_the_hole_face_down() -> None:
    """The bot's insurance call can be the last one, and a peek finding no natural shows nothing.

    The hole is the bot's private edge, so the table re-rendered after that call still hides it.
    """
    round_state = blackjack_round(
        hands=[
            [card(rank="9", suit="♣"), card(rank="8", suit="♦")],
            [card(rank="10"), card(rank="7", suit="♥")],
        ],
        dealer=[card(rank="9", suit="♦"), card(rank="A", suit="♣")],
        seats=[seat(user_id=2, display_name="Bob"), seat(user_id=1, display_name="Bot")],
    )
    round_state.phase = "insurance"
    round_state.players[0].insurance_resolved = True
    view = blackjack_view(round_state=round_state)
    view.bot_user_id = 1
    message = FakeDiscordMessage()

    await view.maybe_play_bot_turn(
        message=as_message(fake=message),
        interaction=as_interaction(fake=FakeInteraction(message=message)),
    )

    assert round_state.phase == "player_actions"
    dealer_seats = [cast("str", edit["embeds"][0].description) for edit in message.edits]
    assert dealer_seats, "the bot's call re-renders the table"
    assert all("🂠" in dealer_seat for dealer_seat in dealer_seats), "the hole stays face down"


async def test_a_bot_bet_too_small_to_insure_declines_without_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1-point bet has no half to insure, so a bot that wants insurance declines instead.

    The refusal is expected rather than a failure, so it is an info record with no traceback.
    """
    round_state = blackjack_round(
        hands=[
            [card(rank="10"), card(rank="7", suit="♥")],
            [card(rank="9", suit="♣"), card(rank="8", suit="♦")],
        ],
        dealer=[card(rank="9", suit="♦"), card(rank="A", suit="♣")],
        seats=[seat(user_id=1, display_name="Bot", bet=1), seat(user_id=2, display_name="Bob")],
    )
    round_state.phase = "insurance"
    # All tens: the bot's count says insurance is worth buying.
    round_state.shoe = [card(rank="10")] * 20
    view = blackjack_view(round_state=round_state)
    infos = capture_logs(monkeypatch=monkeypatch, level="info")
    warnings = capture_logs(monkeypatch=monkeypatch, level="warn")
    message = FakeDiscordMessage()

    await view._dispatch_bot_insurance_locked(
        message=as_message(fake=message),
        bot_player=round_state.players[0],
        interaction=as_interaction(fake=FakeInteraction(message=message)),
    )

    bot = round_state.players[0]
    assert (bot.insurance_resolved, bot.insurance_bet) == (True, 0)
    assert warnings == []
    assert ("Bot bet too small to insure; declining", {"user_id": 1, "bet": 1}) in infos


async def test_a_dealt_bot_counts_the_hole_among_the_face_down_cards() -> None:
    """A king in the hole leaves the shoe at exactly a third tens, so it alone says decline.

    With the hole counted the seven face-down cards hold three tens, past a third, so the bot
    insures (#860). Dealt for real, so the hole is wherever the deal puts it.
    """
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[seat(user_id=1, display_name="Bot"), seat(user_id=2, display_name="Bob")],
    )
    round_state.shoe = [
        card(rank="9"),
        card(rank="8"),  # bot
        card(rank="7"),
        card(rank="6"),  # Bob
        card(rank="K"),  # dealer hole
        card(rank="A"),  # dealer up
        *[card(rank=rank) for rank in ("10", "J", "2", "3", "4", "5")],
    ]
    round_state.deal_initial()
    view = blackjack_view(round_state=round_state)
    message = FakeDiscordMessage()

    await view._dispatch_bot_insurance_locked(
        message=as_message(fake=message),
        bot_player=round_state.players[0],
        interaction=as_interaction(fake=FakeInteraction(message=message)),
    )

    bot = round_state.players[0]
    assert (bot.insurance_resolved, bot.insurance_bet) == (True, 50)


async def test_bot_dispatcher_paces_consecutive_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consecutive bot-owned decisions wait briefly between message edits."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = blackjack_view(round_state=round_state)
    view.bot_user_id = 1
    dispatch_calls = 0
    sleep_calls: list[float] = []

    async def fake_dispatch(**_kwargs: object) -> None:
        nonlocal dispatch_calls
        dispatch_calls += 1
        view._state_revision += 1
        if dispatch_calls == 2:
            view.round_state.stand(user_id=1)

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(view, "_dispatch_bot_action_locked", fake_dispatch)
    monkeypatch.setattr(blackjack_views.asyncio, "sleep", fake_sleep)
    message = FakeDiscordMessage()

    await view._maybe_play_bot_turn_locked(
        message=as_message(fake=message),
        interaction=as_interaction(fake=FakeInteraction(message=message)),
    )

    assert dispatch_calls == 2
    assert sleep_calls == [blackjack_views.BOT_TURN_EDIT_DELAY_SECONDS]


async def test_bot_action_plays_ev_action() -> None:
    """The bot plays the EV engine's hole-aware action where the up-card table would not.

    Hard 16 against a 10 with Surrender on offer is a surrender by the table, but the hole is a
    6 and the shoe holds only tens, so the dealer's 16 must draw and bust: the bot stands. It
    sits in the first seat so the stand hands the turn on instead of settling the table.
    """
    round_state = blackjack_round(
        hands=[
            [card(rank="10"), card(rank="6", suit="♥")],
            [card(rank="9", suit="♣"), card(rank="8", suit="♦")],
        ],
        dealer=[card(rank="6", suit="♣"), card(rank="10", suit="♦")],
        seats=[seat(user_id=1, display_name="Bot"), seat(user_id=2, display_name="Bob")],
    )
    bot_hand = round_state.players[0].hands[0]
    round_state.shoe = [card(rank="10")] * 20
    view = blackjack_view(round_state=round_state)
    message = FakeDiscordMessage()

    await view._dispatch_bot_action_locked(
        message=as_message(fake=message),
        active=round_state.players[0],
        interaction=as_interaction(fake=FakeInteraction(message=message)),
    )

    assert bot_hand.finished is True
    assert (bot_hand.surrendered, bot_hand.doubled, len(bot_hand.cards)) == (False, False, 2)
    assert round_state.active_player() is round_state.players[1]


async def test_apply_bot_action_routes_known_actions() -> None:
    """Every allowed action reaches its own `BlackjackRound` method and is reported applied."""

    def _apply(action: BotAction, player_cards: list[Card]) -> BlackjackPlayerHand:
        """Runs one action against a fresh round and returns the seat it acted on."""
        round_state = blackjack_round(
            # A copy: the hand holds the list itself, so a `hit` would grow the caller's.
            hands=[list(player_cards)],
            dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
        )
        view = blackjack_view(round_state=round_state)
        assert view._apply_bot_action(user_id=1, action=action, allowed=(action,)) is True
        return round_state.players[0]

    stiff = [card(rank="10"), card(rank="7", suit="♥")]
    pair = [card(rank="8"), card(rank="8", suit="♥")]

    assert len(_apply(action="hit", player_cards=stiff).hands[0].cards) == 3
    assert _apply(action="stand", player_cards=stiff).hands[0].finished is True
    doubled = _apply(action="double", player_cards=stiff).hands[0]
    assert (doubled.doubled, doubled.bet) == (True, 200)
    assert len(_apply(action="split", player_cards=pair).hands) == 2
    assert _apply(action="surrender", player_cards=stiff).hands[0].surrendered is True


async def test_apply_bot_action_reports_a_refused_round_call_as_unapplied() -> None:
    """A `ValueError` out of `BlackjackRound` is caught, so the caller can fall back to a stand."""
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="7", suit="♥")]],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = blackjack_view(round_state=round_state)

    # Offered by the caller but refused by the round itself: 10/7 is not a pair.
    assert view._apply_bot_action(user_id=1, action="split", allowed=("split",)) is False
    assert len(round_state.players[0].hands) == 1


async def test_apply_bot_action_rejects_action_not_in_allowed() -> None:
    """An action outside `allowed` is refused even where the round itself would take it.

    A pair of eights can split, so only the `allowed` check stands between it and a second hand.
    """
    round_state = blackjack_round(
        hands=[[card(rank="8"), card(rank="8", suit="♥")]],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = blackjack_view(round_state=round_state)

    applied = view._apply_bot_action(user_id=1, action="split", allowed=("hit", "stand"))
    assert applied is False
    assert len(round_state.players[0].hands) == 1


def _alice_done_bob_to_act() -> BlackjackRound:
    """Builds a two-seat round where Alice has finished and Bob holds the turn."""
    round_state = blackjack_round(
        hands=[
            [card(rank="10"), card(rank="7", suit="♥")],
            [card(rank="5", suit="♣"), card(rank="6", suit="♦")],
        ],
        dealer=[card(rank="9", suit="♣"), card(rank="7", suit="♦")],
        seats=[
            seat(bet=50, balance_at_start=100),
            seat(user_id=2, display_name="Bob", bet=50, balance_at_start=100),
        ],
    )
    round_state.players[0].hands[0].finished = True
    round_state.current_player_index = 1
    return round_state


async def test_blackjack_view_rejects_stale_double_without_mutating_next_player() -> None:
    """A stale Double interaction cannot double the next active player's hand."""
    round_state = _alice_done_bob_to_act()
    bob = round_state.players[1].hands[0]

    message = FakeDiscordMessage()
    view = blackjack_view(round_state=round_state)

    interaction = FakeInteraction(user=FakeUser(user_id=1), message=message)
    await attached_button(view=view, custom_id="bj:double").callback(
        as_interaction(fake=interaction)
    )

    assert bob.bet == 50
    assert [str(card) for card in bob.cards] == ["5♣", "6♦"]
    assert interaction.followup.sent[0]["content"] == "這個操作已經失效，請看最新牌桌"
    assert interaction.followup.sent[0]["ephemeral"] is True
    assert len(message.edits) == 1


async def test_blackjack_view_rejects_stale_hit_without_drawing_for_next_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale Hit interaction cannot draw a card for the next active player."""

    def fail_draw(rng: Random) -> Card:
        """Fails the test if stale Hit reaches card draw."""
        raise AssertionError("stale hit should not draw")

    monkeypatch.setattr("discordbot.cogs.games.blackjack.draw_card", fail_draw)
    round_state = _alice_done_bob_to_act()
    round_state.shoe = []
    bob = round_state.players[1].hands[0]

    message = FakeDiscordMessage()
    view = blackjack_view(round_state=round_state)

    interaction = FakeInteraction(user=FakeUser(user_id=1), message=message)
    await attached_button(view=view, custom_id="bj:hit").callback(as_interaction(fake=interaction))

    assert [str(card) for card in bob.cards] == ["5♣", "6♦"]
    assert interaction.followup.sent[0]["content"] == "這個操作已經失效，請看最新牌桌"
    assert len(message.edits) == 1


async def test_blackjack_view_hit_draws_for_active_split_hand() -> None:
    """A Hit draws for the active split hand, not the already-finished first hand."""
    round_state = blackjack_round(
        hands=[[]],
        dealer=[card(rank="9", suit="♥"), card(rank="7", suit="♦")],
        seats=[seat(bet=50, balance_at_start=100)],
    )
    player = round_state.players[0]
    player.hands = [
        blackjack_hand(
            cards=[card(rank="10"), card(rank="2", suit="♥")],
            bet=50,
            is_split_hand=True,
            finished=True,
        ),
        blackjack_hand(
            cards=[card(rank="9", suit="♣"), card(rank="2", suit="♦")], bet=50, is_split_hand=True
        ),
    ]
    round_state.current_hand_index = 1
    round_state.shoe = [card(rank="5", suit="♣")]

    message = FakeDiscordMessage()
    view = blackjack_view(round_state=round_state)

    await attached_button(view=view, custom_id="bj:hit").callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=1), message=message))
    )

    assert [str(card) for card in player.hands[1].cards] == ["9♣", "2♦", "5♣"]
    assert len(message.edits) == 1

    await view.wait_for_background_tasks()

    assert len(message.edits) == 1


# Starting the table --------------------------------------------------------


@pytest.mark.parametrize(
    argnames="failure",
    argvalues=[
        make_forbidden(message="Missing Access"),
        make_not_found(message="Unknown Message"),
        make_server_error(),
    ],
    ids=["refused", "message_gone", "discord_failing"],
)
async def test_a_blackjack_start_whose_table_never_lands_keeps_the_channel_shoe(
    monkeypatch: pytest.MonkeyPatch, failure: HTTPException
) -> None:
    """No card of the failed deal was shown, so the next start deals from the shoe the bot counted."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    lobby = scripted_blackjack_lobby(dealt=[])
    store = lobby._shoe_store
    assert store is not None
    before = list(store.shoes[7])
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    owner_interaction = FakeInteraction(user=FakeUser(user_id=1), message=message)
    owner_interaction.edit_failure = failure

    # Only a refusal is answered in place; any other failure still reaches the view's on_error.
    with contextlib.suppress(HTTPException):
        await lobby_button(view=lobby, label="開始").callback(
            as_interaction(fake=owner_interaction)
        )

    assert store.shoes.get(7) == before


@pytest.mark.parametrize(
    argnames="failing_step", argvalues=["build_in_progress_embeds", "table_edit_kwargs"]
)
async def test_a_blackjack_start_that_raises_before_its_table_is_up_reopens_and_keeps_the_shoe(
    monkeypatch: pytest.MonkeyPatch, failing_step: str
) -> None:
    """A start whose table is never sent reopens the lobby, whichever step raised.

    Left marked started, the lobby would refuse every press and skip its own timeout cleanup.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    def failing(**_kwargs: object) -> NoReturn:
        raise ValueError(failing_step)

    monkeypatch.setattr(target=blackjack_views, name=failing_step, value=failing)
    lobby = scripted_blackjack_lobby(dealt=[])
    store = lobby._shoe_store
    assert store is not None
    before = list(store.shoes[7])
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)

    # Called directly, the press skips the view's on_error, so the raise reaches the test.
    with pytest.raises(ValueError, match=failing_step):
        await lobby_button(view=lobby, label="開始").callback(
            as_interaction(fake=FakeInteraction(user=FakeUser(user_id=1), message=message))
        )

    assert store.shoes.get(7) == before
    assert not lobby.is_finished()
    await lobby.on_timeout()
    assert scheduled.messages == [message]


# A channel the bot was shut out of ----------------------------------------


async def _start_in_a_shut_out_channel(
    monkeypatch: pytest.MonkeyPatch, dealt: list[Card], bot: GameParticipant | None = None
) -> FakeDiscordMessage:
    """Presses 開始 on Alice's scripted lobby in a channel that refuses every edit.

    The lobby went up on the slash command's token, so it shows in a channel the server shut the
    bot out of afterwards. Returns the lobby message, whose `edits` hold only what landed.
    """
    record_scheduled_deletes(monkeypatch=monkeypatch)
    monkeypatch.setattr(blackjack_views, "PEEK_REVEAL_DELAY_SECONDS", 0)
    monkeypatch.setattr(blackjack_views, "BOT_TURN_EDIT_DELAY_SECONDS", 0)
    await seed_balance(user_id=1, name="alice", amount=100)
    if bot is not None:
        await seed_balance(user_id=bot.user_id, name=bot.account_name, amount=100)
    lobby = scripted_blackjack_lobby(dealt=dealt, bot=bot)
    message = FakeDiscordMessage()
    message.edit_failure = make_forbidden(message="Missing Access")
    lobby.message = as_message(fake=message)
    owner_start = FakeInteraction(user=FakeUser(user_id=1), message=message)
    start_button = lobby_button(view=lobby, label="開始")

    await start_button.callback(as_interaction(fake=owner_start))

    assert owner_start.followup.sent == []
    return message


def _alice_press(message: FakeDiscordMessage) -> FakeInteraction:
    """Builds one press by Alice on the table message."""
    return FakeInteraction(user=FakeUser(user_id=1), message=message)


async def test_a_blackjack_lobby_in_a_channel_the_bot_was_shut_out_of_still_deals_and_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The start and each table press edit through their own token, which no channel can refuse.

    The settling press's token also carries the table's scheduled delete.
    """
    # Nothing but fives: no natural and no insurance, so the deal leaves a table to show.
    message = await _start_in_a_shut_out_channel(monkeypatch=monkeypatch, dealt=[])
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)
    stale_double = attached_button(view=table, custom_id="bj:double")
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    await attached_button(view=table, custom_id="bj:hit").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    stale_press = _alice_press(message=message)
    await stale_double.callback(as_interaction(fake=stale_press))

    assert len(stale_press.followup.sent) == 1
    assert len(message.edits) == 3, "the stale press refreshed the table it was pressed on"

    stand = _alice_press(message=message)
    await attached_button(view=table, custom_id="bj:stand").callback(as_interaction(fake=stand))
    await table.wait_for_background_tasks()

    assert {"view": table} in message.edits, "the controls went dead before the dealer played"
    assert message.edits[-1]["view"] is None, "the settled table replaced the live one"
    assert scheduled.interactions == [stand]


@pytest.mark.parametrize(argnames="expired", argvalues=[False, True], ids=["live", "expired"])
@pytest.mark.parametrize(argnames="last", argvalues=["start", "hit"])
async def test_a_blackjack_table_left_to_time_out_in_a_shut_out_channel_closes_through_its_last_press(
    monkeypatch: pytest.MonkeyPatch, last: str, expired: bool
) -> None:
    """A timeout has no press of its own, so it renders and deletes through the last one.

    That is the start itself when nobody pressed after it. Once that press's token has expired
    only the channel is left, whose refusal is expected and so carries no traceback.
    """
    # Nothing but fives: no natural and no insurance, so the deal leaves a table to show.
    message = await _start_in_a_shut_out_channel(monkeypatch=monkeypatch, dealt=[])
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)
    if last == "start":
        press = cast("FakeInteraction", table.last_press)
        assert press.edits[0]["view"] is table, "the start press dealt the table"
    else:
        press = _alice_press(message=message)
        await attached_button(view=table, custom_id="bj:hit").callback(as_interaction(fake=press))
    press.expired = expired
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    reports = capture_levels(monkeypatch=monkeypatch, levels=("info", "warn"))

    await table.on_timeout()
    await table.wait_for_background_tasks()

    assert (press.edits[-1]["view"] is None) is not expired, "the settled table landed via it"
    assert (scheduled.messages, scheduled.interactions) == ([message], [press]), (
        "the delete rides the same press"
    )
    assert [(level, "_exc_info" in fields) for level, _, fields in reports] == (
        [("warn", False), ("warn", False)] if expired else []
    )


async def test_a_blackjack_table_kept_open_by_a_refused_press_closes_through_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A press refused out of turn restarts the table's timer, so the timeout closes through it.

    By then the press that last moved the table can be past its life, and in a channel the bot
    was shut out of nothing else reaches the table to show its result or delete it.
    """
    # Nothing but fives: no natural and no insurance, so the deal leaves a table to show.
    message = await _start_in_a_shut_out_channel(monkeypatch=monkeypatch, dealt=[])
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)
    cast("FakeInteraction", table.last_press).expired = True
    refused = FakeInteraction(user=FakeUser(user_id=2, name="bob"), message=message)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    assert await table.interaction_check(interaction=as_interaction(fake=refused)) is False
    await table.on_timeout()
    await table.wait_for_background_tasks()

    assert [edit["view"] for edit in refused.edits[-1:]] == [None], "the result landed through it"
    assert (scheduled.messages, scheduled.interactions) == ([message], [refused])
    assert refused.followup.sent == [{"content": "現在輪到 Alice", "ephemeral": True}]


async def test_a_blackjack_natural_at_the_deal_settles_inside_the_start_press(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dealer natural under a ten ends the round at the deal, so the start press shows it all."""
    # Alice 5 5, the dealer's hole an ace under a king.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch, dealt=[card(rank="5")] * 2 + [card(rank="A"), card(rank="K")]
    )
    table = message.edits[0]["view"]
    assert isinstance(table, BlackjackView)
    await table.wait_for_background_tasks()

    # The controls going dead, the peek's two frames, and the settled table.
    assert len(message.edits) == 4
    hidden, revealed = (cast("str", edit["embeds"][0].description) for edit in message.edits[1:3])
    assert "🂠" in hidden, "the peek shows the hole card face down first"
    assert "🂠" not in revealed
    assert message.edits[-1]["view"] is None


async def test_a_blackjack_round_in_a_server_the_bot_is_not_in_is_recorded_under_that_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user install plays in servers the bot never joined, where nothing resolves a guild.

    The press still carries the server's id, so the round's history is not filed as a DM's.
    """
    record_scheduled_deletes(monkeypatch=monkeypatch)
    monkeypatch.setattr(blackjack_views, "PEEK_REVEAL_DELAY_SECONDS", 0)
    await seed_balance(user_id=1, name="alice", amount=100)
    # Alice 5 5, the dealer's hole an ace under a king: the start press settles the round.
    lobby = scripted_blackjack_lobby(dealt=[card(rank="5")] * 2 + [card(rank="A"), card(rank="K")])
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    start = FakeInteraction(user=FakeUser(user_id=1), message=message, guild_id=555)
    start.guild = None

    await lobby_button(view=lobby, label="開始").callback(as_interaction(fake=start))
    table = message.edits[0]["view"]
    assert isinstance(table, BlackjackView)
    await table.wait_for_background_tasks()

    rounds = await fetch_recent_blackjack_rounds(user_id=1, limit=5)
    assert [record.guild_id for record in rounds] == [555]


async def test_a_blackjack_natural_under_an_ace_settles_inside_the_insurance_press(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last insurance call closes the phase on a dealer natural, so its press shows the end."""
    # Alice 5 5, the dealer's hole a king under an ace.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch, dealt=[card(rank="5")] * 2 + [card(rank="K"), card(rank="A")]
    )
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)

    await attached_button(view=table, custom_id="bj:insure_no").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    await table.wait_for_background_tasks()

    # The table, the controls going dead, the peek's two frames, and the settled table.
    assert len(message.edits) == 5
    assert message.edits[-1]["view"] is None


async def test_an_ace_up_peek_without_a_natural_keeps_the_hole_face_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dealer peeks under an ace for a natural only; finding none, the hole stays secret.

    The bot reads the hole as its private edge, so a peek showing it to the table would hand
    every player the same card.
    """
    # Alice 5 5, the dealer's hole a 6 under an ace: insurance, then no natural.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch, dealt=[card(rank="5")] * 2 + [card(rank="6"), card(rank="A")]
    )
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)

    await attached_button(view=table, custom_id="bj:insure_no").callback(
        as_interaction(fake=_alice_press(message=message))
    )

    dealer_seats = [cast("str", edit["embeds"][0].description) for edit in message.edits]
    assert all("🂠" in dealer_seat for dealer_seat in dealer_seats), "the hole stays face down"
    assert len(dealer_seats) == 2, "the table, then the table after the insurance call"
    assert table.round_state.phase == "player_actions"


async def test_the_bot_plays_its_seat_through_the_press_that_handed_it_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bot sits at every table its wallet can fund, and it has no press of its own.

    Every move it makes runs inside a human press still holding the round lock, so it edits the
    table through that press; through the channel, a shut-out table stalls on its first move.
    """
    bot = seat(user_id=999, display_name="Dealer", bet=10, balance_at_start=100)
    # Alice 5 5, the bot 5 5, the dealer's hole a 6 under an ace: insurance, no natural, and a
    # dealer bound for 17 on a shoe of fives, so the bot draws to 20 before it stands.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch,
        dealt=[card(rank="5")] * 4 + [card(rank="6"), card(rank="A")],
        bot=bot,
    )
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)
    assert len(message.edits) == 2, "the table, then the bot's insurance call on the start press"

    await attached_button(view=table, custom_id="bj:insure_no").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    assert len(message.edits) == 3, "no natural under the ace, so only the table follows the call"

    await attached_button(view=table, custom_id="bj:stand").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    await table.wait_for_background_tasks()

    bot_hand = table.round_state.players[1].hands[0]
    assert len(bot_hand.cards) == 4, "the bot drew twice on the press that handed it the turn"
    assert message.edits[-1]["view"] is None, "the bot played its hand out and the table settled"
