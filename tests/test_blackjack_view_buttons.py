"""Tests for the Blackjack table view: its controls, dealer play, bot turns, settling, rendering.

Controls are presence-based, so the button tests assert which custom_ids are
attached rather than which are disabled. Dealer play is deterministic (H17) and
the bot's decisions come from the EV engine, so both are asserted exactly.
"""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random
from typing import Any, Literal, cast
import asyncio

import pytest
from nextcord.ui import Button

from discordbot.cogs.games import blackjack_views
from discordbot.typings.games import (
    BotAction,
    GameParticipant,
    BlackjackPlayerResult,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
)
from discordbot.cogs.games.shoe import BlackjackShoeStore
from discordbot.cogs.games.blackjack import (
    Card,
    BlackjackRound,
    BlackjackHandState,
    BlackjackPlayerHand,
)
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.cogs.games.interactions import table_edit_kwargs
from discordbot.cogs.games.presentation import settlement_metadata
from discordbot.services.economy.database import get_balance, get_casino_ledger
from discordbot.cogs.games.blackjack_views import (
    BlackjackView,
    build_final_embeds,
    build_in_progress_embeds,
)

from tests.helpers.games import (
    ScheduledDeletes,
    card,
    seat,
    component_ids,
    component_rows,
    attached_button,
    settle_only_seat,
    record_scheduled_deletes,
)
from tests.helpers.casting import (
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
)
from tests.helpers.economy import seed_balance
from tests.helpers.discord_mocks import FakeUser, FakeGuild, FakeInteraction, FakeDiscordMessage


def _round_with_two_cards(
    player_cards: list[Card],
    dealer_cards: list[Card],
    player: GameParticipant | None = None,
    finished: bool = False,
) -> BlackjackRound:
    """Builds a one-seat round from fixed cards, still in play unless `finished`.

    A finished round is one whose dealer has played and whose only hand is done, which is what
    a view finalizes or settles.
    """
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[player or seat()]
    )
    hand = round_state.players[0].hands[0]
    hand.cards = player_cards
    round_state.dealer = dealer_cards
    if finished:
        hand.finished = True
        round_state.dealer_played = True
        round_state.phase = "settled"
    return round_state


def _make_view(round_state: BlackjackRound) -> BlackjackView:
    """Builds a BlackjackView for button inspection."""
    return BlackjackView(round_state=round_state, owner=seat())


@pytest.fixture
def scheduled_cleanups(monkeypatch: pytest.MonkeyPatch) -> ScheduledDeletes:
    """Records each table the view hands to the public-message cleanup instead of scheduling it."""
    return record_scheduled_deletes(monkeypatch=monkeypatch)


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
    round_state = _round_with_two_cards(
        player_cards=[card(rank="8"), card(rank="8", suit="♥")],
        dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
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
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="5", suit="♥")],
        dealer_cards=[card(rank="A", suit="♣"), card(rank="9", suit="♦")],
    )
    round_state.phase = "insurance"
    round_state.insurance_offered = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    states = _button_states(view=view)
    assert component_ids(view=view) == {"bj:insure_yes", "bj:insure_no"}
    assert states["bj:insure_yes"] is False
    assert states["bj:insure_no"] is False
    assert component_rows(view=view) == {"bj:insure_yes": 1, "bj:insure_no": 1}


async def test_settled_phase_removes_every_button() -> None:
    """After settlement no controls remain attached to the view."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="9", suit="♥")],
        dealer_cards=[card(rank="K", suit="♣"), card(rank="7", suit="♦")],
    )
    round_state.phase = "settled"
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert component_ids(view=view) == set()


async def test_sync_buttons_drops_insurance_controls_outside_insurance() -> None:
    """Insurance buttons join the view for the insurance phase alone and leave it after."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="8"), card(rank="8", suit="♥")],
        dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    ids = component_ids(view=view)
    assert "bj:insure_yes" not in ids
    assert "bj:insure_no" not in ids

    round_state.phase = "insurance"
    round_state.insurance_offered = True
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
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="7", suit="♥")],
        dealer_cards=[card(rank="A", suit="♣"), card(rank="K", suit="♦")],
    )

    embeds = build_in_progress_embeds(round_state=round_state, force_show_hole=True)
    dealer_embed = embeds[0]

    assert isinstance(dealer_embed.description, str)
    assert "A♣" in dealer_embed.description
    assert "K♦" in dealer_embed.description
    assert "🂠" not in dealer_embed.description


def test_settlement_metadata_shows_vip_bonus_numbers() -> None:
    """A VIP-boosted win shows the total delta and the VIP bonus inside it."""
    metadata = settlement_metadata(delta=150, new_balance=1_150, is_allin=False, vip_bonus=50)

    assert metadata == "-# 本局 `+150` · VIP加成 `+50` · 餘額 `1,150`"


def test_blackjack_in_progress_dealer_seat_hides_hole_card() -> None:
    """The dealer seat embed shows one hidden card marker plus the visible up-card."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(display_name="Bob")]
    )
    round_state.players[0].hands[0].cards = [card(rank="10"), card(rank="7", suit="♥")]
    round_state.dealer = [card(rank="8", suit="♣"), card(rank="K", suit="♦")]

    embeds = build_in_progress_embeds(round_state=round_state)
    dealer_embed = embeds[0]

    assert isinstance(dealer_embed.description, str)
    assert "🂠" in dealer_embed.description
    assert "K♦" in dealer_embed.description
    assert "8♣" not in dealer_embed.description


def test_blackjack_in_progress_dealer_seat_single_card_is_visible() -> None:
    """A one-card dealer fallback should not render as a hidden hole card."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(display_name="Bob")]
    )
    round_state.players[0].hands[0].cards = [card(rank="10"), card(rank="7", suit="♥")]
    round_state.dealer = [card(rank="8", suit="♣")]

    embeds = build_in_progress_embeds(round_state=round_state)
    dealer_embed = embeds[0]

    assert isinstance(dealer_embed.description, str)
    assert "8♣" in dealer_embed.description
    assert "🂠" not in dealer_embed.description


def test_blackjack_table_edit_payload_adds_width_spacer() -> None:
    """Blackjack table edits attach one transparent spacer and reference it from every embed."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="7", suit="♥")],
        dealer_cards=[card(rank="K", suit="♣"), card(rank="9", suit="♦")],
    )
    seat_embeds = build_in_progress_embeds(round_state=round_state)

    payload = table_edit_kwargs(embeds=seat_embeds, view=None)

    assert payload["attachments"] == []
    assert payload["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    for embed in payload["embeds"]:
        assert embed.image.url == embed_spacer_url()


async def test_interaction_check_sends_ephemeral_notice_when_settled() -> None:
    """After the round is settled, clicks get an ephemeral notice rather than silent ignore."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="7", suit="♥")],
        dealer_cards=[card(rank="K", suit="♣"), card(rank="9", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view._settled = True
    press = FakeInteraction()

    allowed = await view.interaction_check(interaction=as_interaction(fake=press))

    assert allowed is False
    assert press.response.sent == [{"content": "這局已經結束, 等下一局吧", "ephemeral": True}]


async def test_a_seat_that_can_never_insure_is_not_sent_to_refresh() -> None:
    """A 1-point seat's half-bet rounds to zero, and no newer table will change that.

    The insurance buttons belong to the table rather than to a seat — every undecided player
    sees the same pair — so the seat that cannot use them only finds out by pressing, and what
    it is told then is the whole of the feature for it.
    """
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="7", suit="♥")],
        dealer_cards=[card(rank="A", suit="♣"), card(rank="9", suit="♦")],
        player=seat(bet=1),
    )
    round_state.phase = "insurance"
    round_state.insurance_offered = True
    view = _make_view(round_state=round_state)
    message = FakeDiscordMessage()
    press = FakeInteraction(message=message)

    decided = await view._take_insurance_locked(
        interaction=as_interaction(fake=press), message=as_message(fake=message), user_id=1
    )

    assert decided is False
    assert round_state.players[0].insurance_bet == 0
    assert press.response.sent == [
        {"content": "你的下注太小，一半不到 1 點，這局沒有保險可買", "ephemeral": True}
    ]


def test_play_dealer_hits_below_17_then_stands_on_hard_17() -> None:
    """Dealer hits ≤16 and stands on a hard 17 under H17 rules."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="9", suit="♥")],
        dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.shoe = [card(rank="6")]

    steps = round_state.play_dealer()

    assert round_state.dealer_played is True
    first_step = steps[0]
    assert first_step.action == "hit"
    assert first_step.total_before == 11
    assert first_step.total_after == 17
    final_step = steps[-1]
    assert final_step.action == "stand"
    assert final_step.total_before == 17


@pytest.mark.parametrize(
    argnames=("dealer_cards", "expected_total"),
    argvalues=[
        ([card(rank="K", suit="♣"), card(rank="7", suit="♦")], 17),
        ([card(rank="K", suit="♣"), card(rank="8", suit="♦")], 18),
    ],
)
def test_play_dealer_stands_on_hard_17_plus(dealer_cards: list[Card], expected_total: int) -> None:
    """Dealer stands deterministically on any hard 17+ total."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="9", suit="♥")], dealer_cards=dealer_cards
    )
    round_state.players[0].hands[0].finished = True

    steps = round_state.play_dealer()

    assert round_state.dealer_played is True
    assert round_state.dealer_total() == expected_total
    step = steps[-1]
    assert step.action == "stand"
    assert step.total_before == expected_total


def test_play_dealer_hits_soft_17() -> None:
    """Dealer hits soft 17 (H17 rule) instead of standing."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="9", suit="♥")],
        dealer_cards=[card(rank="A", suit="♣"), card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.shoe = [card(rank="3")]

    steps = round_state.play_dealer()

    assert [str(card) for card in round_state.dealer] == ["A♣", "6♦", "3♠"]
    first_step = steps[0]
    assert first_step.action == "hit"
    assert first_step.total_before == 17
    final_step = steps[-1]
    assert final_step.action == "stand"
    assert final_step.total_before == 20


def test_play_dealer_records_nothing_after_a_bust() -> None:
    """A dealer that busts shows the hit that did it and no stand after it."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="9", suit="♥")],
        dealer_cards=[card(rank="10", suit="♣"), card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.shoe = [card(rank="K")]

    steps = round_state.play_dealer()

    assert round_state.dealer_played is True
    path = blackjack_views._format_dealer_decision_path(steps=steps)
    assert path == "規則: 16 hit 抽 K♠ → 26"


async def test_bot_dispatcher_skips_when_no_bot_seated() -> None:
    """The bot turn dispatcher is a no-op when no bot is seated."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="9", suit="♥")],
        dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    assert view.bot_user_id is None
    message = FakeDiscordMessage()
    press = FakeInteraction(message=message)
    await view._maybe_play_bot_turn_locked(
        message=as_message(fake=message), interaction=as_interaction(fake=press)
    )
    assert press.edits == []


async def test_bot_dispatcher_skips_when_active_player_is_human() -> None:
    """If the active seat belongs to a human, the bot dispatcher returns immediately."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="7", suit="♥")],
        dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
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
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="7", suit="♥")],
        dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
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
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[seat(user_id=1, display_name="Bot"), seat(user_id=2, display_name="Bob")],
    )
    round_state.players[0].hands[0].cards = [card(rank="10"), card(rank="7", suit="♥")]
    round_state.players[1].hands[0].cards = [card(rank="9", suit="♣"), card(rank="8", suit="♦")]
    round_state.dealer = [card(rank="A", suit="♣"), card(rank="9", suit="♦")]
    round_state.phase = "insurance"
    round_state.insurance_offered = True
    round_state.players[0].insurance_resolved = True
    view = _make_view(round_state=round_state)
    view.bot_user_id = 1
    message = FakeDiscordMessage()

    await view.maybe_play_bot_turn(
        message=as_message(fake=message),
        interaction=as_interaction(fake=FakeInteraction(message=message)),
    )

    assert message.edits == []
    assert round_state.phase == "insurance"
    assert round_state.players[1].insurance_resolved is False


async def test_bot_dispatcher_paces_consecutive_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consecutive bot-owned decisions wait briefly between message edits."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="7", suit="♥")],
        dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.bot_user_id = 1
    dispatch_calls = 0
    sleep_calls: list[float] = []

    async def fake_dispatch(**_kwargs: object) -> None:
        nonlocal dispatch_calls
        dispatch_calls += 1
        view._state_revision += 1
        if dispatch_calls == 2:
            view.round_state.stand(user_id=1)

    async def fake_sleep(*, delay: float) -> None:
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
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[seat(user_id=1, display_name="Bot"), seat(user_id=2, display_name="Bob")],
    )
    bot_hand = round_state.players[0].hands[0]
    bot_hand.cards = [card(rank="10"), card(rank="6", suit="♥")]
    round_state.players[1].hands[0].cards = [card(rank="9", suit="♣"), card(rank="8", suit="♦")]
    round_state.dealer = [card(rank="6", suit="♣"), card(rank="10", suit="♦")]
    round_state.shoe = [card(rank="10")] * 20
    view = _make_view(round_state=round_state)
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

    def _apply(*, action: BotAction, player_cards: list[Card]) -> BlackjackPlayerHand:
        """Runs one action against a fresh round and returns the seat it acted on."""
        round_state = _round_with_two_cards(
            # A copy: the hand holds the list itself, so a `hit` would grow the caller's.
            player_cards=list(player_cards),
            dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
        )
        view = _make_view(round_state=round_state)
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
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="7", suit="♥")],
        dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)

    # Offered by the caller but refused by the round itself: 10/7 is not a pair.
    assert view._apply_bot_action(user_id=1, action="split", allowed=("split",)) is False
    assert len(round_state.players[0].hands) == 1


async def test_apply_bot_action_rejects_action_not_in_allowed() -> None:
    """Actions not in `allowed` are rejected without raising."""
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="7", suit="♥")],
        dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)

    applied = view._apply_bot_action(user_id=1, action="split", allowed=("hit", "stand"))
    assert applied is False


async def test_finalize_persists_remaining_shoe_to_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settling a round writes the round's remaining shoe back into the channel store."""
    store = BlackjackShoeStore()
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="9", suit="♥")],
        dealer_cards=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    round_state.shoe = [
        card(rank="7"),
        card(rank="8", suit="♥"),
        card(rank="2", suit="♦"),
        card(rank="3", suit="♣"),
    ]
    view = BlackjackView(round_state=round_state, owner=seat(), shoe_store=store, channel_id=42)

    async def _stop_after_save(**_kwargs: object) -> None:
        raise RuntimeError("stop after shoe save")

    # Settlement runs after the shoe save, so raising there proves the save already ran.
    monkeypatch.setattr(blackjack_views, "settle_blackjack_player", _stop_after_save)

    with pytest.raises(RuntimeError, match="stop after shoe save"):
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
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="9", suit="♥")],
        dealer_cards=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
        player=seat(bet=50, balance_at_start=100),
        finished=True,
    )
    view = _make_view(round_state=round_state)
    recorded: dict[str, object] = {}

    async def record_blackjack_history(**kwargs: object) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(blackjack_views, "record_blackjack_history", record_blackjack_history)

    await view.finalize(
        message=as_message(fake=FakeDiscordMessage(guild=FakeGuild(guild_id=888))),
        interaction=None,
    )
    round_state.dealer.append(card(rank="K", suit="♣"))
    await view.wait_for_background_tasks()

    assert recorded["dealer_cards"] == [card(rank="10", suit="♣"), card(rank="8", suit="♦")]
    assert recorded["dealer_total"] == 18
    assert recorded["guild_id"] == 888


# Settling a table ---------------------------------------------------------


async def test_blackjack_view_finalizes_once_when_called_concurrently(
    scheduled_cleanups: ScheduledDeletes,
) -> None:
    """Concurrent finalization attempts must not pay out one Blackjack hand twice."""
    await seed_balance(user_id=1, name="alice", amount=100)

    message = FakeDiscordMessage()
    view = _make_view(
        round_state=_round_with_two_cards(
            player_cards=[card(rank="10"), card(rank="Q", suit="♥")],
            dealer_cards=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
            player=seat(bet=50, balance_at_start=100),
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
    view = _make_view(
        round_state=_round_with_two_cards(
            player_cards=[card(rank="10"), card(rank="8", suit="♥")],
            dealer_cards=[card(rank="10", suit="♣"), card(rank="Q", suit="♦")],
            player=seat(bet=50, balance_at_start=100),
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
    reports: list[tuple[str, dict[str, object]]] = []
    for name in ("info", "warn"):
        monkeypatch.setattr(
            target=blackjack_views.logfire,
            name=name,
            value=lambda _message, name=name, **fields: reports.append((name, fields)),
        )
    message = FakeDiscordMessage()
    message.edit_failure = failure
    view = _make_view(
        round_state=_round_with_two_cards(
            player_cards=[card(rank="10"), card(rank="8", suit="♥")],
            dealer_cards=[card(rank="A", suit="♣"), card(rank="K", suit="♦")],
        )
    )

    await view._safe_edit_view_locked(message=as_message(fake=message), interaction=None)
    await view._animate_peek_locked(message=as_message(fake=message), interaction=None)

    # order-contract: the three edits are awaited one after another, and each reports the same.
    assert [(name, "_exc_info" in fields) for name, fields in reports] == [(level, traceback)] * 3


async def test_blackjack_view_dealer_plays_h17_rule(scheduled_cleanups: ScheduledDeletes) -> None:
    """Dealer plays deterministically under H17 (hits below 17, stands on hard 17+)."""
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="7", suit="♥")],
        dealer_cards=[card(rank="10", suit="♣"), card(rank="3", suit="♦")],
        player=seat(bet=50, balance_at_start=100),
    )
    round_state.shoe = [card(rank="5", suit="♣")]

    message = FakeDiscordMessage()
    view = _make_view(round_state=round_state)

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
    round_state = _round_with_two_cards(
        player_cards=[
            card(rank="2"),
            card(rank="3", suit="♥"),
            card(rank="4", suit="♣"),
            card(rank="5", suit="♦"),
            card(rank="7"),
        ],
        dealer_cards=[card(rank="10", suit="♣"), card(rank="6", suit="♦")],
        player=seat(bet=100, balance_at_start=1_000),
    )
    round_state.shoe = [card(rank="5", suit="♣")]
    view = _make_view(round_state=round_state)

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
    view = _make_view(
        round_state=_round_with_two_cards(
            player_cards=[card(rank="10"), card(rank="Q", suit="♥")],
            dealer_cards=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
            player=seat(bet=50, balance_at_start=50),
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
    round_state = _round_with_two_cards(
        player_cards=[card(rank="10"), card(rank="2", suit="♥")],
        dealer_cards=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
        player=seat(bet=50, balance_at_start=100),
    )
    round_state.shoe = [card(rank="5", suit="♣")]
    message = FakeDiscordMessage()
    view = _make_view(round_state=round_state)
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


def _alice_done_bob_to_act() -> BlackjackRound:
    """Builds a two-seat round where Alice has finished and Bob holds the turn."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[
            seat(user_id=1, display_name="Alice", bet=50, balance_at_start=100),
            seat(user_id=2, display_name="Bob", bet=50, balance_at_start=100),
        ],
    )
    alice = round_state.players[0].hands[0]
    alice.cards = [card(rank="10"), card(rank="7", suit="♥")]
    alice.finished = True
    round_state.players[1].hands[0].cards = [card(rank="5", suit="♣"), card(rank="6", suit="♦")]
    round_state.dealer = [card(rank="9", suit="♣"), card(rank="7", suit="♦")]
    round_state.current_player_index = 1
    return round_state


async def test_blackjack_view_rejects_stale_double_without_mutating_next_player() -> None:
    """A stale Double interaction cannot double the next active player's hand."""
    round_state = _alice_done_bob_to_act()
    bob = round_state.players[1].hands[0]

    message = FakeDiscordMessage()
    view = _make_view(round_state=round_state)

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
    view = _make_view(round_state=round_state)

    interaction = FakeInteraction(user=FakeUser(user_id=1), message=message)
    await attached_button(view=view, custom_id="bj:hit").callback(as_interaction(fake=interaction))

    assert [str(card) for card in bob.cards] == ["5♣", "6♦"]
    assert interaction.followup.sent[0]["content"] == "這個操作已經失效，請看最新牌桌"
    assert len(message.edits) == 1


async def test_blackjack_view_hit_draws_for_active_split_hand() -> None:
    """A Hit draws for the active split hand, not the already-finished first hand."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(bet=50, balance_at_start=100)]
    )
    player = round_state.players[0]
    player.hands = [
        BlackjackHandState(
            cards=[card(rank="10"), card(rank="2", suit="♥")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
            finished=True,
        ),
        BlackjackHandState(
            cards=[card(rank="9", suit="♣"), card(rank="2", suit="♦")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
        ),
    ]
    round_state.dealer = [card(rank="9", suit="♥"), card(rank="7", suit="♦")]
    round_state.current_hand_index = 1
    round_state.shoe = [card(rank="5", suit="♣")]

    message = FakeDiscordMessage()
    view = _make_view(round_state=round_state)

    await attached_button(view=view, custom_id="bj:hit").callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=1), message=message))
    )

    assert [str(card) for card in player.hands[1].cards] == ["9♣", "2♦", "5♣"]
    assert len(message.edits) == 1

    await view.wait_for_background_tasks()

    assert len(message.edits) == 1


# Final table embeds --------------------------------------------------------


def _five_card_round(last_card: str, dealer_cards: list[Card]) -> BlackjackRound:
    """Builds a settled one-seat round whose hand is 2-3-4-5 plus `last_card`."""
    round_state = _round_with_two_cards(
        player_cards=[
            card(rank="2"),
            card(rank="3", suit="♥"),
            card(rank="4", suit="♣"),
            card(rank="5", suit="♦"),
            card(rank=last_card),
        ],
        dealer_cards=dealer_cards,
        player=seat(bet=10_000, balance_at_start=100_000),
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
    round_state = _round_with_two_cards(
        player_cards=[card(rank="9"), card(rank="8", suit="♥")],
        dealer_cards=[card(rank="K", suit="♣"), card(rank="A", suit="♦")],
        player=seat(bet=100, balance_at_start=100),
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
