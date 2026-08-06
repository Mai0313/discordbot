"""Pins `BlackjackView`'s control surface, its H17 dealer phase, and its deterministic bot turn.

`cogs/games/blackjack_views.py` keeps a dealt table inside one view for the whole round, so what
a player may press, what the dealer draws, and when the round hands its state back are decided
there rather than in the pure rules next door. Three things are held still here:

- Which controls are attached. `sync_buttons` is presence-based: it removes what is not
  actionable instead of disabling it, so every case below asserts the set of attached custom_ids
  and the row each sits on, never a `disabled` flag. Each case is one `blackjack.py` predicate
  read from the view's side, and the two halves can drift apart: a control the round would refuse
  turns a click into a stale-action notice, and a missing one silently drops a legal play.
- The dealer phase. The view forces `auto_play_dealer` off and runs H17 itself, recording every
  draw as a `BlackjackDealerStep` for the settled embed, so the tests read that recorded path and
  not only the dealer's final total.
- The bot turn loop. The seated bot is deterministic and never an LLM, so what is worth pinning
  are the loop's guards: it leaves a human seat alone, stops once a dispatch fails to move
  `_state_revision` instead of re-deciding forever, and pauses between consecutive bot moves so
  the table does not jump in one edit.

Rounds are hand-built rather than dealt, since `from_participants` seats a table without dealing:
a test assigns the exact player and dealer cards it needs, and only a test about drawing touches
the shoe. The rest of the file covers what settlement leans on: the shared spacer payload every
table edit goes through, the ephemeral notice a click earns once the round is settled, the
remaining shoe handed back to the channel store before any money moves, and the dealer snapshot
the off-critical-path history task is scheduled with.
"""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from types import SimpleNamespace
from random import Random
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nextcord import Embed, Interaction
from nextcord.ui import Button

from discordbot.cogs.games import blackjack_views
from discordbot.typings.games import (
    GameParticipant,
    BlackjackPlayerResult,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
)
from discordbot.cogs.games.shoe import BlackjackShoeStore
from discordbot.cogs.games.blackjack import Card, BlackjackRound, BlackjackHandState
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.cogs.games.blackjack_views import BlackjackView, build_in_progress_embeds

from tests.helpers.casting import as_message


def _participant(user_id: int, display_name: str, bet: int = 100) -> GameParticipant:
    """Returns one seated participant for a hand-built round.

    The 1,000-point balance is what leaves room for the Double / Split extra wager, so those
    controls stay affordable unless a test seats a poorer player itself.
    """
    return GameParticipant(
        user_id=user_id,
        account_name=display_name.lower(),
        display_name=display_name,
        bet=bet,
        balance_at_start=1_000,
        is_allin=False,
    )


def _round_with_two_cards(
    player_cards: list[Card], dealer_cards: list[Card], bet: int = 100
) -> BlackjackRound:
    """Returns a one-seat round holding exactly the given player and dealer cards.

    `from_participants` seats the table without dealing, so the cards are assigned rather than
    drawn and the round keeps its full freshly built shoe: a test that wants a monkeypatched
    `draw_card` empties `shoe` first. The round opens in `player_actions`, and the dealer's
    up-card is the second card, the first being the hole.
    """
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[_participant(user_id=1, display_name="Alice", bet=bet)],
        auto_play_dealer=False,
    )
    round_state.players[0].hands[0].cards = player_cards
    round_state.dealer = dealer_cards
    return round_state


def _make_view(round_state: BlackjackRound) -> BlackjackView:
    """Returns a table view over an already-built round.

    Wires neither a shoe store nor a seated bot, so a test that needs one sets `bot_user_id` or
    builds its own view. Construction already runs `sync_buttons` once and forces
    `auto_play_dealer` off, which is what leaves the dealer phase for the view to run.
    """
    return BlackjackView(
        round_state=round_state,
        starter_id=1,
        author_name="alice",
        system_name="賭場系統",
        system_avatar_url="",
    )


def _button_states(view: BlackjackView) -> dict[str, bool]:
    """Returns `{custom_id: disabled}` for every button attached to the view.

    Presence is the real control, so the flags are here to catch a button re-added with a
    `disabled` left over from an earlier phase.
    """
    states: dict[str, bool] = {}
    for child in view.children:
        cid = getattr(child, "custom_id", None)
        if cid is not None and isinstance(child, Button):
            states[cid] = bool(child.disabled)
    return states


def _button_ids(view: BlackjackView) -> set[str]:
    """Returns every custom_id currently attached to the view."""
    ids: set[str] = set()
    for child in view.children:
        cid = getattr(child, "custom_id", None)
        if cid is not None:
            ids.add(cid)
    return ids


def _button_rows(view: BlackjackView) -> dict[str, int | None]:
    """Returns `{custom_id: row}` for every button in the view."""
    rows: dict[str, int | None] = {}
    for child in view.children:
        cid = getattr(child, "custom_id", None)
        if cid is not None:
            rows[cid] = getattr(child, "row", None)
    return rows


async def test_player_actions_same_rank_pair_enables_every_action_button() -> None:
    """A fresh [8, 8] against a dealer 6 attaches all five action buttons, on their two rows."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="8", suit="♠"), Card(rank="8", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert _button_ids(view=view) == {
        "bj:hit",
        "bj:stand",
        "bj:double",
        "bj:split",
        "bj:surrender",
    }
    assert all(disabled is False for disabled in _button_states(view=view).values())
    assert _button_rows(view=view) == {
        "bj:hit": 0,
        "bj:stand": 0,
        "bj:double": 1,
        "bj:split": 1,
        "bj:surrender": 1,
    }


async def test_player_actions_ten_value_pair_shows_split() -> None:
    """10 + K can be split because both cards have Blackjack value 10."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="K", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert "bj:split" in _button_ids(view=view)


async def test_player_actions_ace_ten_hides_split() -> None:
    """A + 10 is not a same-value pair, so Split is the only action button missing."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="A", suit="♠"), Card(rank="10", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    ids = _button_ids(view=view)
    assert "bj:hit" in ids
    assert "bj:stand" in ids
    assert "bj:double" in ids
    assert "bj:split" not in ids
    assert "bj:surrender" in ids


async def test_player_actions_after_hit_disables_double_split_surrender() -> None:
    """A hand that has already acted keeps only Hit and Stand."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="5", suit="♠"), Card(rank="6", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].cards.append(Card(rank="4", suit="♣"))
    round_state.players[0].hands[0].actions_taken = 1
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert _button_ids(view=view) == {"bj:hit", "bj:stand"}
    assert all(disabled is False for disabled in _button_states(view=view).values())


async def test_player_actions_is_split_hand_disables_double_split_surrender() -> None:
    """A hand born out of Split cannot be doubled (no DAS), re-split, or surrendered."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="8", suit="♠"), Card(rank="3", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].is_split_hand = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert _button_ids(view=view) == {"bj:hit", "bj:stand"}
    assert all(disabled is False for disabled in _button_states(view=view).values())


async def test_split_aces_subhand_disables_hit_and_stand() -> None:
    """A split-Aces sub-hand may take no action at all, so the view attaches no control."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[_participant(user_id=1, display_name="Alice")],
        auto_play_dealer=False,
    )
    finished_hand = BlackjackHandState(
        cards=[Card(rank="A", suit="♠"), Card(rank="5", suit="♥")],
        bet=100,
        base_bet=100,
        is_split_hand=True,
        is_split_aces=True,
        finished=False,
    )
    round_state.players[0].hands = [finished_hand]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert _button_ids(view=view) == set()


async def test_player_actions_low_balance_disables_double_and_split() -> None:
    """A balance too thin for the second wager drops Double and Split but keeps the rest."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[
            GameParticipant(
                user_id=1,
                account_name="alice",
                display_name="Alice",
                bet=100,
                balance_at_start=150,
                is_allin=False,
            )
        ],
        auto_play_dealer=False,
    )
    round_state.players[0].hands[0].cards = [Card(rank="8", suit="♠"), Card(rank="8", suit="♥")]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    ids = _button_ids(view=view)
    assert "bj:hit" in ids
    assert "bj:stand" in ids
    assert "bj:double" not in ids
    assert "bj:split" not in ids
    assert "bj:surrender" in ids


async def test_player_actions_peeked_blackjack_disables_surrender() -> None:
    """A revealed dealer Blackjack closes the Surrender window."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="9", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="K", suit="♦")],
    )
    round_state.peeked_blackjack = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert "bj:surrender" not in _button_ids(view=view)


async def test_insurance_phase_hides_action_buttons_and_shows_insurance() -> None:
    """The insurance phase swaps every action button for the two insurance ones."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="5", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="9", suit="♦")],
    )
    round_state.phase = "insurance"
    round_state.insurance_offered = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    states = _button_states(view=view)
    assert _button_ids(view=view) == {"bj:insure_yes", "bj:insure_no"}
    assert states["bj:insure_yes"] is False
    assert states["bj:insure_no"] is False
    assert _button_rows(view=view) == {"bj:insure_yes": 1, "bj:insure_no": 1}


async def test_settled_phase_removes_every_button() -> None:
    """After settlement no controls remain attached to the view."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="K", suit="♣"), Card(rank="7", suit="♦")],
    )
    round_state.phase = "settled"
    round_state.finished = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert _button_ids(view=view) == set()


async def test_sync_buttons_drops_insurance_controls_outside_insurance() -> None:
    """Insurance buttons come and go with the phase rather than staying on disabled."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="8", suit="♠"), Card(rank="8", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    ids = _button_ids(view=view)
    assert "bj:insure_yes" not in ids
    assert "bj:insure_no" not in ids

    round_state.phase = "insurance"
    round_state.insurance_offered = True
    view.sync_buttons()

    ids = _button_ids(view=view)
    assert "bj:insure_yes" in ids
    assert "bj:insure_no" in ids


async def test_build_in_progress_embeds_force_show_hole_reveals_dealer_total() -> None:
    """`force_show_hole=True` flips the dealer hole card face-up for peek reveal."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="K", suit="♦")],
    )

    embeds = build_in_progress_embeds(
        round_state=round_state, system_name="賭場系統", system_avatar_url="", force_show_hole=True
    )
    dealer_embed = embeds[0]

    assert isinstance(dealer_embed.description, str)
    assert "A♣" in dealer_embed.description
    assert "K♦" in dealer_embed.description
    assert "🂠" not in dealer_embed.description


def test_blackjack_table_edit_payload_adds_width_spacer() -> None:
    """A table edit with no spacer uploaded yet sends one and points every embed at it."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="K", suit="♣"), Card(rank="9", suit="♦")],
    )
    talk_embed = Embed(description="短句")
    seat_embeds = build_in_progress_embeds(
        round_state=round_state, system_name="賭場系統", system_avatar_url=""
    )

    payload = blackjack_views._blackjack_table_edit_kwargs(
        embeds=[talk_embed, *seat_embeds], view=None
    )

    assert payload["attachments"] == []
    assert payload["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    for embed in payload["embeds"]:
        assert embed.image.url == embed_spacer_url()


async def test_interaction_check_sends_ephemeral_notice_when_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the round is settled, clicks get an ephemeral notice rather than silent ignore."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="K", suit="♣"), Card(rank="9", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view._settled = True

    notices: list[str] = []

    async def _fake_notice(
        *, interaction: Interaction[Any], content: str, log_message: str
    ) -> None:
        notices.append(content)

    monkeypatch.setattr(
        "discordbot.cogs.games.blackjack_views.send_ephemeral_notice", _fake_notice
    )

    interaction = MagicMock()
    interaction.user.id = 1
    allowed = await view.interaction_check(interaction=interaction)

    assert allowed is False
    assert notices == ["這局已經結束, 等下一局吧"]


async def test_play_dealer_hits_below_17_then_stands_on_hard_17(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dealer hits ≤16 and stands on a hard 17 under H17 rules."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "dealer"
    # Emptying the shoe is what routes every draw through the monkeypatched `draw_card` fallback.
    round_state.shoe = []
    view = _make_view(round_state=round_state)

    def _draw_six(rng: Random) -> Card:
        return Card(rank="6", suit="♠")

    monkeypatch.setattr("discordbot.cogs.games.blackjack.draw_card", _draw_six)
    await view._play_dealer_locked()

    assert round_state.dealer_played is True
    first_step = view._dealer_steps[0]
    assert first_step.action == "hit"
    assert first_step.source == "auto"
    assert first_step.forced is True
    assert first_step.total_before == 11
    assert first_step.total_after == 17
    final_step = view._dealer_steps[-1]
    assert final_step.action == "stand"
    assert final_step.source == "auto"
    assert final_step.forced is True


@pytest.mark.parametrize(
    argnames=("dealer_cards", "expected_total"),
    argvalues=[
        ([Card(rank="K", suit="♣"), Card(rank="7", suit="♦")], 17),
        ([Card(rank="K", suit="♣"), Card(rank="8", suit="♦")], 18),
    ],
)
async def test_play_dealer_stands_on_hard_17_plus(
    dealer_cards: list[Card], expected_total: int
) -> None:
    """Dealer stands deterministically on any hard 17+ total."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=dealer_cards,
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "dealer"
    view = _make_view(round_state=round_state)

    await view._play_dealer_locked()

    assert round_state.dealer_played is True
    assert round_state.dealer_total() == expected_total
    step = view._dealer_steps[-1]
    assert step.action == "stand"
    assert step.source == "auto"
    assert step.forced is True


async def test_play_dealer_hits_soft_17(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dealer hits soft 17 (H17 rule) instead of standing."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "dealer"
    round_state.shoe = []
    view = _make_view(round_state=round_state)

    def _draw_three(rng: Random) -> Card:
        return Card(rank="3", suit="♠")

    monkeypatch.setattr("discordbot.cogs.games.blackjack.draw_card", _draw_three)
    await view._play_dealer_locked()

    assert [str(card) for card in round_state.dealer] == ["A♣", "6♦", "3♠"]
    first_step = view._dealer_steps[0]
    assert first_step.action == "hit"
    assert first_step.source == "auto"
    assert "soft 17" in first_step.reason
    assert first_step.total_before == 17
    final_step = view._dealer_steps[-1]
    assert final_step.action == "stand"
    assert final_step.source == "auto"


async def test_bot_dispatcher_skips_when_no_bot_seated() -> None:
    """The bot turn dispatcher is a no-op when no bot is seated."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    assert view.bot_user_id is None
    message = MagicMock()
    await view._maybe_play_bot_turn_locked(message=message)
    assert message.edit.called is False


async def test_bot_dispatcher_skips_when_active_player_is_human() -> None:
    """If the active seat belongs to a human, the bot dispatcher returns immediately."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.bot_user_id = 999
    message = MagicMock()
    await view._maybe_play_bot_turn_locked(message=message)
    # The human's hand is untouched because the bot never acts on a human seat.
    assert message.edit.called is False
    assert len(round_state.players[0].hands[0].cards) == 2


async def test_bot_dispatcher_breaks_when_action_does_not_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dispatch that leaves `_state_revision` untouched ends the loop instead of re-deciding."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.bot_user_id = 1
    calls = 0

    async def no_op_dispatch(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(view, "_dispatch_bot_action_locked", no_op_dispatch)

    await view._maybe_play_bot_turn_locked(message=MagicMock())

    assert calls == 1


async def test_bot_dispatcher_paces_consecutive_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consecutive bot moves are paced by one sleep, and the move that ends the turn adds none."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
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

    await view._maybe_play_bot_turn_locked(message=MagicMock())

    assert dispatch_calls == 2
    assert sleep_calls == [blackjack_views.BOT_TURN_EDIT_DELAY_SECONDS]


async def test_bot_action_plays_ev_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bot plays the EV action deterministically (no LLM involved)."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="2", suit="♠"), Card(rank="3", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="10", suit="♦")],
    )
    round_state.shoe = [
        Card(rank="4", suit="♠"),
        Card(rank="9", suit="♥"),
        Card(rank="8", suit="♦"),
        Card(rank="7", suit="♣"),
        Card(rank="6", suit="♠"),
        Card(rank="2", suit="♥"),
    ]
    view = _make_view(round_state=round_state)
    monkeypatch.setattr(view, "_edit_in_progress_locked", AsyncMock())

    await view._dispatch_bot_action_locked(message=MagicMock(), active=round_state.players[0])

    # A stiff hard 5 is always a hit, so the EV engine drives the deterministic action.
    assert round_state.players[0].hands[0].cards[-1] == Card(rank="4", suit="♠")


async def test_apply_bot_action_routes_known_actions() -> None:
    """An allowed stand routes through `BlackjackRound.stand` and reports success."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)

    applied = view._apply_bot_action(user_id=1, action="stand", allowed=("hit", "stand"))
    assert applied is True
    assert round_state.players[0].hands[0].finished is True


async def test_apply_bot_action_rejects_action_not_in_allowed() -> None:
    """Actions not in `allowed` are rejected without raising."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
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
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.shoe = [
        Card(rank="7", suit="♠"),
        Card(rank="8", suit="♥"),
        Card(rank="2", suit="♦"),
        Card(rank="3", suit="♣"),
    ]
    view = BlackjackView(
        round_state=round_state, starter_id=1, author_name="alice", shoe_store=store, channel_id=42
    )
    view.message = MagicMock()
    monkeypatch.setattr(view, "_safe_edit_view_locked", AsyncMock())

    async def _stop_after_save(**_kwargs: object) -> None:
        raise RuntimeError("stop after shoe save")

    # Settlement runs after the shoe save, so raising there proves the save already ran.
    monkeypatch.setattr(blackjack_views, "settle_blackjack_player", _stop_after_save)

    with pytest.raises(RuntimeError, match="stop after shoe save"):
        await view.finalize(message=view.message)

    # The store holds a decoupled copy of the round's remaining shoe.
    assert store.shoes.get(42) == round_state.shoe
    assert store.shoes.get(42) is not round_state.shoe


async def test_history_persistence_uses_scheduled_dealer_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History persistence uses the dealer cards captured when the task is scheduled."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    dealer_cards = list(round_state.dealer)
    dealer_total = round_state.dealer_total()
    captured: dict[str, object] = {}

    async def fake_record_blackjack_history(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(blackjack_views, "record_blackjack_history", fake_record_blackjack_history)
    round_state.dealer.append(Card(rank="K", suit="♣"))
    await view._record_history_later(
        message=as_message(fake=SimpleNamespace(id=999, guild=SimpleNamespace(id=888))),
        results=[
            BlackjackPlayerResult(
                participant=round_state.players[0].participant,
                settlement=BlackjackPlayerSettlement(
                    delta=100,
                    payout=100,
                    new_balance=1_100,
                    casino_balance=0,
                    base_delta=100,
                    vip_bonus=0,
                    is_vip=False,
                    outcome="win",
                    detail="",
                    hands=[
                        BlackjackHandSettlement(
                            cards=round_state.players[0].hands[0].cards,
                            bet=100,
                            outcome="win",
                            delta=100,
                        )
                    ],
                ),
            )
        ],
        dealer_cards=dealer_cards,
        dealer_total=dealer_total,
    )

    assert captured["dealer_cards"] == [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    assert captured["dealer_total"] == 11
