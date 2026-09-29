"""Tests for the Blackjack table view: its controls, dealer play, bot turns, settling, rendering.

Controls are presence-based, so the button tests assert which custom_ids are
attached rather than which are disabled. Dealer play is deterministic (H17) and
the bot's decisions come from the EV engine, so both are asserted exactly.
"""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random
from typing import Any, cast
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from nextcord import Interaction
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
from discordbot.cogs.games.presentation import settlement_metadata
from discordbot.services.economy.database import get_balance, get_casino_ledger
from discordbot.cogs.games.blackjack_views import (
    BlackjackView,
    build_final_embeds,
    build_in_progress_embeds,
)

from tests.helpers.games import (
    seat,
    component_ids,
    component_rows,
    attached_button,
    settle_only_seat,
    longest_hand_the_dealer_must_draw_on,
)
from tests.helpers.casting import as_message, as_interaction
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
        rng=Random(x=0), participants=[player or seat()], auto_play_dealer=False
    )
    hand = round_state.players[0].hands[0]
    hand.cards = player_cards
    round_state.dealer = dealer_cards
    if finished:
        hand.finished = True
        round_state.finished = True
        round_state.dealer_played = True
        round_state.phase = "settled"
    return round_state


def _make_view(round_state: BlackjackRound) -> BlackjackView:
    """Builds a BlackjackView for button inspection."""
    return BlackjackView(
        round_state=round_state,
        starter_id=1,
        author_name="alice",
        system_name="賭場系統",
        system_avatar_url="",
    )


@pytest.fixture
def scheduled_cleanups(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Records each table the view hands to the public-message cleanup instead of scheduling it."""
    scheduled: list[object] = []

    def record(message: object, delay: float = 180, user_name: str | None = None) -> None:
        del delay, user_name
        scheduled.append(message)

    monkeypatch.setattr(blackjack_views, "schedule_public_message_delete", record)
    return scheduled


def _button_states(view: BlackjackView) -> dict[str, bool]:
    """Returns `{custom_id: disabled}` for every button in the view."""
    states: dict[str, bool] = {}
    for child in view.children:
        cid = getattr(child, "custom_id", None)
        if cid is not None and isinstance(child, Button):
            states[cid] = bool(child.disabled)
    return states


async def test_player_actions_same_rank_pair_enables_every_action_button() -> None:
    """Initial deal with [8, 8] vs dealer up 6 enables all five action buttons."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="8", suit="♠"), Card(rank="8", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert component_ids(view=view) == {
        "bj:hit",
        "bj:stand",
        "bj:double",
        "bj:split",
        "bj:surrender",
    }
    assert all(disabled is False for disabled in _button_states(view=view).values())
    assert component_rows(view=view) == {
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

    assert "bj:split" in component_ids(view=view)


async def test_player_actions_ace_ten_hides_split() -> None:
    """A + 10 is not a same-value pair."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="A", suit="♠"), Card(rank="10", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    ids = component_ids(view=view)
    assert "bj:hit" in ids
    assert "bj:stand" in ids
    assert "bj:double" in ids
    assert "bj:split" not in ids
    assert "bj:surrender" in ids


async def test_player_actions_after_hit_removes_double_split_surrender() -> None:
    """After a Hit the first-action-only controls leave the view instead of being disabled."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="5", suit="♠"), Card(rank="6", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].cards.append(Card(rank="4", suit="♣"))
    round_state.players[0].hands[0].actions_taken = 1
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert component_ids(view=view) == {"bj:hit", "bj:stand"}
    assert all(disabled is False for disabled in _button_states(view=view).values())


async def test_player_actions_is_split_hand_removes_double_split_surrender() -> None:
    """A hand born out of Split cannot be doubled (no DAS), re-split, or surrendered."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="8", suit="♠"), Card(rank="3", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].is_split_hand = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert component_ids(view=view) == {"bj:hit", "bj:stand"}
    assert all(disabled is False for disabled in _button_states(view=view).values())


async def test_split_aces_subhand_removes_hit_and_stand() -> None:
    """Split Aces removes Hit and Stand with `finished` still False; Split removed the rest."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat()], auto_play_dealer=False
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

    assert component_ids(view=view) == set()


async def test_player_actions_low_balance_removes_double_and_split() -> None:
    """Insufficient balance for the extra wager hides Double and Split affordances."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(balance_at_start=150)], auto_play_dealer=False
    )
    round_state.players[0].hands[0].cards = [Card(rank="8", suit="♠"), Card(rank="8", suit="♥")]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    ids = component_ids(view=view)
    assert "bj:hit" in ids
    assert "bj:stand" in ids
    assert "bj:double" not in ids
    assert "bj:split" not in ids
    assert "bj:surrender" in ids


async def test_player_actions_peeked_blackjack_removes_surrender() -> None:
    """A revealed dealer Blackjack closes the Surrender window."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="9", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="K", suit="♦")],
    )
    round_state.peeked_blackjack = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert "bj:surrender" not in component_ids(view=view)


async def test_insurance_phase_hides_action_buttons_and_shows_insurance() -> None:
    """During insurance only insure_yes / insure_no are interactive."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="5", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="9", suit="♦")],
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
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="K", suit="♣"), Card(rank="7", suit="♦")],
    )
    round_state.phase = "settled"
    round_state.finished = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    assert component_ids(view=view) == set()


async def test_sync_buttons_drops_insurance_controls_outside_insurance() -> None:
    """Insurance buttons join the view for the insurance phase alone and leave it after."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="8", suit="♠"), Card(rank="8", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
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


def test_settlement_metadata_shows_vip_bonus_numbers() -> None:
    """A VIP-boosted win shows the total delta and the VIP bonus inside it."""
    metadata = settlement_metadata(
        delta=150, new_balance=1_150, is_allin=False, base_delta=100, vip_bonus=50
    )

    assert metadata == "-# 本局 `+150` · VIP加成 `+50` · 餘額 `1,150`"


def test_blackjack_in_progress_dealer_seat_hides_hole_card() -> None:
    """The dealer seat embed shows one hidden card marker plus the visible up-card."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(display_name="Bob")]
    )
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    round_state.dealer = [Card(rank="8", suit="♣"), Card(rank="K", suit="♦")]

    embeds = build_in_progress_embeds(
        round_state=round_state, system_name="賭場系統", system_avatar_url=""
    )
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
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    round_state.dealer = [Card(rank="8", suit="♣")]

    embeds = build_in_progress_embeds(
        round_state=round_state, system_name="賭場系統", system_avatar_url=""
    )
    dealer_embed = embeds[0]

    assert isinstance(dealer_embed.description, str)
    assert "8♣" in dealer_embed.description
    assert "🂠" not in dealer_embed.description


# Helper predicates ---------------------------------------------------------


def test_blackjack_table_edit_payload_adds_width_spacer() -> None:
    """Blackjack table edits attach one transparent spacer and reference it from every embed."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="K", suit="♣"), Card(rank="9", suit="♦")],
    )
    seat_embeds = build_in_progress_embeds(
        round_state=round_state, system_name="賭場系統", system_avatar_url=""
    )

    payload = blackjack_views.table_edit_kwargs(embeds=seat_embeds, view=None)

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


def test_the_dealer_loop_outlasts_the_longest_hand_the_rules_can_force() -> None:
    """`MAX_DEALER_DECISION_STEPS` must not stand the dealer below 17.

    `_play_dealer_locked` spends one iteration per drawn card and one more to record the stand
    or the bust; running out instead appends a `source="guard"` step, which settles the round on
    whatever total the dealer was holding and shows the players it did that.
    """
    longest = longest_hand_the_dealer_must_draw_on()

    assert longest == 12, f"the longest forced hand moved to {longest} cards; re-read the bound"
    needed = (longest - 2) + 2
    assert needed <= blackjack_views.MAX_DEALER_DECISION_STEPS, (
        f"the dealer would stand below 17 on a {longest}-card hand: the loop needs {needed} "
        f"iterations and has {blackjack_views.MAX_DEALER_DECISION_STEPS}"
    )


async def test_a_seat_that_can_never_insure_is_not_sent_to_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1-point seat's half-bet rounds to zero, and no newer table will change that.

    The insurance buttons belong to the table rather than to a seat — every undecided player
    sees the same pair — so the seat that cannot use them only finds out by pressing, and what
    it is told then is the whole of the feature for it.
    """
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="9", suit="♦")],
        player=seat(bet=1),
    )
    round_state.phase = "insurance"
    round_state.insurance_offered = True
    view = _make_view(round_state=round_state)

    notices: list[str] = []

    async def _fake_notice(
        *, interaction: Interaction[Any], content: str, log_message: str
    ) -> None:
        notices.append(content)

    monkeypatch.setattr(
        "discordbot.cogs.games.blackjack_views.send_ephemeral_notice", _fake_notice
    )
    monkeypatch.setattr(BlackjackView, "_edit_in_progress_locked", AsyncMock(return_value=None))

    decided = await view._take_insurance_locked(
        interaction=MagicMock(), message=MagicMock(), user_id=1
    )

    assert decided is False
    assert round_state.players[0].insurance_bet == 0
    assert notices == ["你的下注太小，一半不到 1 點，這局沒有保險可買"]


async def test_play_dealer_hits_below_17_then_stands_on_hard_17() -> None:
    """Dealer hits ≤16 and stands on a hard 17 under H17 rules."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "dealer"
    round_state.shoe = [Card(rank="6", suit="♠")]
    view = _make_view(round_state=round_state)

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


async def test_play_dealer_hits_soft_17() -> None:
    """Dealer hits soft 17 (H17 rule) instead of standing."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "dealer"
    round_state.shoe = [Card(rank="3", suit="♠")]
    view = _make_view(round_state=round_state)

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
    """A no-op bot dispatch exits instead of spinning on the same turn."""
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
    """Consecutive bot-owned decisions wait briefly between message edits."""
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
    """The bot plays the EV engine's hole-aware action where the up-card table would not.

    Hard 16 against a 10 with Surrender on offer is a surrender by the table, but the hole is a
    6 and the shoe holds only tens, so the dealer's 16 must draw and bust: the bot stands. It
    sits in the first seat so the stand hands the turn on instead of settling the table.
    """
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[seat(user_id=1, display_name="Bot"), seat(user_id=2, display_name="Bob")],
        auto_play_dealer=False,
    )
    bot_hand = round_state.players[0].hands[0]
    bot_hand.cards = [Card(rank="10", suit="♠"), Card(rank="6", suit="♥")]
    round_state.players[1].hands[0].cards = [Card(rank="9", suit="♣"), Card(rank="8", suit="♦")]
    round_state.dealer = [Card(rank="6", suit="♣"), Card(rank="10", suit="♦")]
    round_state.shoe = [Card(rank="10", suit="♠")] * 20
    view = _make_view(round_state=round_state)
    monkeypatch.setattr(view, "_edit_in_progress_locked", AsyncMock())

    await view._dispatch_bot_action_locked(message=MagicMock(), active=round_state.players[0])

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
            dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
        )
        view = _make_view(round_state=round_state)
        assert view._apply_bot_action(user_id=1, action=action, allowed=(action,)) is True
        return round_state.players[0]

    stiff = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    pair = [Card(rank="8", suit="♠"), Card(rank="8", suit="♥")]

    assert len(_apply(action="hit", player_cards=stiff).hands[0].cards) == 3
    assert _apply(action="stand", player_cards=stiff).hands[0].finished is True
    doubled = _apply(action="double", player_cards=stiff).hands[0]
    assert (doubled.doubled, doubled.bet) == (True, 200)
    assert len(_apply(action="split", player_cards=pair).hands) == 2
    assert _apply(action="surrender", player_cards=stiff).hands[0].surrendered is True


async def test_apply_bot_action_reports_a_refused_round_call_as_unapplied() -> None:
    """A `ValueError` out of `BlackjackRound` is caught, so the caller can fall back to a stand."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)

    # Offered by the caller but refused by the round itself: 10/7 is not a pair.
    assert view._apply_bot_action(user_id=1, action="split", allowed=("split",)) is False
    assert len(round_state.players[0].hands) == 1


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


async def test_history_persistence_uses_the_dealer_hand_captured_at_settlement(
    monkeypatch: pytest.MonkeyPatch, scheduled_cleanups: list[object]
) -> None:
    """The round history records the dealer hand as it stood when the round settled.

    The write runs in a background task after `finalize` returns, so it has to be handed a copy:
    anything that touches the live round's dealer list afterwards would otherwise reach the row.
    """
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
        player=seat(bet=50, balance_at_start=100),
        finished=True,
    )
    view = _make_view(round_state=round_state)
    recorded: dict[str, object] = {}

    async def record_blackjack_history(**kwargs: object) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(blackjack_views, "record_blackjack_history", record_blackjack_history)

    await view.finalize(message=as_message(fake=FakeDiscordMessage(guild=FakeGuild(guild_id=888))))
    round_state.dealer.append(Card(rank="K", suit="♣"))
    await view.wait_for_background_tasks()

    assert recorded["dealer_cards"] == [Card(rank="10", suit="♣"), Card(rank="8", suit="♦")]
    assert recorded["dealer_total"] == 18
    assert recorded["guild_id"] == 888


# Settling a table ---------------------------------------------------------


async def test_blackjack_view_finalizes_once_when_called_concurrently(
    scheduled_cleanups: list[object],
) -> None:
    """Concurrent finalization attempts must not pay out one Blackjack hand twice."""
    await seed_balance(user_id=1, name="alice", amount=100)

    message = FakeDiscordMessage()
    view = _make_view(
        round_state=_round_with_two_cards(
            player_cards=[Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")],
            dealer_cards=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
            player=seat(bet=50, balance_at_start=100),
            finished=True,
        )
    )

    await asyncio.gather(
        view.finalize(message=as_message(fake=message)),
        view.finalize(message=as_message(fake=message)),
    )

    assert await get_balance(user_id=1) == 150
    ledger = await get_casino_ledger()
    assert ledger.balance == -50
    assert "embeds" not in message.edits[0]
    await view.wait_for_background_tasks()
    assert len(message.edits) == 2
    assert message.edits[1]["view"] is None
    assert scheduled_cleanups == [message]


async def test_blackjack_view_timeout_auto_stands_and_settles(
    scheduled_cleanups: list[object],
) -> None:
    """A player who walks away is treated as standing and the wager resolves."""
    await seed_balance(user_id=1, name="alice", amount=100)

    message = FakeDiscordMessage()
    view = _make_view(
        round_state=_round_with_two_cards(
            player_cards=[Card(rank="10", suit="♠"), Card(rank="8", suit="♥")],
            dealer_cards=[Card(rank="10", suit="♣"), Card(rank="Q", suit="♦")],
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
    assert scheduled_cleanups == [message]


async def test_blackjack_view_dealer_plays_h17_rule(scheduled_cleanups: list[object]) -> None:
    """Dealer plays deterministically under H17 (hits below 17, stands on hard 17+)."""
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="10", suit="♣"), Card(rank="3", suit="♦")],
        player=seat(bet=50, balance_at_start=100),
    )
    round_state.shoe = [Card(rank="5", suit="♣")]

    message = FakeDiscordMessage()
    view = _make_view(round_state=round_state)

    await view.finalize(message=as_message(fake=message))

    assert [str(card) for card in view.round_state.dealer] == ["10♣", "3♦", "5♣"]
    assert view.round_state.dealer_played is True
    assert await get_balance(user_id=1) == 50
    ledger = await get_casino_ledger()
    assert ledger.balance == 50
    assert "embeds" not in message.edits[0]
    final_embeds = message.edits[1]["embeds"]
    description = cast("str", final_embeds[0].description)
    assert "規則: 13 hit 抽 5♣ → 18" in description
    await view.wait_for_background_tasks()
    assert scheduled_cleanups == [message]


async def test_blackjack_view_dealer_hits_soft_17(scheduled_cleanups: list[object]) -> None:
    """Soft 17 forces a hit under the H17 rule."""
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="6", suit="♦")],
        player=seat(bet=50, balance_at_start=100),
    )
    round_state.shoe = [Card(rank="K", suit="♠")]

    message = FakeDiscordMessage()
    view = _make_view(round_state=round_state)

    await view.finalize(message=as_message(fake=message))

    # Soft 17 must trigger a draw; the drawn K lands a hard 17, where the dealer stands.
    assert len(view.round_state.dealer) >= 3
    assert view.round_state.dealer_played is True
    assert "embeds" not in message.edits[0]
    final_embeds = message.edits[1]["embeds"]
    description = cast("str", final_embeds[0].description)
    assert "規則: 17 hit" in description
    await view.wait_for_background_tasks()
    assert scheduled_cleanups == [message]


async def test_blackjack_view_locks_actions_while_finalizing(
    monkeypatch: pytest.MonkeyPatch, scheduled_cleanups: list[object]
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
            payout=50,
            new_balance=150,
            casino_balance=-50,
            hands=[
                BlackjackHandSettlement(
                    cards=[Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")],
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
            player_cards=[Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")],
            dealer_cards=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
            player=seat(bet=50, balance_at_start=50),
        )
    )

    hit_button = attached_button(view=view, custom_id="bj:hit")
    stand_button = attached_button(view=view, custom_id="bj:stand")
    stand_task = asyncio.create_task(
        coro=stand_button.callback(as_interaction(fake=FakeInteraction(message=message)))
    )
    await settlement_started.wait()

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
    assert scheduled_cleanups == [message]


def _alice_done_bob_to_act() -> BlackjackRound:
    """Builds a two-seat round where Alice has finished and Bob holds the turn."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[
            seat(user_id=1, display_name="Alice", bet=50, balance_at_start=100),
            seat(user_id=2, display_name="Bob", bet=50, balance_at_start=100),
        ],
        auto_play_dealer=False,
    )
    alice = round_state.players[0].hands[0]
    alice.cards = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    alice.finished = True
    round_state.players[1].hands[0].cards = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    round_state.dealer = [Card(rank="9", suit="♣"), Card(rank="7", suit="♦")]
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
        rng=Random(x=0), participants=[seat(bet=50, balance_at_start=100)], auto_play_dealer=False
    )
    player = round_state.players[0]
    player.hands = [
        BlackjackHandState(
            cards=[Card(rank="10", suit="♠"), Card(rank="2", suit="♥")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
            finished=True,
        ),
        BlackjackHandState(
            cards=[Card(rank="9", suit="♣"), Card(rank="2", suit="♦")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
        ),
    ]
    round_state.dealer = [Card(rank="9", suit="♥"), Card(rank="7", suit="♦")]
    round_state.current_hand_index = 1
    round_state.shoe = [Card(rank="5", suit="♣")]

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
            Card(rank="2", suit="♠"),
            Card(rank="3", suit="♥"),
            Card(rank="4", suit="♣"),
            Card(rank="5", suit="♦"),
            Card(rank=last_card, suit="♠"),
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
        last_card="7", dealer_cards=[Card(rank="10", suit="♣"), Card(rank="9", suit="♦")]
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
            Card(rank="7", suit="♣"),
            Card(rank="7", suit="♦"),
            Card(rank="7", suit="♥"),
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
        player_cards=[Card(rank="9", suit="♠"), Card(rank="8", suit="♥")],
        dealer_cards=[Card(rank="K", suit="♣"), Card(rank="A", suit="♦")],
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
