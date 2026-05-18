"""Button-state matrix and finalize-flow tests for ``BlackjackView``.

Each test instantiates the view with a stubbed ``DealerAI`` and a
deterministic ``BlackjackRound``, then asserts the ``disabled`` flag and
presence of every action / insurance button across the round lifecycle.
The finalize-flow tests cover the regression fixes: early view stop,
ephemeral notice on settled clicks, dealer LLM skip below 17, and 18+
override safety.
"""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random
from unittest.mock import MagicMock

import pytest

from discordbot.typings.games import GameParticipant, BlackjackDealerDecision
from discordbot.cogs._games.blackjack import Card, BlackjackRound, BlackjackHandState
from discordbot.cogs._games.blackjack_views import BlackjackView, build_in_progress_embed


def _participant(user_id: int, display_name: str, bet: int = 100) -> GameParticipant:
    """Builds a participant with a 1k starting balance for affordability tests."""
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
    """Builds a single-player round with deterministic cards and dealer hand."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[_participant(user_id=1, display_name="Alice", bet=bet)],
        auto_play_dealer=False,
    )
    round_state.players[0].hands[0].cards = player_cards
    round_state.dealer = dealer_cards
    return round_state


def _make_view(round_state: BlackjackRound) -> BlackjackView:
    """Builds a BlackjackView with a stubbed dealer for button inspection."""
    dealer = MagicMock()
    return BlackjackView(
        dealer=dealer,
        round_state=round_state,
        starter_id=1,
        author_name="alice",
        dealer_id=999,
        dealer_name="Dealer",
        dealer_avatar_url="",
        dealer_line="...",
    )


def _button_states(view: BlackjackView) -> dict[str, bool]:
    """Returns ``{custom_id: disabled}`` for every button in the view."""
    states: dict[str, bool] = {}
    for child in view.children:
        cid = getattr(child, "custom_id", None)
        if cid is not None:
            states[cid] = bool(child.disabled)
    return states


def _button_ids(view: BlackjackView) -> set[str]:
    """Returns the set of button custom_ids currently attached to the view."""
    ids: set[str] = set()
    for child in view.children:
        cid = getattr(child, "custom_id", None)
        if cid is not None:
            ids.add(cid)
    return ids


async def test_player_actions_same_rank_pair_enables_every_action_button() -> None:
    """Initial deal with [8, 8] vs dealer up 6 enables all five action buttons."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="8", suit="♠"), Card(rank="8", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    states = _button_states(view=view)
    assert states["bj:hit"] is False
    assert states["bj:stand"] is False
    assert states["bj:double"] is False
    assert states["bj:split"] is False
    assert states["bj:surrender"] is False
    assert "bj:insure_yes" not in states
    assert "bj:insure_no" not in states


async def test_player_actions_different_rank_disables_split_only() -> None:
    """10 + K cannot be split (strict rank rule); double / surrender stay open."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="K", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    states = _button_states(view=view)
    assert states["bj:hit"] is False
    assert states["bj:stand"] is False
    assert states["bj:double"] is False
    assert states["bj:split"] is True
    assert states["bj:surrender"] is False


async def test_player_actions_after_hit_disables_double_split_surrender() -> None:
    """After a Hit, ``actions_taken`` > 0 locks the first-action-only buttons."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="5", suit="♠"), Card(rank="6", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].cards.append(Card(rank="4", suit="♣"))
    round_state.players[0].hands[0].actions_taken = 1
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    states = _button_states(view=view)
    assert states["bj:hit"] is False
    assert states["bj:stand"] is False
    assert states["bj:double"] is True
    assert states["bj:split"] is True
    assert states["bj:surrender"] is True


async def test_player_actions_is_split_hand_disables_double_split_surrender() -> None:
    """A hand born out of Split cannot be doubled (no DAS), re-split, or surrendered."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="8", suit="♠"), Card(rank="3", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].is_split_hand = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    states = _button_states(view=view)
    assert states["bj:hit"] is False
    assert states["bj:stand"] is False
    assert states["bj:double"] is True
    assert states["bj:split"] is True
    assert states["bj:surrender"] is True


async def test_split_aces_subhand_disables_hit_and_stand() -> None:
    """Split Aces forces finished + is_split_aces, blocking Hit and Stand."""
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

    states = _button_states(view=view)
    assert states["bj:hit"] is True
    assert states["bj:stand"] is False


async def test_player_actions_low_balance_disables_double_and_split() -> None:
    """Insufficient balance for the extra wager hides Double and Split affordances."""
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

    states = _button_states(view=view)
    assert states["bj:double"] is True
    assert states["bj:split"] is True
    assert states["bj:surrender"] is False


async def test_player_actions_peeked_blackjack_disables_surrender() -> None:
    """A revealed dealer Blackjack closes the Surrender window."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="9", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="K", suit="♦")],
    )
    round_state.peeked_blackjack = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    states = _button_states(view=view)
    assert states["bj:surrender"] is True


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
    assert states["bj:hit"] is True
    assert states["bj:stand"] is True
    assert states["bj:double"] is True
    assert states["bj:split"] is True
    assert states["bj:surrender"] is True
    assert "bj:insure_yes" in states
    assert "bj:insure_no" in states
    assert states["bj:insure_yes"] is False
    assert states["bj:insure_no"] is False


async def test_settled_phase_disables_every_button() -> None:
    """After settlement no button should accept further clicks."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="K", suit="♣"), Card(rank="7", suit="♦")],
    )
    round_state.phase = "settled"
    round_state.finished = True
    view = _make_view(round_state=round_state)
    view.sync_buttons()

    states = _button_states(view=view)
    for cid in ("bj:hit", "bj:stand", "bj:double", "bj:split", "bj:surrender"):
        assert states[cid] is True


async def test_sync_buttons_drops_insurance_controls_outside_insurance() -> None:
    """Insurance buttons leave the view entirely when phase is not insurance."""
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


async def test_build_in_progress_embed_force_show_hole_reveals_dealer_total() -> None:
    """``force_show_hole=True`` flips the hole card face-up for peek reveal."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="A", suit="♣"), Card(rank="K", suit="♦")],
    )

    embed = build_in_progress_embed(
        dealer_name="Dealer", round_state=round_state, force_show_hole=True
    )

    assert isinstance(embed.description, str)
    assert "A♣" in embed.description
    assert "K♦" in embed.description
    assert "🂠" not in embed.description


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

    async def _fake_notice(*, interaction: object, content: str, log_message: str) -> None:
        notices.append(content)

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.send_ephemeral_notice", _fake_notice
    )

    interaction = MagicMock()
    interaction.user.id = 1
    allowed = await view.interaction_check(interaction=interaction)

    assert allowed is False
    assert notices == ["這局已經結束, 等下一局吧"]


async def test_play_dealer_skips_llm_below_17() -> None:
    """Dealer totals ≤ 16 must hit deterministically without invoking the LLM."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "dealer"
    view = _make_view(round_state=round_state)

    async def _unreachable(**kwargs: object) -> BlackjackDealerDecision:
        msg = "LLM must not be called for dealer total <= 16"
        raise AssertionError(msg)

    view.dealer.decide_blackjack_action = _unreachable

    seeded_rng = Random(x=42)
    round_state.rng = seeded_rng
    await view._play_dealer_locked()

    assert round_state.dealer_played is True
    assert any(step.forced for step in view._dealer_steps)


async def test_play_dealer_overrides_18_plus_hit_to_stand() -> None:
    """When the LLM suggests hit at 18+, the view must force stand for safety."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="K", suit="♣"), Card(rank="8", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "dealer"
    view = _make_view(round_state=round_state)

    async def _aggressive_llm(**kwargs: object) -> BlackjackDealerDecision:
        return BlackjackDealerDecision(action="hit", reason="LLM reckless suggestion")

    view.dealer.decide_blackjack_action = _aggressive_llm

    await view._play_dealer_locked()

    assert round_state.dealer_played is True
    assert round_state.dealer_total() == 18
    step = view._dealer_steps[-1]
    assert step.action == "stand"
    assert step.forced is True
    assert "override" in step.reason
