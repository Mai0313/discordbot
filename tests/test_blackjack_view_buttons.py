"""Button-state matrix and finalize-flow tests for ``BlackjackView``.

Each test instantiates the view with a stubbed ``DealerAI`` and a
deterministic ``BlackjackRound``, then asserts which action / insurance
buttons are attached across the round lifecycle. The finalize-flow tests
cover the regression fixes: early view stop, ephemeral notice on settled
clicks, deterministic hits below 17, and AI dealer decisions at 17+.
"""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random
from unittest.mock import MagicMock

import pytest
from nextcord import Interaction

from discordbot.typings.games import GameParticipant, BlackjackDealerDecision
from discordbot.cogs._games.blackjack import Card, BlackjackRound, BlackjackHandState
from discordbot.cogs._games.blackjack_views import (
    BlackjackView,
    build_in_progress_embed,
    _dealer_decision_table_state,
)


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


def _button_rows(view: BlackjackView) -> dict[str, int | None]:
    """Returns ``{custom_id: row}`` for every button in the view."""
    rows: dict[str, int | None] = {}
    for child in view.children:
        cid = getattr(child, "custom_id", None)
        if cid is not None:
            rows[cid] = getattr(child, "row", None)
    return rows


async def test_player_actions_same_rank_pair_enables_every_action_button() -> None:
    """Initial deal with [8, 8] vs dealer up 6 enables all five action buttons."""
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
    """A + 10 is not a same-value pair."""
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
    """After a Hit, ``actions_taken`` > 0 locks the first-action-only buttons."""
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

    assert _button_ids(view=view) == set()


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

    async def _fake_notice(*, interaction: Interaction, content: str, log_message: str) -> None:
        notices.append(content)

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.send_ephemeral_notice", _fake_notice
    )

    interaction = MagicMock()
    interaction.user.id = 1
    allowed = await view.interaction_check(interaction=interaction)

    assert allowed is False
    assert notices == ["這局已經結束, 等下一局吧"]


async def test_play_dealer_hits_below_17_then_asks_llm_at_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dealer hits ≤16 deterministically, then asks the LLM once it reaches 17+."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "dealer"
    view = _make_view(round_state=round_state)

    decisions: list[int] = []

    async def _stand(
        *, author_name: str, table_state: str, dealer_total: int
    ) -> BlackjackDealerDecision:
        decisions.append(dealer_total)
        return BlackjackDealerDecision(action="stand", reason="AI stands at threshold")

    view.dealer.decide_blackjack_action = _stand

    def _draw_six(rng: Random) -> Card:
        return Card(rank="6", suit="♠")

    monkeypatch.setattr("discordbot.cogs._games.blackjack.draw_card", _draw_six)
    await view._play_dealer_locked()

    assert round_state.dealer_played is True
    assert decisions == [17]
    first_step = view._dealer_steps[0]
    assert first_step.action == "hit"
    assert first_step.source == "auto"
    assert first_step.forced is True
    assert first_step.total_before == 11
    assert first_step.total_after == 17
    final_step = view._dealer_steps[-1]
    assert final_step.action == "stand"
    assert final_step.source == "ai"
    assert final_step.forced is False
    assert final_step.reason == "AI stands at threshold"


@pytest.mark.parametrize(
    argnames=("dealer_cards", "expected_total"),
    argvalues=[
        ([Card(rank="K", suit="♣"), Card(rank="7", suit="♦")], 17),
        ([Card(rank="A", suit="♣"), Card(rank="6", suit="♦")], 17),
        ([Card(rank="K", suit="♣"), Card(rank="8", suit="♦")], 18),
    ],
)
async def test_play_dealer_calls_llm_on_17_plus(
    dealer_cards: list[Card], expected_total: int
) -> None:
    """Dealer lets the LLM decide on every 17+ total."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=dealer_cards,
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "dealer"
    view = _make_view(round_state=round_state)
    decisions: list[int] = []

    async def _stand(
        *, author_name: str, table_state: str, dealer_total: int
    ) -> BlackjackDealerDecision:
        decisions.append(dealer_total)
        return BlackjackDealerDecision(action="stand", reason="AI chooses stand")

    view.dealer.decide_blackjack_action = _stand

    await view._play_dealer_locked()

    assert decisions == [expected_total]
    assert round_state.dealer_played is True
    assert round_state.dealer_total() == expected_total
    step = view._dealer_steps[-1]
    assert step.action == "stand"
    assert step.source == "ai"
    assert step.forced is False
    assert step.reason == "AI chooses stand"


async def test_play_dealer_obeys_llm_hit_on_17_plus(monkeypatch: pytest.MonkeyPatch) -> None:
    """When DealerAI says hit at 17+, the dealer really draws."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="K", suit="♣"), Card(rank="8", suit="♦")],
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "dealer"
    view = _make_view(round_state=round_state)

    def _draw_queen(rng: Random) -> Card:
        return Card(rank="Q", suit="♠")

    async def _hit(
        *, author_name: str, table_state: str, dealer_total: int
    ) -> BlackjackDealerDecision:
        return BlackjackDealerDecision(action="hit", reason="AI chooses hit")

    monkeypatch.setattr("discordbot.cogs._games.blackjack.draw_card", _draw_queen)
    view.dealer.decide_blackjack_action = _hit

    await view._play_dealer_locked()

    assert [str(card) for card in round_state.dealer] == ["K♣", "8♦", "Q♠"]
    first_step = view._dealer_steps[0]
    assert first_step.action == "hit"
    assert first_step.source == "ai"
    assert first_step.reason == "AI chooses hit"
    assert first_step.forced is False


def test_dealer_decision_table_state_includes_player_actions_and_insurance() -> None:
    """DealerAI receives player totals, action context, and insurance state."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[
            _participant(user_id=1, display_name="Alice", bet=100),
            _participant(user_id=2, display_name="Bob", bet=100),
        ],
        auto_play_dealer=False,
    )
    alice_hand = round_state.players[0].hands[0]
    alice_hand.cards = [
        Card(rank="7", suit="♠"),
        Card(rank="5", suit="♥"),
        Card(rank="5", suit="♣"),
    ]
    alice_hand.actions_taken = 1
    alice_hand.finished = True
    round_state.players[0].insurance_bet = 50
    round_state.players[0].insurance_resolved = True
    bob_hand = round_state.players[1].hands[0]
    bob_hand.cards = [Card(rank="10", suit="♦"), Card(rank="8", suit="♣")]
    bob_hand.finished = True
    round_state.players[1].insurance_resolved = True
    round_state.dealer = [Card(rank="9", suit="♣"), Card(rank="A", suit="♦")]
    round_state.insurance_offered = True
    round_state.phase = "dealer"

    table_state = _dealer_decision_table_state(round_state=round_state)

    assert "莊家總點數: 20" in table_state
    assert "保險是否提供: 是" in table_state
    assert "Alice" in table_state
    assert "total=17" in table_state
    assert "status=stand" in table_state
    assert "player_draws_after_initial=1" in table_state
    assert "actions_taken=1" in table_state
    assert "insurance=taken, bet=50" in table_state
    assert "Bob" in table_state
    assert "total=18" in table_state
    assert "insurance=declined" in table_state
