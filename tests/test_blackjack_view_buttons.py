"""Button-state matrix and dealer / bot turn tests for `BlackjackView`.

Each test instantiates the view with a stubbed `SystemNarrator` and a
deterministic `BlackjackRound`, then asserts which action / insurance buttons
are attached across the round lifecycle. Dealer play is now deterministic
(H17), so dealer tests cover the rule path; bot-turn tests cover the new bot
player AI integration.
"""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random
from unittest.mock import MagicMock

import pytest
from nextcord import Embed, Interaction

from discordbot.cogs._games import blackjack_views
from discordbot.typings.games import GameParticipant
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.cogs._games.blackjack import Card, BlackjackRound, BlackjackHandState
from discordbot.cogs._games.blackjack_views import BlackjackView, build_in_progress_embeds


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
    """Builds a BlackjackView with a stubbed narrator for button inspection."""
    narrator = MagicMock()
    return BlackjackView(
        narrator=narrator,
        round_state=round_state,
        starter_id=1,
        author_name="alice",
        system_name="賭場系統",
        system_avatar_url="",
        system_line="...",
    )


def _button_states(view: BlackjackView) -> dict[str, bool]:
    """Returns `{custom_id: disabled}` for every button in the view."""
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
    """Returns `{custom_id: row}` for every button in the view."""
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
    """After a Hit, `actions_taken` > 0 locks the first-action-only buttons."""
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
    """Blackjack table edits attach one transparent spacer and reference it from every embed."""
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
    assert payload["file"].filename == DEFAULT_EMBED_SPACER_FILENAME
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
    round_state.shoe = []
    view = _make_view(round_state=round_state)

    def _draw_six(rng: Random) -> Card:
        return Card(rank="6", suit="♠")

    monkeypatch.setattr("discordbot.cogs._games.blackjack.draw_card", _draw_six)
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

    monkeypatch.setattr("discordbot.cogs._games.blackjack.draw_card", _draw_three)
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


async def test_bot_dispatcher_skips_when_bot_player_ai_missing() -> None:
    """The bot turn dispatcher is a no-op when no bot is seated."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    assert view.bot_player_ai is None
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
    bot_ai = MagicMock()
    view.bot_player_ai = bot_ai
    view.bot_user_id = 999
    message = MagicMock()
    await view._maybe_play_bot_turn_locked(message=message)
    assert bot_ai.decide_bot_action.called is False
    assert bot_ai.decide_bot_insurance.called is False


async def test_bot_dispatcher_breaks_when_action_does_not_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-op bot dispatch exits instead of spinning on the same turn."""
    round_state = _round_with_two_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer_cards=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    view = _make_view(round_state=round_state)
    view.bot_player_ai = MagicMock()
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
    view.bot_player_ai = MagicMock()
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


async def test_apply_bot_action_routes_known_actions() -> None:
    """`_apply_bot_action` calls the matching BlackjackRound API for each known action."""
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
