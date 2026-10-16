"""Tests for the Blackjack table's text: early-finish notes, hands and settlement lines."""

import pytest

from discordbot.cogs.games.presentation import (
    render_hand,
    settlement_metadata,
    blackjack_player_early_finish_note,
)

from tests.helpers.games import card, blackjack_round


def test_blackjack_early_finish_note_ignores_regular_twenty_one() -> None:
    """A non-natural 21 should not be described as an early Blackjack finish."""
    player = blackjack_round(
        hands=[[card(rank="7", suit="♣"), card(rank="7", suit="♦"), card(rank="7")]], dealer=[]
    ).players[0]
    assert (
        blackjack_player_early_finish_note(
            player=player,
            dealer=[card(rank="9"), card(rank="7", suit="♥")],
            peeked_blackjack=False,
        )
        is None
    )


def test_blackjack_player_early_finish_note_names_peeked_up_card() -> None:
    """Peek notes tell players the dealer used the visible up-card plus hole card."""
    player = blackjack_round(
        hands=[[card(rank="9"), card(rank="8", suit="♥")]], dealer=[]
    ).players[0]

    note = blackjack_player_early_finish_note(
        player=player,
        dealer=[card(rank="A", suit="♣"), card(rank="K", suit="♦")],
        peeked_blackjack=True,
    )

    assert note == "莊家明牌 K♦, peek 暗牌確認 Blackjack, 本局直接結算"


@pytest.mark.parametrize(
    argnames=("peeked_blackjack", "expected"),
    argvalues=[
        (False, "你起手 Blackjack, 本局直接結算"),
        (True, "莊家明牌 K♦, peek 暗牌確認 Blackjack, 你也起手 Blackjack, 本局直接平手"),
    ],
    ids=["player-natural", "both-natural"],
)
def test_blackjack_early_finish_note_names_a_dealt_natural(
    peeked_blackjack: bool, expected: str
) -> None:
    """A seat dealt a natural is told that is what ended its round."""
    player = blackjack_round(
        hands=[[card(rank="A"), card(rank="K", suit="♥")]], dealer=[]
    ).players[0]

    note = blackjack_player_early_finish_note(
        player=player,
        dealer=[card(rank="A", suit="♣"), card(rank="K", suit="♦")],
        peeked_blackjack=peeked_blackjack,
    )

    assert note == expected


def test_render_hand_hides_first_card() -> None:
    """When the hole card is hidden, only the up-card and a back glyph appear."""
    cards = [card(rank="A"), card(rank="K", suit="♥")]
    rendered = render_hand(cards=cards, hide_first=True)
    assert "🂠" in rendered
    assert "A" not in rendered
    assert "K" in rendered


def test_settlement_metadata_shows_vip_bonus_numbers() -> None:
    """A VIP-boosted win shows the total delta and the VIP bonus inside it."""
    metadata = settlement_metadata(delta=150, new_balance=1_150, is_allin=False, vip_bonus=50)

    assert metadata == "-# 本局 `+150` · VIP加成 `+50` · 餘額 `1,150`"
