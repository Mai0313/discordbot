"""Tests for the Blackjack table's text: early-finish notes, hands and settlement lines."""

from discordbot.cogs.games.presentation import (
    settlement_metadata,
    blackjack_player_early_finish_note,
)
from discordbot.services.economy.presentation import amount_code

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

    assert note is not None
    assert str(card(rank="K", suit="♦")) in note, "the up-card everyone saw"
    assert str(card(rank="A", suit="♣")) not in note, "never the hole"


def test_blackjack_early_finish_note_names_a_dealt_natural() -> None:
    """A seat dealt a natural is told that is what ended its round.

    Its own natural, a push against the dealer's, and the dealer's alone each read differently.
    """
    dealer = [card(rank="A", suit="♣"), card(rank="K", suit="♦")]
    natural, stiff = blackjack_round(
        hands=[
            [card(rank="A"), card(rank="K", suit="♥")],
            [card(rank="9"), card(rank="8", suit="♥")],
        ],
        dealer=[],
    ).players

    own = blackjack_player_early_finish_note(player=natural, dealer=dealer, peeked_blackjack=False)
    both = blackjack_player_early_finish_note(player=natural, dealer=dealer, peeked_blackjack=True)
    dealer_only = blackjack_player_early_finish_note(
        player=stiff, dealer=dealer, peeked_blackjack=True
    )

    assert own is not None
    assert both is not None
    assert str(card(rank="K", suit="♦")) in both, "the push names the dealer's up-card"
    assert len({own, both, dealer_only}) == 3


def test_settlement_metadata_shows_vip_bonus_numbers() -> None:
    """A VIP-boosted win shows the total delta and the VIP bonus inside it."""
    metadata = settlement_metadata(delta=150, new_balance=1_150, is_allin=False, vip_bonus=50)

    assert amount_code(amount=150, signed=True, compact=True) in metadata
    assert amount_code(amount=50, signed=True, compact=True) in metadata
    assert amount_code(amount=1_150, compact=True) in metadata
