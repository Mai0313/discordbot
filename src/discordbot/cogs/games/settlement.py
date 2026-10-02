"""Settlement helpers for Blackjack rounds."""

from discordbot.typings.games import (
    Card,
    SettleOutcome,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
    BlackjackInsuranceSettlement,
)
from discordbot.typings.economy import apply_vip_blackjack_bonus
from discordbot.cogs.games.blackjack import (
    BlackjackRound,
    BlackjackHandState,
    BlackjackPlayerHand,
    settle_hand,
)
from discordbot.services.economy.database import get_vip, apply_blackjack_settlement


def _aggregate_outcome(
    hand_settlements: list[BlackjackHandSettlement],
    insurance: BlackjackInsuranceSettlement | None,
    base_delta: int,
) -> SettleOutcome:
    """Returns the single outcome label for a (possibly multi-hand) result."""
    if len(hand_settlements) == 1 and insurance is None:
        return hand_settlements[0].outcome
    if base_delta > 0:
        return "win"
    if base_delta < 0:
        return "lose"
    return "push"


def _hand_settlement_from_state(
    hand: BlackjackHandState, dealer: list[Card]
) -> BlackjackHandSettlement:
    """Wraps `settle_hand` into a `BlackjackHandSettlement` row."""
    outcome, delta = settle_hand(hand=hand, dealer=dealer)
    five_card_twenty_one = outcome == "five_card_twenty_one"
    return BlackjackHandSettlement(
        cards=list(hand.cards),
        bet=hand.bet,
        outcome=outcome,
        delta=delta,
        # One bet: `blackjack_ev.py::_stand_ev_unit` prices a five-card 21 on that size.
        five_card_bonus=hand.bet if five_card_twenty_one else 0,
        five_card_twenty_one=five_card_twenty_one,
        doubled=hand.doubled,
        surrendered=hand.surrendered,
        is_split_hand=hand.is_split_hand,
    )


def _insurance_settlement(
    player: BlackjackPlayerHand, peeked_blackjack: bool
) -> BlackjackInsuranceSettlement | None:
    """Computes the insurance side-bet result, if any was taken."""
    if player.insurance_bet <= 0:
        return None
    bet = player.insurance_bet
    if peeked_blackjack:
        return BlackjackInsuranceSettlement(bet=bet, won=True, delta=bet * 2)
    return BlackjackInsuranceSettlement(bet=bet, won=False, delta=-bet)


async def settle_blackjack_player(
    round_state: BlackjackRound, player: BlackjackPlayerHand
) -> BlackjackPlayerSettlement:
    """Settles every sub-hand plus insurance side bet for one participant.

    The aggregate casino-paid delta (per-hand deltas plus insurance) takes the
    VIP bonus once at the player level, never per hand. Five-card 21 adds a
    system-funded bonus to the player-side delta without moving the casino
    ledger, and the VIP bonus credited is the larger of the one on the
    dealer-paid win and the one on the five-card 21 bonus — a max, not a sum.

    Bets are not deducted when a round starts, so an unfinished in-memory round
    vanishes on bot restart without touching balances. The VIP flag is
    permanent, so reading it outside the settlement transaction is safe: a
    freshly-bought VIP that races a settlement only misses the bonus on that one
    in-flight round.

    Args:
        round_state: Round providing the dealer cards and peek state.
        player: Player to settle; its participant names the account written.

    Returns:
        Aggregated settlement covering every sub-hand and any insurance bet.
    """
    hand_settlements = [
        _hand_settlement_from_state(hand=hand, dealer=round_state.dealer) for hand in player.hands
    ]
    insurance = _insurance_settlement(player=player, peeked_blackjack=round_state.peeked_blackjack)
    base_delta = sum(settlement.delta for settlement in hand_settlements)
    if insurance is not None:
        base_delta += insurance.delta
    five_card_bonus = sum(settlement.five_card_bonus for settlement in hand_settlements)

    participant = player.participant
    is_vip = await get_vip(user_id=participant.user_id)
    casino_paid_delta = apply_vip_blackjack_bonus(delta=base_delta, is_vip=is_vip)
    casino_paid_vip_bonus = casino_paid_delta - base_delta
    five_card_vip_delta = apply_vip_blackjack_bonus(delta=five_card_bonus, is_vip=is_vip)
    vip_bonus = max(casino_paid_vip_bonus, five_card_vip_delta - five_card_bonus)
    effective_delta = base_delta + vip_bonus + five_card_bonus
    result = await apply_blackjack_settlement(
        player_id=participant.user_id,
        player_account_name=participant.account_name,
        player_avatar_url=participant.avatar_url,
        player_delta=effective_delta,
        casino_delta=-casino_paid_delta,
    )
    return BlackjackPlayerSettlement(
        outcome=_aggregate_outcome(
            hand_settlements=hand_settlements, insurance=insurance, base_delta=base_delta
        ),
        delta=result.applied_player_delta,
        new_balance=result.player_balance,
        casino_balance=result.casino_balance,
        base_delta=base_delta,
        vip_bonus=vip_bonus,
        is_vip=is_vip,
        hands=hand_settlements,
        insurance=insurance,
        five_card_bonus=five_card_bonus,
    )
