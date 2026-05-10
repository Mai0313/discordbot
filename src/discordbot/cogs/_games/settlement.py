"""Settlement helpers shared by game commands and interactive views."""

from dataclasses import dataclass

from discordbot.cogs._games.blackjack import (
    OutcomeLabel,
    BlackjackHand,
    settle,
    is_bust,
    is_blackjack,
)
from discordbot.cogs._economy.database import settle_game, house_settle


@dataclass(frozen=True)
class BlackjackSettlement:
    """Database-backed settlement result for a finished Blackjack round.

    Attributes:
        outcome: Player-facing outcome label from the Blackjack rules engine.
        delta: Net point change relative to the withdrawn bet.
        payout: Gross amount credited back to the player after the upfront bet withdrawal.
        new_balance: Player balance after crediting the payout.
        house_balance: Dealer ledger balance after mirroring the player's net change.
        detail: Short game-state summary for the dealer AI prompt.
    """

    outcome: OutcomeLabel
    delta: int
    payout: int
    new_balance: int
    house_balance: int
    detail: str


def blackjack_payout(*, bet: int, delta: int) -> int:
    """Returns the gross payout after an upfront bet withdrawal."""
    return max(bet + delta, 0)


def blackjack_detail(hand: BlackjackHand) -> str:
    """Builds a concise Blackjack result summary for dealer banter."""
    player_blackjack = is_blackjack(cards=hand.player)
    dealer_blackjack = is_blackjack(cards=hand.dealer)
    if player_blackjack and dealer_blackjack:
        return "雙方都是 Blackjack, 平手"
    if player_blackjack:
        return f"玩家 21 點 Blackjack, 莊家 {hand.dealer_total()} 點"
    if dealer_blackjack:
        return f"莊家 21 點 Blackjack, 玩家 {hand.player_total()} 點"
    if is_bust(cards=hand.player):
        return f"玩家爆牌 {hand.player_total()} 點"
    if is_bust(cards=hand.dealer):
        return f"莊家爆牌 {hand.dealer_total()} 點, 玩家 {hand.player_total()} 點"
    return f"玩家 {hand.player_total()} 點 vs 莊家 {hand.dealer_total()} 點"


def blackjack_early_finish_note(hand: BlackjackHand) -> str | None:
    """Explains why a hand ended before the player could hit or stand."""
    player_blackjack = is_blackjack(cards=hand.player)
    dealer_blackjack = is_blackjack(cards=hand.dealer)
    if player_blackjack and dealer_blackjack:
        return "雙方起手 Blackjack, 本局直接平手。"
    if player_blackjack:
        return "你起手 Blackjack, 本局直接結算。"
    if dealer_blackjack:
        return "莊家起手 Blackjack, 依規則本局直接結算。"
    return None


async def settle_blackjack_round(
    *,
    hand: BlackjackHand,
    player_id: int,
    player_account_name: str,
    dealer_id: int,
    dealer_name: str,
) -> BlackjackSettlement:
    """Settles player payout and mirrored house ledger for one finished hand."""
    outcome, delta = settle(hand=hand)
    payout = blackjack_payout(bet=hand.bet, delta=delta)
    new_balance = await settle_game(user_id=player_id, name=player_account_name, delta=payout)
    house_balance = await house_settle(user_id=dealer_id, name=dealer_name, delta=-delta)
    return BlackjackSettlement(
        outcome=outcome,
        delta=delta,
        payout=payout,
        new_balance=new_balance,
        house_balance=house_balance,
        detail=blackjack_detail(hand=hand),
    )
