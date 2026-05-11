"""Settlement helpers shared by game commands and interactive views."""

from dataclasses import dataclass

from discordbot.cogs._games.blackjack import (
    OutcomeLabel,
    BlackjackHand,
    settle,
    is_bust,
    is_blackjack,
)
from discordbot.cogs._economy.database import apply_round_settlement


@dataclass(frozen=True)
class WagerSettlement:
    """Database-backed settlement result for a finished wager.

    Attributes:
        delta: Net point change relative to the withdrawn bet.
        payout: Gross amount credited back to the player after the upfront bet withdrawal.
        new_balance: Player balance after crediting the payout.
        house_balance: Dealer ledger balance after mirroring the player's net change.
    """

    delta: int
    payout: int
    new_balance: int
    house_balance: int


@dataclass(frozen=True)
class BlackjackSettlement(WagerSettlement):
    """Database-backed settlement result for a finished Blackjack round.

    Attributes:
        outcome: Player-facing outcome label from the Blackjack rules engine.
        detail: Short game-state summary for the dealer AI prompt.
    """

    outcome: OutcomeLabel
    detail: str


def wager_payout(*, bet: int, delta: int) -> int:
    """Returns the gross payout after an upfront bet withdrawal.

    Args:
        bet: Effective bet amount that was already withdrawn.
        delta: Player net point change for the round.

    Returns:
        Amount to credit back to the player, clamped to at least zero.
    """
    return max(bet + delta, 0)


def blackjack_detail(hand: BlackjackHand) -> str:
    """Builds a concise Blackjack result summary for dealer banter.

    Args:
        hand: Blackjack hand to summarize.

    Returns:
        A short Chinese summary of the final hand state.
    """
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
    """Explains why a hand ended before the player could hit or stand.

    Args:
        hand: Blackjack hand to inspect.

    Returns:
        The early-finish explanation, or `None` when no natural Blackjack
        ended the hand.
    """
    player_blackjack = is_blackjack(cards=hand.player)
    dealer_blackjack = is_blackjack(cards=hand.dealer)
    if player_blackjack and dealer_blackjack:
        return "雙方起手 Blackjack, 本局直接平手。"
    if player_blackjack:
        return "你起手 Blackjack, 本局直接結算。"
    if dealer_blackjack:
        return "莊家起手 Blackjack, 依規則本局直接結算。"
    return None


async def settle_wager(  # noqa: PLR0913 -- settlement needs both player and dealer ledger keys
    *,
    player_id: int,
    player_account_name: str,
    dealer_id: int,
    dealer_name: str,
    bet: int,
    delta: int,
) -> WagerSettlement:
    """Credits player payout and mirrors the net result into the house ledger.

    The two ledger writes share a single SQLite transaction via
    `apply_round_settlement`, so a crash between them cannot leave the dealer
    ledger drifting from the player payout.

    Args:
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        dealer_id: Discord user ID for the dealer ledger row.
        dealer_name: Account name to store for the dealer ledger row.
        bet: Effective bet amount that was already withdrawn.
        delta: Player net point change for the round.

    Returns:
        Database-backed settlement result after both ledgers are updated.
    """
    payout = wager_payout(bet=bet, delta=delta)
    new_balance, house_balance = await apply_round_settlement(
        player_id=player_id,
        player_account_name=player_account_name,
        payout=payout,
        dealer_id=dealer_id,
        dealer_name=dealer_name,
        dealer_delta=-delta,
    )
    return WagerSettlement(
        delta=delta, payout=payout, new_balance=new_balance, house_balance=house_balance
    )


async def settle_blackjack_round(
    *,
    hand: BlackjackHand,
    player_id: int,
    player_account_name: str,
    dealer_id: int,
    dealer_name: str,
) -> BlackjackSettlement:
    """Settles player payout and mirrored house ledger for one finished hand.

    Args:
        hand: Finished Blackjack hand to settle.
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        dealer_id: Discord user ID for the dealer ledger row.
        dealer_name: Account name to store for the dealer ledger row.

    Returns:
        Blackjack settlement result including outcome and dealer prompt detail.

    Raises:
        ValueError: The hand is not finished yet.
    """
    outcome, delta = settle(hand=hand)
    wager = await settle_wager(
        player_id=player_id,
        player_account_name=player_account_name,
        dealer_id=dealer_id,
        dealer_name=dealer_name,
        bet=hand.bet,
        delta=delta,
    )
    return BlackjackSettlement(
        outcome=outcome,
        delta=wager.delta,
        payout=wager.payout,
        new_balance=wager.new_balance,
        house_balance=wager.house_balance,
        detail=blackjack_detail(hand=hand),
    )
