"""Settlement helpers shared by game commands and interactive views."""

from discordbot.typings.games import WagerSettlement, BlackjackSettlement
from discordbot.cogs._games.blackjack import BlackjackHand, settle, is_bust, is_blackjack
from discordbot.cogs._economy.database import (
    get_vip,
    apply_round_settlement,
    apply_vip_blackjack_bonus,
)


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
        return "雙方起手 Blackjack, 本局直接平手"
    if player_blackjack:
        return "你起手 Blackjack, 本局直接結算"
    if dealer_blackjack:
        return "莊家起手 Blackjack, 依規則本局直接結算"
    return None


async def settle_wager(  # noqa: PLR0913 -- settlement needs both player and dealer ledger keys
    *,
    player_id: int,
    player_account_name: str,
    player_avatar_url: str = "",
    dealer_id: int,
    dealer_name: str,
    dealer_avatar_url: str = "",
    bet: int,
    delta: int,
) -> WagerSettlement:
    """Applies player net delta and mirrors the result into the house ledger.

    The two ledger writes share a single SQLite transaction via
    `apply_round_settlement`, so a crash between them cannot leave the dealer
    ledger drifting from the player result. Bets are not deducted when a round
    starts; unfinished in-memory rounds vanish on bot restart without touching
    balances.

    VIP players receive a 1.5x payout on winning rounds; pushes and losses
    are passed through unchanged. The VIP flag is permanent, so reading it
    outside the settlement transaction is safe — a freshly-bought VIP that
    races a settlement only misses the bonus on a single in-flight round.

    Args:
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        player_avatar_url: Last-seen Discord avatar URL for the player.
        dealer_id: Discord user ID for the dealer ledger row.
        dealer_name: Account name to store for the dealer ledger row.
        dealer_avatar_url: Last-seen Discord avatar URL for the dealer.
        bet: Effective bet amount for the finished round.
        delta: Player net point change for the round.

    Returns:
        Database-backed settlement result after both ledgers are updated.
    """
    is_vip = await get_vip(user_id=player_id)
    effective_delta = apply_vip_blackjack_bonus(delta=delta, is_vip=is_vip)
    new_balance, house_balance = await apply_round_settlement(
        player_id=player_id,
        player_account_name=player_account_name,
        player_avatar_url=player_avatar_url,
        player_delta=effective_delta,
        dealer_id=dealer_id,
        dealer_name=dealer_name,
        dealer_avatar_url=dealer_avatar_url,
        dealer_delta=-effective_delta,
    )
    return WagerSettlement(
        delta=effective_delta,
        payout=max(effective_delta, 0),
        new_balance=new_balance,
        house_balance=house_balance,
    )


async def settle_blackjack_round(  # noqa: PLR0913 -- settlement needs both player and dealer ledger keys
    *,
    hand: BlackjackHand,
    player_id: int,
    player_account_name: str,
    player_avatar_url: str = "",
    dealer_id: int,
    dealer_name: str,
    dealer_avatar_url: str = "",
) -> BlackjackSettlement:
    """Settles player payout and mirrored house ledger for one finished hand.

    Args:
        hand: Finished Blackjack hand to settle.
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        player_avatar_url: Last-seen Discord avatar URL for the player.
        dealer_id: Discord user ID for the dealer ledger row.
        dealer_name: Account name to store for the dealer ledger row.
        dealer_avatar_url: Last-seen Discord avatar URL for the dealer.

    Returns:
        Blackjack settlement result including outcome and dealer prompt detail.

    Raises:
        ValueError: The hand is not finished yet.
    """
    outcome, delta = settle(hand=hand)
    wager = await settle_wager(
        player_id=player_id,
        player_account_name=player_account_name,
        player_avatar_url=player_avatar_url,
        dealer_id=dealer_id,
        dealer_name=dealer_name,
        dealer_avatar_url=dealer_avatar_url,
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


async def settle_dragon_gate_player(  # noqa: PLR0913 -- settlement needs both player and dealer ledger keys
    *,
    player_id: int,
    player_account_name: str,
    player_avatar_url: str = "",
    dealer_id: int,
    dealer_name: str,
    dealer_avatar_url: str = "",
    delta: int,
) -> WagerSettlement:
    """Settles one player's cumulative 射龍門 table delta.

    射龍門 follows its own pot odds and does not apply the Blackjack-only VIP
    payout multiplier. Positive deltas still use the generic casino payout
    path so loan auto-repayment behaves consistently with other casino income.

    Args:
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        player_avatar_url: Last-seen Discord avatar URL for the player.
        dealer_id: Discord user ID for the dealer ledger row.
        dealer_name: Account name to store for the dealer ledger row.
        dealer_avatar_url: Last-seen Discord avatar URL for the dealer.
        delta: Cumulative player net change for the full 射龍門 table.

    Returns:
        Database-backed settlement result after both ledgers are updated.
    """
    new_balance, house_balance = await apply_round_settlement(
        player_id=player_id,
        player_account_name=player_account_name,
        player_avatar_url=player_avatar_url,
        player_delta=delta,
        dealer_id=dealer_id,
        dealer_name=dealer_name,
        dealer_avatar_url=dealer_avatar_url,
        dealer_delta=-delta,
    )
    return WagerSettlement(
        delta=delta, payout=max(delta, 0), new_balance=new_balance, house_balance=house_balance
    )
