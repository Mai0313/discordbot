"""Settlement helpers shared by game commands and interactive views."""

from random import Random

from discordbot.typings.games import (
    Card,
    SettleOutcome,
    WagerSettlement,
    BlackjackSettlement,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
    BlackjackInsuranceSettlement,
)
from discordbot.cogs._games.blackjack import (
    BlackjackHand,
    BlackjackRound,
    BlackjackHandState,
    BlackjackPlayerHand,
    settle,
    is_bust,
    hand_value,
    settle_hand,
    is_blackjack,
)
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


def blackjack_detail_player(
    player: BlackjackPlayerHand,
    dealer: list[Card],
    hand_settlements: list[BlackjackHandSettlement],
    insurance: BlackjackInsuranceSettlement | None,
) -> str:
    """Builds a multi-hand Blackjack summary for dealer banter.

    Args:
        player: Player whose hands are being summarized.
        dealer: Dealer cards at settlement time.
        hand_settlements: Per-hand results in display order.
        insurance: Optional insurance side-bet result.

    Returns:
        Short Chinese summary; for single-hand players this is shaped to
        match the legacy ``blackjack_detail`` text.
    """
    if len(hand_settlements) == 1 and insurance is None:
        only = hand_settlements[0]
        wrapped = BlackjackHand(
            rng=Random(x=0),  # noqa: S311 -- value never drives randomness; only fills the required field
            bet=only.bet,
            player=list(only.cards),
            dealer=list(dealer),
            finished=True,
        )
        return blackjack_detail(hand=wrapped)
    dealer_total = hand_value(cards=dealer)
    hand_parts: list[str] = []
    for index, settlement in enumerate(hand_settlements, start=1):
        hand_total = hand_value(cards=settlement.cards)
        prefix = f"手{index}"
        if settlement.surrendered:
            hand_parts.append(f"{prefix} 投降 (-{abs(settlement.delta)})")
        elif settlement.outcome == "blackjack":
            hand_parts.append(f"{prefix} Blackjack ({settlement.delta:+d})")
        elif settlement.outcome == "player_bust":
            hand_parts.append(f"{prefix} 爆牌 {hand_total} ({settlement.delta:+d})")
        elif settlement.outcome == "dealer_bust":
            hand_parts.append(f"{prefix} {hand_total} 莊家爆牌 ({settlement.delta:+d})")
        elif settlement.outcome == "push":
            hand_parts.append(f"{prefix} {hand_total} 平手")
        else:
            hand_parts.append(
                f"{prefix} {hand_total} vs 莊家 {dealer_total} ({settlement.delta:+d})"
            )
    summary = "; ".join(hand_parts)
    if insurance is not None:
        if insurance.won:
            summary += f"; 保險 {insurance.bet} → 中獎 (+{insurance.delta})"
        else:
            summary += f"; 保險 {insurance.bet} → 莊家無 BJ ({insurance.delta:+d})"
    return summary


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


def blackjack_player_early_finish_note(  # noqa: PLR0911 -- one branch per early-finish reason keeps the mapping explicit
    player: BlackjackPlayerHand, dealer: list[Card], peeked_blackjack: bool
) -> str | None:
    """Returns a short explanation for round paths that skipped player actions.

    Args:
        player: Player to inspect.
        dealer: Dealer cards at settlement time.
        peeked_blackjack: Whether the dealer revealed a Blackjack via peek.

    Returns:
        The explanation text, or ``None`` when no early-finish path applies.
    """
    dealer_bj = is_blackjack(cards=dealer)
    if not player.hands:
        return None
    first_hand = player.hands[0]
    player_bj = (
        len(player.hands) == 1
        and not first_hand.is_split_hand
        and is_blackjack(cards=first_hand.cards)
    )
    if peeked_blackjack and player_bj:
        return "莊家 peek 確認 Blackjack, 你也起手 Blackjack, 本局直接平手"
    if peeked_blackjack:
        return "莊家 peek 確認 Blackjack, 本局直接結算"
    if dealer_bj and player_bj:
        return "雙方起手 Blackjack, 本局直接平手"
    if player_bj:
        return "你起手 Blackjack, 本局直接結算"
    if dealer_bj:
        return "莊家起手 Blackjack, 依規則本局直接結算"
    return None


async def settle_wager(  # noqa: PLR0913 -- settlement needs both player and dealer ledger keys
    player_id: int,
    player_account_name: str,
    dealer_id: int,
    dealer_name: str,
    delta: int,
    player_avatar_url: str = "",
    dealer_avatar_url: str = "",
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
        delta: Player net point change for the round.

    Returns:
        Database-backed settlement result after both ledgers are updated.
    """
    is_vip = await get_vip(user_id=player_id)
    effective_delta = apply_vip_blackjack_bonus(delta=delta, is_vip=is_vip)
    vip_bonus = effective_delta - delta
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
        base_delta=delta,
        vip_bonus=vip_bonus,
        is_vip=is_vip,
    )


async def settle_blackjack_round(  # noqa: PLR0913 -- settlement needs both player and dealer ledger keys
    hand: BlackjackHand,
    player_id: int,
    player_account_name: str,
    dealer_id: int,
    dealer_name: str,
    player_avatar_url: str = "",
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
        delta=delta,
    )
    return BlackjackSettlement(
        outcome=outcome,
        delta=wager.delta,
        payout=wager.payout,
        new_balance=wager.new_balance,
        house_balance=wager.house_balance,
        base_delta=wager.base_delta,
        vip_bonus=wager.vip_bonus,
        is_vip=wager.is_vip,
        detail=blackjack_detail(hand=hand),
    )


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
    hand: BlackjackHandState, dealer: list[Card], rng: Random
) -> BlackjackHandSettlement:
    """Wraps `settle_hand` into a `BlackjackHandSettlement` row."""
    outcome, delta = settle_hand(hand=hand, dealer=dealer, rng=rng)
    return BlackjackHandSettlement(
        cards=list(hand.cards),
        bet=hand.bet,
        outcome=outcome,
        delta=delta,
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


async def settle_blackjack_player(  # noqa: PLR0913 -- settlement needs every ledger key
    *,
    round_state: BlackjackRound,
    player: BlackjackPlayerHand,
    player_id: int,
    player_account_name: str,
    dealer_id: int,
    dealer_name: str,
    player_avatar_url: str = "",
    dealer_avatar_url: str = "",
) -> BlackjackPlayerSettlement:
    """Settles every sub-hand plus insurance side bet for one participant.

    The aggregate delta (sum of per-hand deltas plus insurance) is passed
    through the existing VIP bonus rule once at the player level, then
    written through a single `apply_round_settlement` call so the player
    and house ledgers always move together.

    Args:
        round_state: Round providing the dealer cards, RNG, and peek state.
        player: Player to settle.
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        player_avatar_url: Last-seen Discord avatar URL for the player.
        dealer_id: Discord user ID for the dealer ledger row.
        dealer_name: Account name to store for the dealer ledger row.
        dealer_avatar_url: Last-seen Discord avatar URL for the dealer.

    Returns:
        Aggregated settlement covering every sub-hand and any insurance bet.
    """
    hand_settlements = [
        _hand_settlement_from_state(hand=hand, dealer=round_state.dealer, rng=round_state.rng)
        for hand in player.hands
    ]
    insurance = _insurance_settlement(player=player, peeked_blackjack=round_state.peeked_blackjack)
    base_delta = sum(settlement.delta for settlement in hand_settlements)
    if insurance is not None:
        base_delta += insurance.delta

    is_vip = await get_vip(user_id=player_id)
    effective_delta = apply_vip_blackjack_bonus(delta=base_delta, is_vip=is_vip)
    vip_bonus = effective_delta - base_delta
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
    return BlackjackPlayerSettlement(
        outcome=_aggregate_outcome(
            hand_settlements=hand_settlements, insurance=insurance, base_delta=base_delta
        ),
        detail=blackjack_detail_player(
            player=player,
            dealer=round_state.dealer,
            hand_settlements=hand_settlements,
            insurance=insurance,
        ),
        delta=effective_delta,
        payout=max(effective_delta, 0),
        new_balance=new_balance,
        house_balance=house_balance,
        base_delta=base_delta,
        vip_bonus=vip_bonus,
        is_vip=is_vip,
        hands=hand_settlements,
        insurance=insurance,
    )
