"""Settlement helpers shared by game commands and interactive views."""

from discordbot.typings.games import (
    Card,
    SettleOutcome,
    WagerSettlement,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
    BlackjackInsuranceSettlement,
)
from discordbot.cogs._games.blackjack import (
    BlackjackRound,
    BlackjackHandState,
    BlackjackPlayerHand,
    is_bust,
    hand_value,
    settle_hand,
    is_blackjack,
    dealer_up_card,
    is_five_card_win,
    is_five_card_twenty_one,
)
from discordbot.cogs._economy.database import (
    get_vip,
    apply_round_settlement,
    apply_vip_blackjack_bonus,
    apply_blackjack_settlement,
)


def _blackjack_detail_for_hand(cards: list[Card], dealer: list[Card]) -> str:
    """Builds a concise Blackjack result summary for dealer banter.

    Args:
        cards: Player sub-hand cards.
        dealer: Dealer cards at settlement time.

    Returns:
        A short Chinese summary of the final hand state.
    """
    player_total = hand_value(cards=cards)
    dealer_total = hand_value(cards=dealer)
    player_blackjack = is_blackjack(cards=cards)
    dealer_blackjack = is_blackjack(cards=dealer)
    if player_blackjack and dealer_blackjack:
        detail = "雙方都是 Blackjack, 平手"
    elif player_blackjack:
        detail = f"玩家 21 點 Blackjack, 莊家 {dealer_total} 點"
    elif is_five_card_twenty_one(cards=cards):
        if dealer_total == 21:
            detail = "玩家過五關 21 點, 莊家 21 點, 主局平手"
        else:
            detail = f"玩家過五關 21 點, 莊家 {dealer_total} 點"
    elif is_five_card_win(cards=cards):
        detail = f"玩家過五關 {player_total} 點, 未爆直接獲勝"
    elif dealer_blackjack:
        detail = f"莊家 21 點 Blackjack, 玩家 {player_total} 點"
    elif is_bust(cards=cards):
        detail = f"玩家爆牌 {player_total} 點"
    elif is_bust(cards=dealer):
        detail = f"莊家爆牌 {dealer_total} 點, 玩家 {player_total} 點"
    else:
        detail = f"玩家 {player_total} 點 vs 莊家 {dealer_total} 點"
    return detail


def _blackjack_hand_detail_part(
    index: int, settlement: BlackjackHandSettlement, dealer_total: int
) -> str:
    """Formats one settled Blackjack sub-hand for dealer banter detail."""
    hand_total = hand_value(cards=settlement.cards)
    prefix = f"手{index}"
    if settlement.surrendered:
        detail = f"{prefix} 投降 (-{abs(settlement.delta)})"
    elif settlement.five_card_twenty_one:
        bonus = f", 過五關 bonus +{settlement.five_card_bonus}"
        if settlement.delta == 0:
            detail = f"{prefix} 過五關 21 (主局平手{bonus})"
        else:
            detail = f"{prefix} 過五關 21 ({settlement.delta:+d}{bonus})"
    elif settlement.outcome == "five_card_win":
        detail = f"{prefix} 過五關 {hand_total} ({settlement.delta:+d})"
    elif settlement.outcome == "blackjack":
        detail = f"{prefix} Blackjack ({settlement.delta:+d})"
    elif settlement.outcome == "player_bust":
        detail = f"{prefix} 爆牌 {hand_total} ({settlement.delta:+d})"
    elif settlement.outcome == "dealer_bust":
        detail = f"{prefix} {hand_total} 莊家爆牌 ({settlement.delta:+d})"
    elif settlement.outcome == "push":
        detail = f"{prefix} {hand_total} 平手"
    else:
        detail = f"{prefix} {hand_total} vs 莊家 {dealer_total} ({settlement.delta:+d})"
    return detail


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
        Short Chinese summary. Single-hand players keep the concise detail
        shape used by dealer banter.
    """
    if len(hand_settlements) == 1 and insurance is None:
        only = hand_settlements[0]
        return _blackjack_detail_for_hand(cards=list(only.cards), dealer=dealer)
    dealer_total = hand_value(cards=dealer)
    hand_parts: list[str] = []
    for index, settlement in enumerate(hand_settlements, start=1):
        hand_parts.append(
            _blackjack_hand_detail_part(
                index=index, settlement=settlement, dealer_total=dealer_total
            )
        )
    summary = "; ".join(hand_parts)
    if insurance is not None:
        if insurance.won:
            summary += f"; 保險 {insurance.bet} → 中獎 (+{insurance.delta})"
        else:
            summary += f"; 保險 {insurance.bet} → 莊家無 BJ ({insurance.delta:+d})"
    return summary


def blackjack_player_early_finish_note(  # noqa: PLR0911 -- one branch per early-finish reason keeps the mapping explicit
    player: BlackjackPlayerHand, dealer: list[Card], peeked_blackjack: bool
) -> str | None:
    """Returns a short explanation for round paths that skipped player actions.

    Args:
        player: Player to inspect.
        dealer: Dealer cards at settlement time.
        peeked_blackjack: Whether the dealer revealed a Blackjack via peek.

    Returns:
        The explanation text, or `None` when no early-finish path applies.
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
        return f"{_dealer_peek_note(dealer=dealer)}, 你也起手 Blackjack, 本局直接平手"
    if peeked_blackjack:
        return f"{_dealer_peek_note(dealer=dealer)}, 本局直接結算"
    if dealer_bj and player_bj:
        return "雙方起手 Blackjack, 本局直接平手"
    if player_bj:
        return "你起手 Blackjack, 本局直接結算"
    if dealer_bj:
        return "莊家起手 Blackjack, 依規則本局直接結算"
    return None


def _dealer_peek_note(dealer: list[Card]) -> str:
    """Returns the reason text for dealer Blackjack revealed by a hole-card peek."""
    up = dealer_up_card(dealer=dealer)
    if up is None:
        return "莊家 peek 暗牌確認 Blackjack"
    return f"莊家明牌 {up}, peek 暗牌確認 Blackjack"


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

    The aggregate dealer-paid delta (sum of per-hand deltas plus insurance)
    is passed through the existing VIP bonus rule once at the player level.
    Five-card 21 adds a system-funded bonus to the player-side delta without
    moving the house ledger. Both player and dealer rows are still written
    through one DB transaction.

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
        _hand_settlement_from_state(hand=hand, dealer=round_state.dealer) for hand in player.hands
    ]
    insurance = _insurance_settlement(player=player, peeked_blackjack=round_state.peeked_blackjack)
    base_delta = sum(settlement.delta for settlement in hand_settlements)
    if insurance is not None:
        base_delta += insurance.delta
    five_card_bonus = sum(settlement.five_card_bonus for settlement in hand_settlements)

    is_vip = await get_vip(user_id=player_id)
    dealer_paid_delta = apply_vip_blackjack_bonus(delta=base_delta, is_vip=is_vip)
    dealer_paid_vip_bonus = dealer_paid_delta - base_delta
    five_card_vip_delta = apply_vip_blackjack_bonus(delta=five_card_bonus, is_vip=is_vip)
    vip_bonus = max(dealer_paid_vip_bonus, five_card_vip_delta - five_card_bonus)
    effective_delta = base_delta + vip_bonus + five_card_bonus
    new_balance, house_balance = await apply_blackjack_settlement(
        player_id=player_id,
        player_account_name=player_account_name,
        player_avatar_url=player_avatar_url,
        player_delta=effective_delta,
        dealer_id=dealer_id,
        dealer_name=dealer_name,
        dealer_avatar_url=dealer_avatar_url,
        dealer_delta=-dealer_paid_delta,
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
        five_card_bonus=five_card_bonus,
    )
