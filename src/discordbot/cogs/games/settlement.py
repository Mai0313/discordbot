"""The money half of the Blackjack table: every point a `/games blackjack` round moves lands here.

`blackjack.py` is the pure rules engine and stops at the dealer-paid delta; `blackjack_views.py`
only renders what comes back. This file is the seam between them, and it owns the three things
that belong to neither:

- The system-funded 過五關 21 bonus, minted at 1x the hand's bet. It credits the player and is
  deliberately withheld from the casino ledger, so a five-card 21 never moves `/casino`.
- The VIP 1.2x payout, applied once per seat rather than once per hand. What is credited is the
  LARGER of the 0.2x on the dealer-paid win and the 0.2x on the five-card bonus, never their sum.
- Aggregation. A seat that split holds two hands plus a possible insurance side bet, and all of it
  goes out in a single `apply_blackjack_settlement` call, so one participant moves money once.

A wager is never debited when the round starts, so that one write is the whole of what a round
costs; an unfinished in-memory round vanishing on a restart leaves no balance to unwind.

`settle_wager` is the plain single-delta path for a game that settles one number per player. It
predates the Blackjack aggregate and no production caller is left: 射龍門 settles against the
shared `jackpot_pool` row through its own helpers, and Blackjack goes through
`settle_blackjack_player`.

The detail builders are a leftover kept alive by `BlackjackPlayerSettlement.detail`: they still
assemble the Chinese round summary, but nothing renders or persists it now that the casino has no
narrator and the bot player runs on no LLM. `blackjack_player_early_finish_note` is the one string
built here that is still shown, by the seat embeds in `blackjack_views.py`.
"""

from discordbot.typings.games import (
    Card,
    SettleOutcome,
    WagerSettlement,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
    BlackjackInsuranceSettlement,
)
from discordbot.cogs.games.blackjack import (
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
from discordbot.services.economy.database import (
    get_vip,
    apply_round_settlement,
    apply_vip_blackjack_bonus,
    apply_blackjack_settlement,
)


def _blackjack_detail_for_hand(cards: list[Card], dealer: list[Card]) -> str:
    """Builds the concise summary used when a seat holds exactly one hand and no insurance.

    Only that path reaches it, which is what lets it read the raw cards: `is_blackjack` cannot
    tell a split-derived two-card 21 from a natural, and a split seat always carries two
    settlements and takes the multi-hand shape instead.

    Args:
        cards (list[Card]): Player sub-hand cards.
        dealer (list[Card]): Dealer cards at settlement time.

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
    """Formats one settled sub-hand as a fragment of the multi-hand summary.

    The five-card 21 branch splits on `delta` because that hand's main leg pushes against a
    dealer 21 while the bonus is still paid, so a `+0` main result would misread as nothing won.

    Args:
        index (int): 1-based position of this hand in display order.
        settlement (BlackjackHandSettlement): Settled hand to describe.
        dealer_total (int): Dealer's final total, printed only on the plain comparison branch.

    Returns:
        One `手N ...` fragment.
    """
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
    """Builds the round summary stored on `BlackjackPlayerSettlement.detail`.

    Nothing renders or persists that field today; see the module docstring. A single hand with
    no insurance keeps the concise one-hand shape, everything else joins one fragment per hand
    and appends the insurance line.

    Args:
        player (BlackjackPlayerHand): Player being summarized; currently unread by this
            function.
        dealer (list[Card]): Dealer cards at settlement time.
        hand_settlements (list[BlackjackHandSettlement]): Per-hand results in display order.
        insurance (BlackjackInsuranceSettlement | None): Insurance side-bet result, or `None`
            when the player never took insurance.

    Returns:
        A short Chinese summary of the whole seat.
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

    Rendered under the seat's settlement line, so a hand that never got a Hit button does not
    read as a bug. The natural-Blackjack test deliberately requires the seat's only hand to be
    unsplit, matching `BlackjackHandState.is_blackjack`: a split half that reaches 21 in two
    cards is not a natural and does not end the round.

    Args:
        player (BlackjackPlayerHand): Player to inspect.
        dealer (list[Card]): Dealer cards at settlement time.
        peeked_blackjack (bool): Whether the dealer revealed a Blackjack via peek.

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
    """Returns the reason text for a dealer Blackjack revealed by a hole-card peek.

    Args:
        dealer (list[Card]): Dealer cards at settlement time.

    Returns:
        The reason text, naming the visible up-card unless the dealer holds no cards at all.
    """
    up = dealer_up_card(dealer=dealer)
    if up is None:
        return "莊家 peek 暗牌確認 Blackjack"
    return f"莊家明牌 {up}, peek 暗牌確認 Blackjack"


async def settle_wager(
    player_id: int, player_account_name: str, delta: int, player_avatar_url: str = ""
) -> WagerSettlement:
    """Applies one player net delta and mirrors the whole of it into the casino ledger.

    The single-delta path, for a game that settles one number per seat. No production caller is
    left; Blackjack aggregates through `settle_blackjack_player` instead.

    VIP players receive a 1.2x payout on winning rounds; pushes and losses are passed through
    unchanged. The VIP flag is permanent, so reading it outside the settlement transaction is
    safe — a freshly-bought VIP that races a settlement only misses the bonus on a single
    in-flight round.

    Args:
        player_id (int): Discord user ID for the player account.
        player_account_name (str): Account name to store for the player.
        delta (int): Player net point change for the round, before the VIP bonus.
        player_avatar_url (str): Last-seen Discord avatar URL for the player.

    Returns:
        Database-backed settlement result after both ledgers are updated.
    """
    is_vip = await get_vip(user_id=player_id)
    effective_delta = apply_vip_blackjack_bonus(delta=delta, is_vip=is_vip)
    vip_bonus = effective_delta - delta
    result = await apply_round_settlement(
        player_id=player_id,
        player_account_name=player_account_name,
        player_avatar_url=player_avatar_url,
        player_delta=effective_delta,
        casino_delta=-effective_delta,
    )
    return WagerSettlement(
        delta=effective_delta,
        payout=max(effective_delta, 0),
        new_balance=result.player_balance,
        casino_balance=result.casino_balance,
        base_delta=delta,
        vip_bonus=vip_bonus,
        is_vip=is_vip,
    )


def _aggregate_outcome(
    hand_settlements: list[BlackjackHandSettlement],
    insurance: BlackjackInsuranceSettlement | None,
    base_delta: int,
) -> SettleOutcome:
    """Collapses a settled seat into the one outcome label the round history stores.

    A lone hand with no insurance keeps its own label, so `blackjack` / `five_card_win` /
    `surrender` survive. Anything else is decided on `base_delta`, which is the dealer-paid
    side only: a system-funded five-card bonus therefore never relabels a losing seat as a win.

    Args:
        hand_settlements (list[BlackjackHandSettlement]): Per-hand results in display order.
        insurance (BlackjackInsuranceSettlement | None): Insurance side-bet result, or `None`
            when the player never took insurance.
        base_delta (int): Dealer-paid net delta for the seat, insurance included.

    Returns:
        The label stored on `BlackjackPlayerSettlement.outcome`.
    """
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
    """Settles one sub-hand and mints its five-card 21 bonus.

    `settle_hand` returns the dealer-paid delta alone, so the system-funded 過五關 21 bonus is
    added here at 1x the hand's bet and kept in its own field, where the casino ledger cannot
    reach it.

    Args:
        hand (BlackjackHandState): Finished sub-hand to settle.
        dealer (list[Card]): Dealer cards at settlement time.

    Returns:
        The settled row for this hand.
    """
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
    """Resolves the insurance side bet, if the seat took one.

    Reads the round's recorded peek instead of re-testing the dealer cards: insurance is only
    offered on an Ace up-card, and the peek at the close of that phase is what settled the bet
    before anyone acted. A win pays 2:1, so the delta is twice the bet.

    Args:
        player (BlackjackPlayerHand): Player whose insurance bet to resolve.
        peeked_blackjack (bool): Whether the hole-card peek revealed a Blackjack.

    Returns:
        The insurance row, or `None` when the player never took insurance.
    """
    if player.insurance_bet <= 0:
        return None
    bet = player.insurance_bet
    if peeked_blackjack:
        return BlackjackInsuranceSettlement(bet=bet, won=True, delta=bet * 2)
    return BlackjackInsuranceSettlement(bet=bet, won=False, delta=-bet)


async def settle_blackjack_player(
    *,
    round_state: BlackjackRound,
    player: BlackjackPlayerHand,
    player_id: int,
    player_account_name: str,
    player_avatar_url: str = "",
) -> BlackjackPlayerSettlement:
    """Settles every sub-hand plus the insurance side bet for one participant, in one write.

    The aggregate casino-paid delta (sum of per-hand deltas plus insurance) is passed through
    the VIP bonus rule once at the player level, not per hand. Five-card 21 adds a
    system-funded bonus to the player-side delta without moving the casino ledger, and the VIP
    bonus credited is the larger of the 0.2x on the dealer-paid win and the 0.2x on the
    five-card 21 bonus (a max, not a sum). Everything the player receives beyond
    `casino_paid_delta` is therefore minted by the system, which is why the two deltas handed
    to `apply_blackjack_settlement` deliberately disagree.

    Args:
        round_state (BlackjackRound): Round supplying the dealer cards and the peek result.
        player (BlackjackPlayerHand): Player to settle, hands and insurance bet included.
        player_id (int): Discord user ID for the player account.
        player_account_name (str): Account name to store for the player.
        player_avatar_url (str): Last-seen Discord avatar URL for the player.

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
    casino_paid_delta = apply_vip_blackjack_bonus(delta=base_delta, is_vip=is_vip)
    casino_paid_vip_bonus = casino_paid_delta - base_delta
    five_card_vip_delta = apply_vip_blackjack_bonus(delta=five_card_bonus, is_vip=is_vip)
    vip_bonus = max(casino_paid_vip_bonus, five_card_vip_delta - five_card_bonus)
    effective_delta = base_delta + vip_bonus + five_card_bonus
    result = await apply_blackjack_settlement(
        player_id=player_id,
        player_account_name=player_account_name,
        player_avatar_url=player_avatar_url,
        player_delta=effective_delta,
        casino_delta=-casino_paid_delta,
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
        new_balance=result.player_balance,
        casino_balance=result.casino_balance,
        base_delta=base_delta,
        vip_bonus=vip_bonus,
        is_vip=is_vip,
        hands=hand_settlements,
        insurance=insurance,
        five_card_bonus=five_card_bonus,
    )
