"""Shared presentation helpers for casino game embeds."""

from typing import Final

from discordbot.typings.games import Card, SettleOutcome
from discordbot.typings.colors import DISCORD_RED, NEUTRAL_BLUE, DISCORD_GREEN, DISCORD_YELLOW
from discordbot.cogs.games.blackjack import BlackjackPlayerHand, dealer_up_card
from discordbot.services.economy.presentation import amount_code

WIN_COLOR = DISCORD_GREEN
LOSE_COLOR = DISCORD_RED
PUSH_COLOR = DISCORD_YELLOW
ERROR_COLOR = DISCORD_RED
IN_PROGRESS_COLOR = NEUTRAL_BLUE

SYSTEM_NARRATOR_NAME: Final[str] = "賭場系統"

LOBBY_PLAYERS_FIELD_EMOJI = "👥"
POT_FIELD_EMOJI = "💰"
TURN_FIELD_EMOJI = "🎯"
LAST_HAND_FIELD_EMOJI = "⏮️"
FINISH_REASON_FIELD_EMOJI = "🏁"

WIN_RESULT_EMOJI = "🎉"
LOSE_RESULT_EMOJI = "😢"
BUST_RESULT_EMOJI = "💥"
DEALER_BUST_RESULT_EMOJI = "🎊"
NATURAL_RESULT_EMOJI = "✨"


def delta_color(delta: int) -> int:
    """Returns the win, lose, or push color for a signed point change.

    Args:
        delta (int): Signed point change.
    """
    if delta > 0:
        return WIN_COLOR
    if delta < 0:
        return LOSE_COLOR
    return PUSH_COLOR


def card_line(cards_text: str) -> str:
    """Returns `cards_text` as an H1 line with doubled inter-card spacing, or `""` when empty.

    Args:
        cards_text (str): Pre-rendered hand string (e.g. `"A♠ K♥"` or `"🂠 K♥"`).
    """
    if not cards_text:
        return ""
    spaced = cards_text.replace(" ", "  ")
    return f"# {spaced}"


def render_hand(cards: list[Card], hide_first: bool = False) -> str:
    """Returns a space-separated display string for `cards`.

    Args:
        cards (list[Card]): Cards to render.
        hide_first (bool): Whether to replace the first card with a hidden-card marker.
    """
    if hide_first and cards:
        rest = " ".join(str(card) for card in cards[1:])
        return f"🂠 {rest}".strip()
    return " ".join(str(card) for card in cards)


def metadata_line(text: str) -> str:
    """Returns `text` formatted as a `-#` small-text metadata line.

    Args:
        text (str): Metadata text to wrap.
    """
    return f"-# {text}"


def lobby_participant_line(
    index: int, display_name: str, bet: int | None = None, is_allin: bool = False
) -> str:
    """Returns one Markdown lobby roster line with optional bet metadata.

    Args:
        index (int): 1-based position in the join order.
        display_name (str): Player display name.
        bet (int | None): Optional bet amount to append as inline code.
        is_allin (bool): Whether to mark the row with an `all-in` suffix.
    """
    bet_suffix = ""
    if bet is not None:
        allin_suffix = " · all-in" if is_allin else ""
        bet_suffix = f" · 下注 {amount_code(amount=bet, compact=True)}{allin_suffix}"
    return f"**{index}. {display_name}**{bet_suffix}"


def settlement_metadata(
    delta: int, new_balance: int, is_allin: bool, vip_bonus: int = 0, five_card_bonus: int = 0
) -> str:
    """Returns the `-#` small-text settlement metadata line.

    Args:
        delta (int): Player net point change for the round.
        new_balance (int): Player balance after settlement.
        is_allin (bool): Whether the wager consumed the full balance.
        vip_bonus (int): Extra points added by the VIP payout bonus.
        five_card_bonus (int): System-funded bonus from five-card 21.
    """
    segments = [f"本局 {amount_code(amount=delta, signed=True, compact=True)}"]
    if vip_bonus > 0:
        segments.append(f"VIP加成 {amount_code(amount=vip_bonus, signed=True, compact=True)}")
    if five_card_bonus > 0:
        segments.append(
            f"過五關 bonus {amount_code(amount=five_card_bonus, signed=True, compact=True)}"
        )
    if is_allin:
        segments.append("all-in")
    segments.append(f"餘額 {amount_code(amount=new_balance, compact=True)}")
    return "-# " + " · ".join(segments)


def player_result_title(outcome: SettleOutcome, player_total: int, dealer_total: int) -> str:  # noqa: PLR0911 -- one branch per SettleOutcome label keeps the mapping obvious
    """Returns the Markdown H2 result line for one sub-hand at Blackjack settlement.

    Args:
        outcome (SettleOutcome): Player-facing Blackjack outcome label.
        player_total (int): Final player hand total.
        dealer_total (int): Final dealer hand total.
    """
    if outcome == "blackjack":
        return f"## {NATURAL_RESULT_EMOJI} Blackjack · {player_total}"
    if outcome == "five_card_twenty_one":
        return f"## {NATURAL_RESULT_EMOJI} 過五關 · {player_total}"
    if outcome == "five_card_win":
        return f"## {WIN_RESULT_EMOJI} 過五關 · {player_total}"
    if outcome == "dealer_bust":
        return f"## {DEALER_BUST_RESULT_EMOJI} 莊家爆牌, 你贏了 · {dealer_total}"
    if outcome == "player_bust":
        return f"## {BUST_RESULT_EMOJI} 你爆牌了 · {player_total}"
    if outcome == "win":
        return f"## {WIN_RESULT_EMOJI} 你贏了 · {player_total} > {dealer_total}"
    if outcome == "lose":
        return f"## {LOSE_RESULT_EMOJI} 你輸了 · {player_total} < {dealer_total}"
    if outcome == "surrender":
        return "## 🏳️ 投降 · 退一半"
    return f"## 平手 · {player_total} = {dealer_total}"


def blackjack_player_early_finish_note(
    player: BlackjackPlayerHand, dealer: list[Card], peeked_blackjack: bool
) -> str | None:
    """Returns a short explanation for round paths that skipped player actions, or None.

    Args:
        player (BlackjackPlayerHand): Player to inspect.
        dealer (list[Card]): Dealer cards at settlement time.
        peeked_blackjack (bool): Whether the dealer revealed a Blackjack via peek.
    """
    if not player.hands:
        return None
    first_hand = player.hands[0]
    player_bj = len(player.hands) == 1 and first_hand.is_blackjack()
    if peeked_blackjack and player_bj:
        return f"{_dealer_peek_note(dealer=dealer)}, 你也起手 Blackjack, 本局直接平手"
    if peeked_blackjack:
        return f"{_dealer_peek_note(dealer=dealer)}, 本局直接結算"
    if player_bj:
        return "你起手 Blackjack, 本局直接結算"
    return None


def _dealer_peek_note(dealer: list[Card]) -> str:
    """Returns the reason text for dealer Blackjack revealed by a hole-card peek."""
    return f"莊家明牌 {dealer_up_card(dealer=dealer)}, peek 暗牌確認 Blackjack"
