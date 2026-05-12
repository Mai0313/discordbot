"""Shared presentation helpers for casino game embeds."""

from discordbot.typings.games import SettleOutcome
from discordbot.cogs._economy.presentation import currency_text

IN_PROGRESS_COLOR = 0x5865F2
WIN_COLOR = 0x57F287
LOSE_COLOR = 0xED4245
PUSH_COLOR = 0xFEE75C
ERROR_COLOR = 0xED4245


def blackjack_outcome_presentation(outcome: SettleOutcome) -> tuple[str, int]:
    """Returns presentation values for a Blackjack outcome.

    Args:
        outcome: Player-facing Blackjack outcome.

    Returns:
        A `(label, color)` tuple for the final embed.
    """
    blackjack_result = {
        "win": ("你贏了", WIN_COLOR),
        "lose": ("你輸了", LOSE_COLOR),
        "push": ("平手", PUSH_COLOR),
        "blackjack": ("Blackjack!", WIN_COLOR),
        "player_bust": ("你爆牌了", LOSE_COLOR),
        "dealer_bust": ("莊家爆牌, 你贏了", WIN_COLOR),
    }
    return blackjack_result[outcome]


def allin_note(*, is_allin: bool) -> str:
    """Returns the shared suffix for auto all-in rounds.

    Args:
        is_allin: Whether the requested bet was clamped to the full balance.

    Returns:
        The suffix text, or an empty string for non all-in rounds.
    """
    return " | all-in" if is_allin else ""


def wager_footer(*, bet: int, balance_at_start: int, is_allin: bool, status: str) -> str:
    """Formats the shared footer for an unresolved round.

    Args:
        bet: Effective wager for the round.
        balance_at_start: Player balance observed when the round started.
        is_allin: Whether the requested bet was clamped to the full balance.
        status: Short status text for the round.

    Returns:
        Footer text for an in-progress game embed.
    """
    return (
        f"下注 {currency_text(amount=bet)} | "
        f"目前餘額 {currency_text(amount=balance_at_start)} | {status}"
        f"{allin_note(is_allin=is_allin)}"
    )


def settlement_footer(*, delta: int, new_balance: int, is_allin: bool) -> str:
    """Formats the shared final-round settlement footer.

    Keeps only the two numbers the player needs to see at a glance: the
    round delta and the post-settlement balance. ``/house`` carries
    the dealer ledger when the player explicitly asks for it.

    Args:
        delta: Player net point change for the round.
        new_balance: Player balance after settlement.
        is_allin: Whether the requested bet was clamped to the full balance.

    Returns:
        Footer text for a final game embed.
    """
    return (
        f"本局 {currency_text(amount=delta, signed=True)} | "
        f"餘額 {currency_text(amount=new_balance)}{allin_note(is_allin=is_allin)}"
    )


def dealer_quote(*, text: str) -> str:
    """Formats dealer banter as a compact quote block."""
    if not text:
        return ""
    return "> " + text.replace("\n", "\n> ")


def duel_lines(*, player_name: str, player_value: str, dealer_name: str, dealer_value: str) -> str:
    """Formats a two-sided game board as one embed field value."""
    return f"**{player_name}**\n{player_value}\n\n**{dealer_name}**\n{dealer_value}"
