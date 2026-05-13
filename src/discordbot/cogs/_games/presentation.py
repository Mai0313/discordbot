"""Shared presentation helpers for casino game embeds."""

from typing import Literal

from nextcord import Embed

from discordbot.typings.games import SettleOutcome
from discordbot.cogs._economy.presentation import currency_text

IN_PROGRESS_COLOR = 0x5865F2
WIN_COLOR = 0x57F287
LOSE_COLOR = 0xED4245
PUSH_COLOR = 0xFEE75C
ERROR_COLOR = 0xED4245

DEALER_FIELD_EMOJI = "🎩"
PLAYER_FIELD_EMOJI = "👤"
DEALER_TALK_FIELD_EMOJI = "💬"
LOBBY_PLAYERS_FIELD_EMOJI = "👥"
POT_FIELD_EMOJI = "💰"
TURN_FIELD_EMOJI = "🎯"
GATE_FIELD_EMOJI = "🚪"
BET_RANGE_FIELD_EMOJI = "🪙"
LAST_HAND_FIELD_EMOJI = "⏮️"
SCOREBOARD_FIELD_EMOJI = "📊"
FINISH_REASON_FIELD_EMOJI = "🏁"

OWNER_BADGE_EMOJI = "👑"
LOBBY_WAIT_EMOJI = "🎴"

WIN_RESULT_EMOJI = "🎉"
LOSE_RESULT_EMOJI = "😢"
BUST_RESULT_EMOJI = "💥"
DEALER_BUST_RESULT_EMOJI = "🎊"
PUSH_RESULT_EMOJI = "🤝"
NATURAL_RESULT_EMOJI = "✨"

PlayerStatusKind = Literal["blackjack", "bust", "active", "stand", "waiting"]


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


def allin_note(is_allin: bool) -> str:
    """Returns the shared suffix for auto all-in rounds.

    Args:
        is_allin: Whether the requested bet was clamped to the full balance.

    Returns:
        The suffix text, or an empty string for non all-in rounds.
    """
    return " | all-in" if is_allin else ""


def wager_footer(bet: int, balance_at_start: int, is_allin: bool, status: str) -> str:
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


def settlement_footer(delta: int, new_balance: int, is_allin: bool) -> str:
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


def dealer_quote(text: str) -> str:
    """Formats dealer banter as a compact quote block."""
    if not text:
        return ""
    return "> " + text.replace("\n", "\n> ")


def duel_lines(player_name: str, player_value: str, dealer_name: str, dealer_value: str) -> str:
    """Formats a two-sided game board as one embed field value."""
    return f"**{player_name}**\n{player_value}\n\n**{dealer_name}**\n{dealer_value}"


def card_line(cards_text: str) -> str:
    """Renders a hand string as an H1 line with doubled inter-card spacing.

    Single-space `A♠ K♥` becomes `# A♠  K♥` so each card breathes a bit
    more inside the heading. Empty strings are left alone so callers can
    short-circuit without producing a stray ``#``.

    Args:
        cards_text: Pre-rendered hand string (e.g. ``"A♠ K♥"`` or
            ``"🂠 K♥"``).

    Returns:
        Markdown-ready H1 line for placement in an embed field value.
    """
    if not cards_text:
        return ""
    spaced = cards_text.replace(" ", "  ")
    return f"# {spaced}"


def status_line(emoji: str, label: str, total: int | None = None) -> str:
    """Formats an H2 status line for a player or dealer hand.

    The total slot is optional so the same helper covers Blackjack
    `## ✋ 17 · stand` and result `## 🎉 你贏了 · 20 > 19`.

    Args:
        emoji: Emoji shown before the textual status.
        label: Status label or short result phrase.
        total: Optional numeric total to interleave between emoji and label.

    Returns:
        Markdown H2 line.
    """
    if total is None:
        return f"## {emoji} {label}"
    return f"## {emoji} {total} · {label}"


def small_status_line(emoji: str, label: str, total: int) -> str:
    """Formats an H3 status line for non-active waiting players."""
    return f"### {emoji} {total} · {label}"


def metadata_line(text: str) -> str:
    """Formats a `-#` small text metadata line."""
    return f"-# {text}"


def dealer_talk_field_value(text: str) -> str:
    """Formats dealer banter for placement inside its own embed field."""
    if not text:
        return "> ..."
    return "> " + text.replace("\n", "\n> ")


def build_dealer_talk_embed(
    dealer_line: str, dealer_name: str, dealer_avatar_url: str = ""
) -> Embed:
    """Builds a standalone embed dedicated to the dealer's quote.

    The author slot shows the dealer name and avatar in the top-left corner so
    readers immediately know who is speaking; the description carries the
    quoted line itself.
    """
    embed = Embed(description=dealer_talk_field_value(text=dealer_line), color=IN_PROGRESS_COLOR)
    if dealer_avatar_url:
        embed.set_author(name=dealer_name, icon_url=dealer_avatar_url)
    else:
        embed.set_author(name=dealer_name)
    return embed


def lobby_participant_line(
    index: int, display_name: str, bet: int | None = None, is_allin: bool = False
) -> str:
    """Renders one lobby participant row with optional bet metadata.

    Args:
        index: 1-based position in the join order.
        display_name: Player display name.
        bet: Optional bet amount to append as inline code.
        is_allin: Whether to mark the row with an ``all-in`` suffix.

    Returns:
        A single Markdown line for the lobby roster.
    """
    bet_suffix = ""
    if bet is not None:
        allin_suffix = " · all-in" if is_allin else ""
        bet_suffix = f" · 下注 `{bet:,}`{allin_suffix}"
    return f"**{index}. {display_name}**{bet_suffix}"


def settlement_metadata(delta: int, new_balance: int, is_allin: bool) -> str:
    """Renders the small-text settlement metadata line.

    Args:
        delta: Player net point change for the round.
        new_balance: Player balance after settlement.
        is_allin: Whether the wager consumed the full balance.

    Returns:
        ``-# 本局 +X · 餘額 Y`` style metadata, with an ``· all-in`` suffix
        when the round was all-in.
    """
    allin = " · all-in" if is_allin else ""
    return f"-# 本局 `{delta:+,}`{allin} · 餘額 `{new_balance:,}`"


def player_result_title(outcome: SettleOutcome, player_total: int, dealer_total: int) -> str:
    """Formats the H2 result line for one player at Blackjack settlement.

    Args:
        outcome: Player-facing Blackjack outcome label.
        player_total: Final player hand total.
        dealer_total: Final dealer hand total.

    Returns:
        Markdown H2 line such as ``## 🎉 你贏了 · 20 > 19``.
    """
    return f"## {player_result_inline(outcome=outcome, player_total=player_total, dealer_total=dealer_total)}"


def player_result_inline(outcome: SettleOutcome, player_total: int, dealer_total: int) -> str:
    """Single-line result label without heading prefix, for embed titles."""
    if outcome == "blackjack":
        return f"{NATURAL_RESULT_EMOJI} Blackjack · {player_total}"
    if outcome == "dealer_bust":
        return f"{DEALER_BUST_RESULT_EMOJI} 莊家爆牌, 你贏了 · {dealer_total}"
    if outcome == "player_bust":
        return f"{BUST_RESULT_EMOJI} 你爆牌了 · {player_total}"
    if outcome == "win":
        return f"{WIN_RESULT_EMOJI} 你贏了 · {player_total} > {dealer_total}"
    if outcome == "lose":
        return f"{LOSE_RESULT_EMOJI} 你輸了 · {player_total} < {dealer_total}"
    return f"平手 · {player_total} = {dealer_total}"
