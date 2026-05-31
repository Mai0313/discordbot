"""Markdown text formatter for the `/games blackjack_history` query.

Renders a player's recent Blackjack rounds as a monospace code-block table
inside an embed description instead of a PNG, so the command stays cheap (no
Pillow render, no file attachment) and the result is publicly shareable.
Discord markdown has no real tables, so columns are aligned with space padding
inside a ``` fenced block. A code block cannot carry color, so each round's
outcome is conveyed by a short ASCII tag plus the signed P&L. Player and dealer
hands are separated into their own columns rather than per-row labels.
"""

from typing import Final
from collections.abc import Sequence

from nextcord import Embed
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.games import SettleOutcome, BlackjackHistoryRecord, BlackjackHistoryPayload
from discordbot.cogs._games.presentation import WIN_COLOR, LOSE_COLOR, PUSH_COLOR

# Embed description hard limit is 4096; keep headroom for the title, summary
# line, code fences, and a possible truncation note.
_DESCRIPTION_BUDGET: Final[int] = 3800
_PLAYER_CELL_CAP: Final[int] = 26
_DEALER_CELL_CAP: Final[int] = 16

_RESULT_TAGS: Final[dict[SettleOutcome, str]] = {
    "win": "WIN",
    "lose": "LOSE",
    "push": "PUSH",
    "blackjack": "BJ",
    "five_card_win": "5CARD",
    "five_card_twenty_one": "5C21",
    "player_bust": "BUST",
    "dealer_bust": "WIN",
    "surrender": "SUR",
}


class _HistorySummary(BaseModel):
    """Aggregate win/loss/push counts and net delta over the rendered rounds."""

    model_config = ConfigDict(frozen=True)

    rounds: int = Field(description="Number of rounds included in the summary.")
    wins: int = Field(description="Rounds with a positive net delta.")
    losses: int = Field(description="Rounds with a negative net delta.")
    pushes: int = Field(description="Rounds with a zero net delta.")
    net_delta: int = Field(description="Sum of every round's net delta.")


class _Row(BaseModel):
    """One pre-formatted table row before column padding."""

    model_config = ConfigDict(frozen=True)

    when: str = Field(description="Round timestamp as MM/DD HH:MM.")
    player: str = Field(description="Player hand cell, possibly truncated.")
    dealer: str = Field(description="Dealer hand cell, possibly truncated.")
    bet: str = Field(description="Comma-formatted bet amount.")
    pnl: str = Field(description="Signed comma-formatted net delta.")
    tag: str = Field(description="Short ASCII outcome tag.")


def _summarize(records: Sequence[BlackjackHistoryRecord]) -> _HistorySummary:
    """Counts wins, losses, and pushes by net round delta."""
    wins = sum(1 for record in records if record.delta > 0)
    losses = sum(1 for record in records if record.delta < 0)
    pushes = sum(1 for record in records if record.delta == 0)
    net_delta = sum(record.delta for record in records)
    return _HistorySummary(
        rounds=len(records), wins=wins, losses=losses, pushes=pushes, net_delta=net_delta
    )


def _signed(value: int) -> str:
    """Formats a signed, comma-grouped amount; zero renders without a sign."""
    return f"{value:+,}" if value != 0 else "0"


def _truncate(text: str, width: int) -> str:
    """Clamps `text` to `width` characters with a trailing ellipsis."""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return f"{text[: width - 1]}…"


def _hand_cell(payload: BlackjackHistoryPayload) -> str:
    """Renders the player's hand(s); split hands are space-separated."""
    parts = [
        "".join(str(card) for card in hand.cards) + f"({hand.total})" for hand in payload.hands
    ]
    return " ".join(parts) if parts else "-"


def _dealer_cell(payload: BlackjackHistoryPayload) -> str:
    """Renders the dealer hand and final total."""
    cards = "".join(str(card) for card in payload.dealer_cards)
    return f"{cards}({payload.dealer_total})" if cards else "-"


def _build_rows(records: Sequence[BlackjackHistoryRecord]) -> list[_Row]:
    """Pre-formats every record into a padding-ready row."""
    return [
        _Row(
            when=record.created_at.strftime("%m/%d %H:%M"),
            player=_truncate(text=_hand_cell(payload=record.payload), width=_PLAYER_CELL_CAP),
            dealer=_truncate(text=_dealer_cell(payload=record.payload), width=_DEALER_CELL_CAP),
            bet=f"{record.bet:,}",
            pnl=_signed(value=record.delta),
            tag=_RESULT_TAGS.get(record.outcome, record.outcome.upper()),
        )
        for record in records
    ]


def _render_block(rows: Sequence[_Row]) -> str:
    """Aligns rows into a fenced monospace table block."""
    player_width = max(len(row.player) for row in rows)
    dealer_width = max(len(row.dealer) for row in rows)
    bet_width = max(len(row.bet) for row in rows)
    pnl_width = max(len(row.pnl) for row in rows)
    lines = [
        f"{row.when}  {row.player:<{player_width}}  {row.dealer:<{dealer_width}}  "
        f"{row.bet:>{bet_width}}  {row.pnl:>{pnl_width}} {row.tag}"
        for row in rows
    ]
    body = "\n".join(lines)
    return f"```\n{body}\n```"


def _net_color(net_delta: int) -> int:
    """Returns the embed accent color for the overall net result."""
    if net_delta > 0:
        return WIN_COLOR
    if net_delta < 0:
        return LOSE_COLOR
    return PUSH_COLOR


def build_blackjack_history_embed(
    *, player_name: str, records: Sequence[BlackjackHistoryRecord]
) -> Embed:
    """Builds the public embed for a player's recent Blackjack rounds."""
    title = f"🃏 {player_name} 的二十一點紀錄"
    if not records:
        return Embed(title=title, description="還沒有任何二十一點對局紀錄。", color=PUSH_COLOR)
    summary = _summarize(records=records)
    rows = _build_rows(records=records)
    omitted = 0
    while len(rows) > 1 and len(_render_block(rows=rows)) > _DESCRIPTION_BUDGET:
        rows = rows[:-1]
        omitted += 1
    summary_line = (
        f"近 {summary.rounds} 場 · "
        f"{summary.wins} 勝 {summary.losses} 敗 {summary.pushes} 和 · "
        f"淨損益 {_signed(value=summary.net_delta)}"
    )
    parts = [summary_line, _render_block(rows=rows)]
    if omitted:
        parts.append(f"-# 還有 {omitted} 場較舊紀錄未顯示")
    return Embed(title=title, description="\n".join(parts), color=_net_color(net_delta=summary.net_delta))
