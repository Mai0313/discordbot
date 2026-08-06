"""Markdown text formatter for the `/games blackjack_history` query.

Renders a player's recent Blackjack rounds as a monospace code-block table inside an embed
description instead of a PNG, so the command stays cheap (no Pillow render, no file attachment)
and the result is publicly shareable. It is a file of its own rather than part of `cog.py`
because all of it is layout arithmetic — per-column caps, padding widths, and the trim loop that
keeps the block inside Discord's description limit — while the command itself only fetches rows
and posts what comes back. The shared casino embed vocabulary (the win / lose / push colors)
still comes from `presentation.py`; only the table shape is specific to this one command.

Discord markdown has no real tables, so columns are aligned with space padding inside a ``` fenced
block. A code block cannot carry color, so each round's outcome is conveyed by a short ASCII tag
plus the signed P&L, and the embed accent color carries the overall net result. Player and dealer
hands are separated into their own columns rather than per-row labels, which keeps one round on
one line even after a Split.

Records arrive newest first from `fetch_recent_blackjack_rounds`, so the budget trim drops the
oldest rounds and says how many it dropped; the summary line above the table is measured over
everything that was fetched, trimmed rounds included.
"""

from typing import Final
from collections.abc import Sequence

from nextcord import Embed
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.games import SettleOutcome, BlackjackHistoryRecord, BlackjackHistoryPayload
from discordbot.cogs.games.presentation import WIN_COLOR, LOSE_COLOR, PUSH_COLOR

# Embed description hard limit is 4096; keep headroom for the title, summary
# line, code fences, and a possible truncation note.
_DESCRIPTION_BUDGET: Final[int] = 3800
# A Split puts two hands in the player cell, so it gets the wider cap of the two.
_PLAYER_CELL_CAP: Final[int] = 26
_DEALER_CELL_CAP: Final[int] = 16

# Tags short enough to keep the table narrow. `dealer_bust` shares "WIN" because it is the same
# result to the player, and the P&L column carries the amount either way.
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

    rounds: int = Field(..., description="Number of rounds included in the summary.")
    wins: int = Field(..., description="Rounds with a positive net delta.")
    losses: int = Field(..., description="Rounds with a negative net delta.")
    pushes: int = Field(..., description="Rounds with a zero net delta.")
    net_delta: int = Field(..., description="Sum of every round's net delta.")


class _Row(BaseModel):
    """One pre-formatted table row before column padding."""

    model_config = ConfigDict(frozen=True)

    when: str = Field(..., description="Round timestamp as MM/DD HH:MM.")
    player: str = Field(..., description="Player hand cell, possibly truncated.")
    dealer: str = Field(..., description="Dealer hand cell, possibly truncated.")
    bet: str = Field(..., description="Comma-formatted bet amount.")
    pnl: str = Field(..., description="Signed comma-formatted net delta.")
    tag: str = Field(..., description="Short ASCII outcome tag.")


def _summarize(records: Sequence[BlackjackHistoryRecord]) -> _HistorySummary:
    """Counts wins, losses, and pushes by net round delta.

    Classifies on the sign of `delta` rather than on `outcome`, because the summary line is
    about money: a round whose main leg pushed while a bonus paid belongs on the winning side,
    and the tag column already carries the outcome label itself.

    Args:
        records (Sequence[BlackjackHistoryRecord]): The rounds to aggregate.

    Returns:
        Round count, win / loss / push counts, and the summed net delta.
    """
    wins = sum(1 for record in records if record.delta > 0)
    losses = sum(1 for record in records if record.delta < 0)
    pushes = sum(1 for record in records if record.delta == 0)
    net_delta = sum(record.delta for record in records)
    return _HistorySummary(
        rounds=len(records), wins=wins, losses=losses, pushes=pushes, net_delta=net_delta
    )


def _signed(value: int) -> str:
    """Formats a signed, comma-grouped amount.

    Zero drops the sign: `+0` in a P&L column reads as a win that paid nothing.

    Args:
        value (int): The amount to render.

    Returns:
        The amount with an explicit sign and thousands separators, or a bare `0`.
    """
    return f"{value:+,}" if value != 0 else "0"


def _truncate(text: str, width: int) -> str:
    """Clamps `text` to `width` characters with a trailing ellipsis.

    The ellipsis spends one of the `width` characters, so a width of 1 or less is cut bare
    instead: the marker alone would otherwise overflow the column it is meant to fit.

    Args:
        text (str): The cell contents to clamp.
        width (int): Maximum characters the cell may occupy.

    Returns:
        `text` unchanged when it already fits, else the truncated cell.
    """
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return f"{text[: width - 1]}…"


def _hand_cell(payload: BlackjackHistoryPayload) -> str:
    """Renders the player's hand(s); split hands are space-separated.

    Falls back to `-` rather than an empty cell: every payload field carries a default, so a row
    persisted before hands were stored still validates and would otherwise leave a hole in the
    column.

    Args:
        payload (BlackjackHistoryPayload): The round snapshot to read hands from.

    Returns:
        Each hand as its cards followed by `(total)`, or `-` when the row carries none.
    """
    parts = [
        "".join(str(card) for card in hand.cards) + f"({hand.total})" for hand in payload.hands
    ]
    return " ".join(parts) if parts else "-"


def _dealer_cell(payload: BlackjackHistoryPayload) -> str:
    """Renders the dealer hand and final total.

    Same `-` fallback as the player cell, for the same legacy-row reason.

    Args:
        payload (BlackjackHistoryPayload): The round snapshot to read the dealer hand from.

    Returns:
        The dealer's cards followed by `(total)`, or `-` when the row carries none.
    """
    cards = "".join(str(card) for card in payload.dealer_cards)
    return f"{cards}({payload.dealer_total})" if cards else "-"


def _build_rows(records: Sequence[BlackjackHistoryRecord]) -> list[_Row]:
    """Pre-formats every record into a padding-ready row.

    Cells are clamped to their per-column caps here and padded later, since a column's width is
    only knowable once every row exists. An outcome missing from `_RESULT_TAGS` falls back to its
    own uppercased name, so a newly added `SettleOutcome` renders untagged instead of raising.

    Args:
        records (Sequence[BlackjackHistoryRecord]): The rounds to format, in display order.

    Returns:
        One row per record, each cell already a string but not yet padded.
    """
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
    """Aligns rows into a fenced monospace table block.

    Every column is sized to its own widest cell, so the block only spends the width it needs
    against the description budget. `rows` must be non-empty; the width scan is a `max()` over it
    and the caller is what guarantees at least one row.

    Args:
        rows (Sequence[_Row]): The pre-formatted rows to align, in display order.

    Returns:
        The whole table as one fenced code block, ready to sit in an embed description.
    """
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
    """Picks the embed accent color for the overall net result.

    The fenced table can carry no color of its own, so the accent is the only place the run of
    rounds shows as good or bad at a glance.

    Args:
        net_delta (int): Summed net delta over the rendered rounds.

    Returns:
        The shared win / lose / push embed color.
    """
    if net_delta > 0:
        return WIN_COLOR
    if net_delta < 0:
        return LOSE_COLOR
    return PUSH_COLOR


def build_blackjack_history_embed(
    *, player_name: str, records: Sequence[BlackjackHistoryRecord]
) -> Embed:
    """Builds the public embed for a player's recent Blackjack rounds.

    Rows are rendered in full and then dropped from the oldest end one at a time until the block
    fits `_DESCRIPTION_BUDGET`; only the block is measured, and the budget's headroom is what pays
    for the title, the summary line and the omission note. The last row always survives, so a
    table too wide to fit is posted rather than replaced by an empty block. The summary counts
    every record fetched, trimmed ones included, so it describes the query while the `-#` note
    says how much of it the table could hold. An empty history returns a plain notice with no
    table at all.

    Args:
        player_name (str): Display name to title the embed with.
        records (Sequence[BlackjackHistoryRecord]): The player's rounds, newest first.

    Returns:
        The embed to post publicly, accented by the net result over `records`.
    """
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
    return Embed(
        title=title, description="\n".join(parts), color=_net_color(net_delta=summary.net_delta)
    )
