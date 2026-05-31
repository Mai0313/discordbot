"""Pillow board renderer for the `/games blackjack_history` query.

Renders a player's recent Blackjack rounds as a fixed-column PNG table so the
hands, dealer hands, bets, and results stay aligned and readable beyond
Discord's embed limits. The player's own hands and the dealer hand are split by
column headers rather than per-row labels. This module stays presentation-only:
it consumes already-typed `BlackjackHistoryRecord`s and reuses the shared
`pil_text` font and anchoring helpers.
"""

from io import BytesIO
from typing import Final, TypedDict
from functools import cache
from collections.abc import Sequence

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from discordbot.utils.pil_text import Font, fit_text, load_font, draw_text_right
from discordbot.typings.games import SettleOutcome, BlackjackHistoryRecord, BlackjackHistoryPayload
from discordbot.utils.number_text import compact_amount

BLACKJACK_HISTORY_BOARD_FILENAME = "blackjack_history.png"

_BOARD_WIDTH = 1180
_BOARD_MARGIN = 30
_BOARD_HEADER_HEIGHT = 84
_TABLE_HEADER_HEIGHT = 40
_ROW_HEIGHT = 52
_BOARD_FOOTER_HEIGHT = 26

_BACKGROUND = (28, 31, 36)
_SURFACE = (38, 42, 49)
_ROW_ALT = (33, 37, 43)
_GRID = (70, 76, 88)
_TEXT = (234, 237, 242)
_MUTED = (169, 177, 190)
_ACCENT = (88, 101, 242)
_GAIN = (87, 242, 135)
_LOSS = (237, 66, 69)
_PUSH = (254, 231, 92)

_TIME_X: Final[int] = 50
_PLAYER_X: Final[int] = 178
_PLAYER_MAX_WIDTH: Final[int] = 330
_DEALER_X: Final[int] = 528
_DEALER_MAX_WIDTH: Final[int] = 230
_RESULT_X: Final[int] = 778
_BET_RIGHT: Final[int] = 1010
_DELTA_RIGHT: Final[int] = 1150

_OUTCOME_LABELS: Final[dict[SettleOutcome, str]] = {
    "win": "贏",
    "lose": "輸",
    "push": "和",
    "blackjack": "BJ",
    "five_card_win": "過五關",
    "five_card_twenty_one": "過五關21",
    "player_bust": "爆牌",
    "dealer_bust": "贏",
    "surrender": "投降",
}


class _BoardFonts(TypedDict):
    """Font set used by the Blackjack history board image."""

    title: Font
    summary: Font
    header: Font
    body: Font
    cards: Font
    small: Font


class _HistorySummary(BaseModel):
    """Aggregate win/loss/push counts and net delta over the rendered rounds."""

    model_config = ConfigDict(frozen=True)

    rounds: int
    wins: int
    losses: int
    pushes: int
    net_delta: int


def _summarize(records: Sequence[BlackjackHistoryRecord]) -> _HistorySummary:
    """Counts wins, losses, and pushes by net round delta."""
    wins = sum(1 for record in records if record.delta > 0)
    losses = sum(1 for record in records if record.delta < 0)
    pushes = sum(1 for record in records if record.delta == 0)
    net_delta = sum(record.delta for record in records)
    return _HistorySummary(
        rounds=len(records), wins=wins, losses=losses, pushes=pushes, net_delta=net_delta
    )


def _delta_color(delta: int) -> tuple[int, int, int]:
    """Returns the player-centric color for a signed round delta."""
    if delta > 0:
        return _GAIN
    if delta < 0:
        return _LOSS
    return _PUSH


def _player_hands_text(payload: BlackjackHistoryPayload) -> str:
    """Joins a player's hands (two after a Split) into one cell string."""
    parts = [
        f"{' '.join(str(card) for card in hand.cards)} ({hand.total})" for hand in payload.hands
    ]
    return "  ·  ".join(parts) if parts else "—"


def _dealer_text(payload: BlackjackHistoryPayload) -> str:
    """Formats the dealer hand and final total for one round."""
    cards = " ".join(str(card) for card in payload.dealer_cards)
    return f"{cards} ({payload.dealer_total})" if cards else "—"


@cache
def _history_board_fonts() -> _BoardFonts:
    """Loads CJK-capable fonts for the history board."""
    return {
        "title": load_font(size=32, bold=True),
        "summary": load_font(size=20, bold=False),
        "header": load_font(size=18, bold=True),
        "body": load_font(size=21, bold=False),
        "cards": load_font(size=22, bold=False),
        "small": load_font(size=15, bold=False),
    }


def build_blackjack_history_board(
    *, player_name: str, records: Sequence[BlackjackHistoryRecord]
) -> bytes:
    """Renders a player's recent Blackjack rounds as a PNG table."""
    summary = _summarize(records=records)
    row_count = max(len(records), 1)
    height = (
        _BOARD_MARGIN * 2
        + _BOARD_HEADER_HEIGHT
        + _TABLE_HEADER_HEIGHT
        + row_count * _ROW_HEIGHT
        + _BOARD_FOOTER_HEIGHT
    )
    image = Image.new(mode="RGB", size=(_BOARD_WIDTH, height), color=_BACKGROUND)
    draw = ImageDraw.Draw(im=image)
    fonts = _history_board_fonts()
    _draw_header(draw=draw, fonts=fonts, player_name=player_name, summary=summary)
    table_top = _BOARD_MARGIN + _BOARD_HEADER_HEIGHT
    _draw_table_header(draw=draw, fonts=fonts, y=table_top)
    for index, record in enumerate(iterable=records):
        y = table_top + _TABLE_HEADER_HEIGHT + index * _ROW_HEIGHT
        _draw_round_row(draw=draw, fonts=fonts, record=record, position=index + 1, y=y)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _draw_header(
    draw: ImageDraw.ImageDraw, fonts: _BoardFonts, player_name: str, summary: _HistorySummary
) -> None:
    """Draws the board title and aggregate summary line."""
    x = _BOARD_MARGIN
    y = _BOARD_MARGIN
    title = fit_text(
        draw=draw,
        text=f"{player_name} 的二十一點紀錄",
        font=fonts["title"],
        max_width=_BOARD_WIDTH - _BOARD_MARGIN * 2,
    )
    draw.text(xy=(x, y), text=title, font=fonts["title"], fill=_TEXT)
    record_summary = (
        f"近 {summary.rounds} 場 · "
        f"{summary.wins} 勝 {summary.losses} 敗 {summary.pushes} 和"
    )
    draw.text(xy=(x, y + 44), text=record_summary, font=fonts["summary"], fill=_MUTED)
    draw_text_right(
        draw=draw,
        text=f"淨損益 {compact_amount(amount=summary.net_delta, signed=True)}",
        xy=(_BOARD_WIDTH - _BOARD_MARGIN, y + 44),
        font=fonts["summary"],
        fill=_delta_color(delta=summary.net_delta),
    )


def _draw_table_header(draw: ImageDraw.ImageDraw, fonts: _BoardFonts, y: int) -> None:
    """Draws the column headers."""
    draw.rectangle(
        xy=(_BOARD_MARGIN, y, _BOARD_WIDTH - _BOARD_MARGIN, y + _TABLE_HEADER_HEIGHT),
        fill=_SURFACE,
    )
    baseline = y + 11
    draw.text(xy=(_TIME_X, baseline), text="時間", font=fonts["header"], fill=_MUTED)
    draw.text(xy=(_PLAYER_X, baseline), text="玩家手牌", font=fonts["header"], fill=_MUTED)
    draw.text(xy=(_DEALER_X, baseline), text="莊家手牌", font=fonts["header"], fill=_MUTED)
    draw.text(xy=(_RESULT_X, baseline), text="結果", font=fonts["header"], fill=_MUTED)
    draw_text_right(
        draw=draw, text="下注", xy=(_BET_RIGHT, baseline), font=fonts["header"], fill=_MUTED
    )
    draw_text_right(
        draw=draw, text="損益", xy=(_DELTA_RIGHT, baseline), font=fonts["header"], fill=_MUTED
    )


def _draw_round_row(
    draw: ImageDraw.ImageDraw,
    fonts: _BoardFonts,
    record: BlackjackHistoryRecord,
    position: int,
    y: int,
) -> None:
    """Draws one round row."""
    fill = _SURFACE if position % 2 == 1 else _ROW_ALT
    draw.rectangle(xy=(_BOARD_MARGIN, y, _BOARD_WIDTH - _BOARD_MARGIN, y + _ROW_HEIGHT), fill=fill)
    draw.line(
        xy=(_BOARD_MARGIN, y + _ROW_HEIGHT, _BOARD_WIDTH - _BOARD_MARGIN, y + _ROW_HEIGHT),
        fill=_GRID,
        width=1,
    )
    text_y = y + 13
    draw.text(
        xy=(_TIME_X, text_y),
        text=record.created_at.strftime("%m/%d %H:%M"),
        font=fonts["small"],
        fill=_MUTED,
    )
    player_text = fit_text(
        draw=draw,
        text=_player_hands_text(payload=record.payload),
        font=fonts["cards"],
        max_width=_PLAYER_MAX_WIDTH,
    )
    draw.text(xy=(_PLAYER_X, text_y), text=player_text, font=fonts["cards"], fill=_TEXT)
    dealer_text = fit_text(
        draw=draw,
        text=_dealer_text(payload=record.payload),
        font=fonts["cards"],
        max_width=_DEALER_MAX_WIDTH,
    )
    draw.text(xy=(_DEALER_X, text_y), text=dealer_text, font=fonts["cards"], fill=_MUTED)
    color = _delta_color(delta=record.delta)
    draw.text(
        xy=(_RESULT_X, text_y),
        text=_OUTCOME_LABELS.get(record.outcome, record.outcome),
        font=fonts["body"],
        fill=color,
    )
    draw_text_right(
        draw=draw,
        text=compact_amount(amount=record.bet),
        xy=(_BET_RIGHT, text_y),
        font=fonts["body"],
        fill=_TEXT,
    )
    draw_text_right(
        draw=draw,
        text=compact_amount(amount=record.delta, signed=True),
        xy=(_DELTA_RIGHT, text_y),
        font=fonts["body"],
        fill=color,
    )
