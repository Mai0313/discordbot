"""Pillow board renderers for public economy rankings."""

from io import BytesIO
from time import monotonic
from typing import Final
from functools import cache
from collections.abc import Sequence

from PIL import Image, ImageDraw
from pydantic import Field, BaseModel, ConfigDict

from discordbot.utils.pil_text import Font, fit_text, load_font, draw_text_right, draw_text_center
from discordbot.typings.economy import LeaderboardEntry, LossLeaderboardEntry
from discordbot.utils.number_text import compact_amount
from discordbot.cogs._economy.presentation import CURRENCY_NAME

BALANCE_LEADERBOARD_BOARD_FILENAME = "economy_leaderboard.png"
LOSS_LEADERBOARD_BOARD_FILENAME = "economy_loss_leaderboard.png"
_BOARD_WIDTH = 960
_BOARD_MARGIN = 30
_BOARD_HEADER_HEIGHT = 70
_TABLE_HEADER_HEIGHT = 42
_ROW_HEIGHT = 54
_BOARD_FOOTER_HEIGHT = 26
_BACKGROUND = (28, 31, 36)
_SURFACE = (38, 42, 49)
_ROW_ALT = (33, 37, 43)
_GRID = (70, 76, 88)
_TEXT = (234, 237, 242)
_MUTED = (169, 177, 190)
_BALANCE_ACCENT = (254, 231, 92)
_LOSS_ACCENT = (230, 126, 34)
_RANK_X = 52
_NAME_X = 128
_AMOUNT_RIGHT = 908
_NAME_MAX_WIDTH = 520
_BOARD_IMAGE_CACHE_TTL_SECONDS: Final[float] = 5.0


class _BoardFonts(BaseModel):
    """Font set used by economy board images."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: Font = Field(..., description="Board title font.")
    header: Font = Field(..., description="Table header font.")
    rank: Font = Field(..., description="Ranking number font.")
    body: Font = Field(..., description="Row text font.")
    small: Font = Field(..., description="Subtitle and badge font.")


class _RankingBoardSpec(BaseModel):
    """Data needed to render one ranking board."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Board title text.")
    subtitle: str = Field(..., description="Subtitle line under the title.")
    amount_header: str = Field(..., description="Amount column header text.")
    amount_label: str = Field(
        ..., description="Label prefixed to each amount cell, empty for none."
    )
    accent: tuple[int, int, int] = Field(
        ..., description="Accent RGB color for headers and medals."
    )
    rows: tuple[tuple[str, int], ...] = Field(
        ..., description="Ranked (name, amount) rows to render."
    )


class _RankingRow(BaseModel):
    """One row in a rendered ranking board."""

    model_config = ConfigDict(frozen=True)

    position: int = Field(..., description="One-based ranking position.")
    name: str = Field(..., description="Player display name.")
    amount: int = Field(..., description="Amount shown in the row.")


_board_image_cache: dict[_RankingBoardSpec, tuple[float, bytes]] = {}


def invalidate_economy_board_cache() -> None:
    """Clears process-local rendered economy board images."""
    _board_image_cache.clear()


def build_balance_leaderboard_board_image(rows: Sequence[LeaderboardEntry]) -> bytes:
    """Renders the public balance leaderboard as a PNG board."""
    return _build_ranking_board_image(
        spec=_RankingBoardSpec(
            title=f"{CURRENCY_NAME} 排行榜",
            subtitle="Top 10 public balances",
            amount_header="餘額",
            amount_label="",
            accent=_BALANCE_ACCENT,
            rows=tuple((row.name, row.balance) for row in rows),
        )
    )


def build_loss_leaderboard_board_image(rows: Sequence[LossLeaderboardEntry]) -> bytes:
    """Renders the public daily loss leaderboard as a PNG board."""
    return _build_ranking_board_image(
        spec=_RankingBoardSpec(
            title="今日輸錢榜",
            subtitle="Gross casino loss · Asia/Taipei 00:00 reset",
            amount_header="累計輸",
            amount_label="",
            accent=_LOSS_ACCENT,
            rows=tuple((row.name, row.loss_amount) for row in rows),
        )
    )


def _build_ranking_board_image(spec: _RankingBoardSpec) -> bytes:
    """Returns a cached rendered ranking board image."""
    now = monotonic()
    cached = _board_image_cache.get(spec)
    if cached is not None:
        cached_at, image = cached
        if now - cached_at <= _BOARD_IMAGE_CACHE_TTL_SECONDS:
            return image
    image = _render_ranking_board_image(spec=spec)
    _board_image_cache[spec] = (now, image)
    return image


def _render_ranking_board_image(spec: _RankingBoardSpec) -> bytes:
    """Renders a fixed-column ranking board."""
    rows = spec.rows
    row_count = max(len(rows), 1)
    height = (
        _BOARD_MARGIN * 2
        + _BOARD_HEADER_HEIGHT
        + _TABLE_HEADER_HEIGHT
        + row_count * _ROW_HEIGHT
        + _BOARD_FOOTER_HEIGHT
    )
    image = Image.new(mode="RGB", size=(_BOARD_WIDTH, height), color=_BACKGROUND)
    draw = ImageDraw.Draw(im=image)
    fonts = _board_fonts()
    _draw_header(
        draw=draw, fonts=fonts, title=spec.title, subtitle=spec.subtitle, accent=spec.accent
    )
    table_top = _BOARD_MARGIN + _BOARD_HEADER_HEIGHT
    _draw_table_header(
        draw=draw, fonts=fonts, y=table_top, amount_header=spec.amount_header, accent=spec.accent
    )
    if rows:
        for index, (name, amount) in enumerate(iterable=rows):
            y = table_top + _TABLE_HEADER_HEIGHT + index * _ROW_HEIGHT
            _draw_rank_row(
                draw=draw,
                fonts=fonts,
                row=_RankingRow(position=index + 1, name=name, amount=amount),
                spec=spec,
                y=y,
            )
    else:
        _draw_empty_row(draw=draw, fonts=fonts, y=table_top + _TABLE_HEADER_HEIGHT)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@cache
def _board_fonts() -> _BoardFonts:
    """Loads CJK-capable fonts for ranking boards."""
    return _BoardFonts(
        title=load_font(size=34, bold=True),
        header=load_font(size=19, bold=True),
        rank=load_font(size=26, bold=True),
        body=load_font(size=24, bold=False),
        small=load_font(size=16, bold=False),
    )


def _draw_header(
    draw: ImageDraw.ImageDraw,
    fonts: _BoardFonts,
    title: str,
    subtitle: str,
    accent: tuple[int, int, int],
) -> None:
    """Draws the board title area."""
    x = _BOARD_MARGIN
    y = _BOARD_MARGIN
    draw.text(xy=(x, y), text=title, font=fonts.title, fill=_TEXT)
    draw.text(xy=(x, y + 42), text=subtitle, font=fonts.small, fill=_MUTED)
    draw.rounded_rectangle(
        xy=(_BOARD_WIDTH - 162, y + 14, _BOARD_WIDTH - _BOARD_MARGIN, y + 48),
        radius=10,
        fill=(48, 52, 58),
        outline=accent,
        width=2,
    )
    draw_text_center(
        draw=draw, text="PUBLIC", center=(_BOARD_WIDTH - 96, y + 23), font=fonts.small, fill=accent
    )


def _draw_table_header(
    draw: ImageDraw.ImageDraw,
    fonts: _BoardFonts,
    y: int,
    amount_header: str,
    accent: tuple[int, int, int],
) -> None:
    """Draws table headers."""
    draw.rectangle(
        xy=(_BOARD_MARGIN, y, _BOARD_WIDTH - _BOARD_MARGIN, y + _TABLE_HEADER_HEIGHT),
        fill=_SURFACE,
    )
    baseline = y + 12
    draw.text(xy=(_RANK_X, baseline), text="排名", font=fonts.header, fill=_MUTED)
    draw.text(xy=(_NAME_X, baseline), text="玩家", font=fonts.header, fill=_MUTED)
    draw_text_right(
        draw=draw, text=amount_header, xy=(_AMOUNT_RIGHT, baseline), font=fonts.header, fill=accent
    )


def _draw_rank_row(
    draw: ImageDraw.ImageDraw,
    fonts: _BoardFonts,
    row: _RankingRow,
    spec: _RankingBoardSpec,
    y: int,
) -> None:
    """Draws one ranking row."""
    position = row.position
    fill = _SURFACE if position % 2 == 1 else _ROW_ALT
    draw.rectangle(xy=(_BOARD_MARGIN, y, _BOARD_WIDTH - _BOARD_MARGIN, y + _ROW_HEIGHT), fill=fill)
    draw.line(
        xy=(_BOARD_MARGIN, y + _ROW_HEIGHT, _BOARD_WIDTH - _BOARD_MARGIN, y + _ROW_HEIGHT),
        fill=_GRID,
        width=1,
    )
    draw.text(
        xy=(_RANK_X, y + 13),
        text=_rank_text(position=position),
        font=fonts.rank,
        fill=spec.accent if position <= 3 else _MUTED,
    )
    display_name = fit_text(
        draw=draw, text=row.name or "未知玩家", font=fonts.body, max_width=_NAME_MAX_WIDTH
    )
    draw.text(xy=(_NAME_X, y + 13), text=display_name, font=fonts.body, fill=_TEXT)
    draw_text_right(
        draw=draw,
        text=_ranking_amount_text(spec=spec, amount=row.amount),
        xy=(_AMOUNT_RIGHT, y + 13),
        font=fonts.body,
        fill=_TEXT,
    )


def _draw_empty_row(draw: ImageDraw.ImageDraw, fonts: _BoardFonts, y: int) -> None:
    """Draws an empty-state row."""
    draw.rectangle(
        xy=(_BOARD_MARGIN, y, _BOARD_WIDTH - _BOARD_MARGIN, y + _ROW_HEIGHT), fill=_SURFACE
    )
    draw.text(xy=(_RANK_X, y + 16), text="目前沒有排行資料", font=fonts.body, fill=_MUTED)


def _ranking_amount_text(spec: _RankingBoardSpec, amount: int) -> str:
    """Formats the amount column for one ranking row."""
    amount_text = compact_amount(amount=amount)
    if not spec.amount_label:
        return amount_text
    return f"{spec.amount_label} {amount_text}"


def _rank_text(position: int) -> str:
    """Formats a ranking number."""
    medals = {1: "1", 2: "2", 3: "3"}
    return medals.get(position, str(position))
