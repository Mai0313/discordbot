"""Pillow renderers for the two public economy ranking boards, plus the cache that serves them.

`/leaderboard` and `/loss_leaderboard` each post a top-ten table. Ten rows of CJK names against
兆-scale amounts do not line up in an embed's proportional text, so the table is drawn here on a
fixed column grid and attached as a PNG under the filenames this module exports, leaving the embed
beside it to carry only the champion. Both boards are the same drawing: one renderer consumes a
`_RankingBoardSpec`, and the two `build_*` entry points differ only in title, subtitle, accent
colour and which field of the caller's row model is ranked.

This is the economy cog's own rendering half rather than a `services/` module because nothing else
draws these boards, and because the ledger must not reach Pillow — #415 moved the file out of
`services/economy/` for exactly that. Nothing here touches Discord either, so the cog stays about
commands and this file stays about pixels; the shared font, measuring and anchoring primitives sit
one layer further down, in `utils/pil_text.py`.

Rendered bytes are cached on the whole spec, rows included, so a cached board can never go stale in
content: a balance change mints a new key rather than poisoning the old one. No ledger write path
invalidates this cache and none may, which is what keeps the import direction one-way; the TTL
bounds the dict's size rather than its freshness. `_drop_expired_boards` has the rest.
"""

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
from discordbot.services.economy.presentation import CURRENCY_NAME

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
    """The five faces one board render draws with.

    `Font` is a Pillow union pydantic cannot model, hence `arbitrary_types_allowed`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: Font = Field(..., description="Board title font.")
    header: Font = Field(..., description="Table header font.")
    rank: Font = Field(..., description="Ranking number font.")
    body: Font = Field(..., description="Row text font.")
    small: Font = Field(..., description="Subtitle and badge font.")


class _RankingBoardSpec(BaseModel):
    """Everything one ranking board is drawn from, and the render cache's key.

    Frozen so it can be that key. The rows travel inside it, which is what makes a cached board
    impossible to serve stale: different rows are a different spec.
    """

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Board title text.")
    subtitle: str = Field(..., description="Subtitle line under the title.")
    amount_header: str = Field(..., description="Amount column header text.")
    amount_label: str = Field(
        ..., description="Label prefixed to each amount cell, empty for none."
    )
    accent: tuple[int, int, int] = Field(
        ..., description="Accent RGB color for the badge, amount header and top three ranks."
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


def build_balance_leaderboard_board_image(rows: Sequence[LeaderboardEntry]) -> bytes:
    """Renders the balance board `/leaderboard` attaches.

    The ranking is the caller's: rows are drawn in the order given and nothing here sorts, dedupes
    or truncates them, so the ten rows a user sees are `top_n`'s own limit rather than this file's.

    Args:
        rows (Sequence[LeaderboardEntry]): Accounts in ranking order, highest balance first.

    Returns:
        PNG bytes, freshly rendered or served from the render cache.
    """
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
    """Renders the daily casino loss board `/loss_leaderboard` attaches.

    Same contract as the balance board: the order is the caller's. The subtitle states the
    Asia/Taipei reset because the amounts come from `casino_account`'s day counter, not from a
    window this renderer computes.

    Args:
        rows (Sequence[LossLeaderboardEntry]): Accounts in ranking order, largest loss first.

    Returns:
        PNG bytes, freshly rendered or served from the render cache.
    """
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
    """Returns the board for `spec`, rendering it only on a cache miss.

    The sweep runs before the lookup, so an expired entry is never handed back. A hit does not
    refresh the stored timestamp either: an entry lives at most one TTL from the render that
    created it, however often it is asked for.

    Args:
        spec (_RankingBoardSpec): The board to draw, and the cache key it is stored under.

    Returns:
        PNG bytes for the board.
    """
    now = monotonic()
    _drop_expired_boards(now=now)
    cached = _board_image_cache.get(spec)
    if cached is not None:
        _, image = cached
        return image
    image = _render_ranking_board_image(spec=spec)
    _board_image_cache[spec] = (now, image)
    return image


def _drop_expired_boards(now: float) -> None:
    """Evicts board images past the TTL.

    The cache key carries the rows it rendered, so an entry can never go stale in content: a
    balance change mints a new key and strands the old one instead of poisoning it. Expiry is
    therefore the size bound rather than a freshness rule, and nothing on the write side has to
    clear this.

    Args:
        now (float): The `monotonic()` reading every entry's age is measured against.
    """
    expired = [
        spec
        for spec, (cached_at, _) in _board_image_cache.items()
        if now - cached_at > _BOARD_IMAGE_CACHE_TTL_SECONDS
    ]
    for spec in expired:
        del _board_image_cache[spec]


def _render_ranking_board_image(spec: _RankingBoardSpec) -> bytes:
    """Draws one board and encodes it, uncached.

    Only the height varies: it is computed from the row count, so a longer table grows the image
    instead of being clipped or paged. An empty table still reserves one row, which is what the
    placeholder line is drawn into.

    Args:
        spec (_RankingBoardSpec): The board to draw.

    Returns:
        The encoded PNG bytes.
    """
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
    """Loads the board's five faces, once per process.

    Cached because opening a face is the expensive half of a render and the sizes never vary. The
    consequence is that a font installed or replaced on disk is only picked up after a restart.

    Returns:
        The shared font set; callers must treat it as read-only.
    """
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
    """Draws the title, the subtitle line and the `PUBLIC` badge.

    Every offset is taken from the top margin and none is measured, so this block occupies exactly
    `_BOARD_HEADER_HEIGHT` and the table below can start from that constant.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        fonts (_BoardFonts): Faces for the title, subtitle and badge.
        title (str): Board title text.
        subtitle (str): Line under the title.
        accent (tuple[int, int, int]): RGB colour for the badge outline and caption.
    """
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
    """Draws the column header strip at `y`.

    Uses the same column constants and the same right anchor as `_draw_rank_row`, which is the
    only thing keeping a header over its own cells.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        fonts (_BoardFonts): Faces for the header row.
        y (int): Top edge of the strip.
        amount_header (str): Right-anchored caption of the amount column.
        accent (tuple[int, int, int]): RGB colour for that caption.
    """
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
    """Draws one ranked row band at `y`.

    The banding and the top-three highlight both key off the one-based position, so a row drawn out
    of order would take the wrong stripe and the wrong colour. A blank name falls back to 未知玩家,
    and the name is trimmed to `_NAME_MAX_WIDTH` before it is drawn so it cannot run into the
    right-anchored amount.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        fonts (_BoardFonts): Faces for the rank number and the row text.
        row (_RankingRow): Position, name and amount for this row.
        spec (_RankingBoardSpec): Board the row belongs to, read for the accent and amount label.
        y (int): Top edge of the row band.
    """
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
    """Draws the placeholder line into the row an empty table reserved.

    Both commands answer an empty ranking with an embed of their own before they ever build a
    board, so this is the fallback for a direct caller rather than something a user normally sees.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        fonts (_BoardFonts): Faces for the placeholder text.
        y (int): Top edge of the reserved row.
    """
    draw.rectangle(
        xy=(_BOARD_MARGIN, y, _BOARD_WIDTH - _BOARD_MARGIN, y + _ROW_HEIGHT), fill=_SURFACE
    )
    draw.text(xy=(_RANK_X, y + 16), text="目前沒有排行資料", font=fonts.body, fill=_MUTED)


def _ranking_amount_text(spec: _RankingBoardSpec, amount: int) -> str:
    """Formats the amount cell for one row.

    Compact units are what keep a 兆-scale balance inside its column; the exact digits would
    overrun it. Both shipped specs leave `amount_label` empty, so the prefix is there for a future
    board whose column needs naming in the cell.

    Args:
        spec (_RankingBoardSpec): Board the row belongs to, read for its amount label.
        amount (int): Value to render.

    Returns:
        The compact amount, prefixed with the label and a space when the spec carries one.
    """
    amount_text = compact_amount(amount=amount)
    if not spec.amount_label:
        return amount_text
    return f"{spec.amount_label} {amount_text}"


def _rank_text(position: int) -> str:
    """Formats the rank cell for one position.

    The top three are singled out by the accent colour in `_draw_rank_row`, not by their text: the
    medal emoji this table once held went with the embed-era rows, and every entry left in it maps
    to its own plain digit.

    Args:
        position (int): One-based ranking position.

    Returns:
        The position as decimal text.
    """
    medals = {1: "1", 2: "2", 3: "3"}
    return medals.get(position, str(position))
