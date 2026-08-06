"""Shared Pillow font, measurement and anchoring primitives for the PNG board renderers.

`cogs/economy/boards.py` (the ranking boards) and `cogs/stock/presentation.py` (the market board)
are the only two importers. Both draw CJK text onto a PNG, so both need a face that can actually
render CJK, a pixel width for a string, the trim that keeps an overlong one inside its column, and
a way to place it flush right; only the ranking boards also center a string. One copy of them lives
here so the two boards keep the same look, and it sits in `utils/` rather than in either cog
because a cog may not import a peer cog to reach a helper. The stock cog's 7D chart is not a
caller: it draws its own ASCII-only labels in the Pillow default face, so none of its text rides
the fallback chain below.

The contract is deliberately small. `load_font` never raises on a missing font: it walks a list of
candidates ending in a Latin-only DejaVu face and then the Pillow default, so a container with no
Noto CJK installed renders boxes rather than failing the command outright. Nothing here caches a
loaded face (both callers build their font set once behind `functools.cache`) and nothing here
owns an image, a canvas or a palette: the caller passes its own `ImageDraw.ImageDraw` and its own
colors.

There is no layout engine either. Every helper measures or draws exactly one line; wrapping,
column geometry and row heights stay in the renderer that knows its own board, which only has to
shorten an overlong string through `fit_text` so the measurement and the placement agree.
"""

from PIL import ImageDraw, ImageFont
import logfire

type Font = ImageFont.ImageFont | ImageFont.FreeTypeFont

REGULAR_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "NotoSansCJK-Regular.ttc",
    "DejaVuSans.ttf",
)
BOLD_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "NotoSansCJK-Bold.ttc",
    "DejaVuSans-Bold.ttf",
)

# One warn per weight: every render calls load_font several times, and a deployment can be
# missing only the bold face.
_font_fallback_warned: set[bool] = set()


def load_font(size: int, bold: bool) -> Font:
    """Loads a CJK-capable font at `size`, falling back to the Pillow default.

    Walks the weight's candidate list: the packaged Debian/Ubuntu paths first, then the bare
    filenames Pillow resolves against its own search path, which is why a miss is caught rather
    than probed for on disk. Running out of candidates is not an error, only a downgrade, with two
    caveats on that tail: the last candidate is Latin-only DejaVu, so a CJK downgrade to it is
    silent, and the Pillow default below it is loaded at its own fixed size, ignoring `size`.

    Only a missing font is absorbed. `truetype` raises `ValueError` for a `size` of zero or less,
    and re-raises the deferred `ImportError` when Pillow was built without FreeType; neither is an
    `OSError`, so either one escapes the first candidate and the fallback chain never runs.

    Args:
        size (int): Point size to load the face at.
        bold (bool): Selects the bold candidate list instead of the regular one.

    Returns:
        The first candidate that loaded, or the Pillow default font.
    """
    candidates = BOLD_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    for candidate in candidates:
        try:
            return ImageFont.truetype(font=candidate, size=size)
        except OSError:
            continue
    if bold not in _font_fallback_warned:
        _font_fallback_warned.add(bold)
        logfire.warn(
            "No CJK font found; PNG renders fall back to the Pillow default",
            bold=bold,
            candidates=list(candidates),
        )
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: Font) -> int:
    """Returns the rendered pixel width of `text` in `font`.

    The one measurement the rest of the module agrees on: `fit_text` budgets against it and both
    anchored draws offset by it, so a string trimmed to a column's width lands inside that column.

    Args:
        draw (ImageDraw.ImageDraw): Canvas whose text metrics do the measuring.
        text (str): String to measure.
        font (Font): Face it will be drawn in.

    Returns:
        The width in pixels.
    """
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    return int(bbox[2] - bbox[0])


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: Font, max_width: int) -> str:
    """Truncates `text` with a trailing `...` so it fits within `max_width` pixels.

    Binary-searches the longest prefix that still fits once the ellipsis is appended: mixed
    CJK/Latin width is not proportional to character count, so the cut has to be measured, and the
    search keeps that to a handful of measurements rather than one per character. When not even a
    single character fits beside it, the bare ellipsis comes back and may itself overflow
    `max_width`, so the cell shows that something was dropped instead of going blank.

    Args:
        draw (ImageDraw.ImageDraw): Canvas whose text metrics do the measuring.
        text (str): String to fit.
        font (Font): Face it will be drawn in.
        max_width (int): Pixel budget the result must fit into.

    Returns:
        `text` unchanged when it already fits, else a prefix with `...` appended.
    """
    if text_width(draw=draw, text=text, font=font) <= max_width:
        return text
    suffix = "..."
    low = 0
    high = len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        candidate = f"{text[:midpoint]}{suffix}"
        if text_width(draw=draw, text=candidate, font=font) <= max_width:
            low = midpoint
        else:
            high = midpoint - 1
    if low == 0:
        return suffix
    return f"{text[:low]}{suffix}"


def draw_text_right(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: Font,
    fill: tuple[int, int, int],
) -> None:
    """Draws `text` with its right edge at the x in `xy`.

    Only the x is derived, by shifting left by `text_width`; the y is passed to Pillow untouched
    and means exactly what it would in a plain `draw.text` call.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        text (str): String to draw.
        xy (tuple[int, int]): Right edge x and the y a plain `draw.text` would take.
        font (Font): Face to draw it in.
        fill (tuple[int, int, int]): RGB colour.
    """
    right, y = xy
    draw.text(
        xy=(right - text_width(draw=draw, text=text, font=font), y),
        text=text,
        font=font,
        fill=fill,
    )


def draw_text_center(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: tuple[int, int],
    font: Font,
    fill: tuple[int, int, int],
) -> None:
    """Draws `text` centered horizontally on the x in `center`.

    The half-width is integer division, which floors the leftward shift, so an odd-width string
    sits half a pixel right of the point and an even-width one lands exactly on it; the y is passed
    through untouched, as in `draw_text_right`.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        text (str): String to draw.
        center (tuple[int, int]): Center x and the y a plain `draw.text` would take.
        font (Font): Face to draw it in.
        fill (tuple[int, int, int]): RGB colour.
    """
    x, y = center
    width = text_width(draw=draw, text=text, font=font)
    draw.text(xy=(x - width // 2, y), text=text, font=font, fill=fill)
