"""Shared Pillow font and text-drawing primitives for board/chart renderers.

The economy ranking boards render CJK text onto PNGs through these font
loading and text-anchoring helpers. This module is the single source for
those primitives so every board renderer stays aligned.
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

# Weights that have already warned. One warn per weight: every render calls load_font several
# times, and a deployment can be missing only the bold face.
_WARNED_FALLBACK_WEIGHTS: set[str] = set()


def load_font(size: int, bold: bool) -> Font:
    """Loads a CJK-capable font when available, else the Pillow default."""
    candidates = BOLD_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    for candidate in candidates:
        try:
            return ImageFont.truetype(font=candidate, size=size)
        except OSError:
            continue
    weight = "bold" if bold else "regular"
    if weight not in _WARNED_FALLBACK_WEIGHTS:
        _WARNED_FALLBACK_WEIGHTS.add(weight)
        logfire.warn(
            "No CJK font found; PNG renders fall back to the Pillow default",
            bold=bold,
            candidates=list(candidates),
        )
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: Font) -> int:
    """Returns the rendered pixel width of `text`."""
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    return int(bbox[2] - bbox[0])


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: Font, max_width: int) -> str:
    """Truncates `text` with an ellipsis so it fits within `max_width` pixels."""
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
    """Draws `text` with its right edge anchored at x."""
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
    """Draws `text` centered horizontally around a point."""
    x, y = center
    width = text_width(draw=draw, text=text, font=font)
    draw.text(xy=(x - width // 2, y), text=text, font=font, fill=fill)
