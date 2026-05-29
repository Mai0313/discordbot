"""Shared Pillow font and text-drawing primitives for board/chart renderers.

The economy ranking boards and the stock market board both render CJK text
onto PNGs with the same font loading and text-anchoring helpers. This module
is the single source for those primitives so the two renderers stay aligned.
"""

from PIL import ImageDraw, ImageFont

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


def load_font(size: int, bold: bool) -> Font:
    """Loads a CJK-capable font when available, else the Pillow default."""
    candidates = BOLD_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    for candidate in candidates:
        try:
            return ImageFont.truetype(font=candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: Font) -> int:
    """Returns the rendered pixel width of `text`."""
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    return bbox[2] - bbox[0]


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
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    draw.text(xy=(right - (bbox[2] - bbox[0]), y), text=text, font=font, fill=fill)


def draw_text_center(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: tuple[int, int],
    font: Font,
    fill: tuple[int, int, int],
) -> None:
    """Draws `text` centered horizontally around a point."""
    x, y = center
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    width = bbox[2] - bbox[0]
    draw.text(xy=(x - width // 2, y), text=text, font=font, fill=fill)
