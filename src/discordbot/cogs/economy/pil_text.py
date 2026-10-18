"""Pillow font loading and text-drawing primitives for PNG board renderers."""

from PIL import ImageDraw, ImageFont
import logfire

type Font = ImageFont.ImageFont | ImageFont.FreeTypeFont

# Weights that have already warned. One warn per weight: every render calls load_font several
# times, and a deployment can be missing only the bold face.
_WARNED_FALLBACK_WEIGHTS: set[str] = set()


def load_font(size: int, bold: bool) -> Font:
    """Returns a CJK-capable font when available, else the Pillow default.

    Args:
        size (int): Font size in points.
        bold (bool): Whether to load the bold weight.
    """
    candidates = (
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
            "NotoSansCJK-Bold.ttc",
            "DejaVuSans-Bold.ttf",
        )
        if bold
        else (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "NotoSansCJK-Regular.ttc",
            "DejaVuSans.ttf",
        )
    )
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
    """Returns the rendered pixel width of `text`.

    Args:
        draw (ImageDraw.ImageDraw): Drawing context used to measure `text`.
        text (str): Text to measure.
        font (Font): Font used to measure `text`.
    """
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    return int(bbox[2] - bbox[0])


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: Font, max_width: int) -> str:
    """Returns `text` truncated with an ellipsis to fit within `max_width` pixels.

    Args:
        draw (ImageDraw.ImageDraw): Drawing context used to measure `text`.
        text (str): Text to fit.
        font (Font): Font used to measure `text`.
        max_width (int): Maximum allowed width in pixels.
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
    """Draws `text` with its right edge anchored at `xy`.

    Args:
        draw (ImageDraw.ImageDraw): Drawing context to draw on.
        text (str): Text to draw.
        xy (tuple[int, int]): Right-edge x and top y coordinates.
        font (Font): Font to render with.
        fill (tuple[int, int, int]): RGB fill color.
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
    """Draws `text` centered horizontally around `center`.

    Args:
        draw (ImageDraw.ImageDraw): Drawing context to draw on.
        text (str): Text to draw.
        center (tuple[int, int]): Center x and top y coordinates.
        font (Font): Font to render with.
        fill (tuple[int, int, int]): RGB fill color.
    """
    x, y = center
    width = text_width(draw=draw, text=text, font=font)
    draw.text(xy=(x - width // 2, y), text=text, font=font, fill=fill)
