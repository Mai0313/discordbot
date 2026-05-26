"""Shared Pillow text drawing helpers for board / market PNG renderers."""

from PIL import ImageDraw, ImageFont

type BoardFont = ImageFont.ImageFont | ImageFont.FreeTypeFont

REGULAR_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "NotoSansCJK-Regular.ttc",
    "DejaVuSans.ttf",
)
BOLD_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "NotoSansCJK-Bold.ttc",
    "DejaVuSans-Bold.ttf",
)


def load_font(size: int, bold: bool) -> BoardFont:
    """Loads a CJK-capable font when available."""
    candidates = BOLD_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    for candidate in candidates:
        try:
            return ImageFont.truetype(font=candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: BoardFont) -> int:
    """Returns rendered text width."""
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    return bbox[2] - bbox[0]


def draw_text_right(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: BoardFont,
    fill: tuple[int, int, int],
) -> None:
    """Draws text with its right edge anchored at x."""
    right, y = xy
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    draw.text(xy=(right - (bbox[2] - bbox[0]), y), text=text, font=font, fill=fill)


def draw_text_center(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: tuple[int, int],
    font: BoardFont,
    fill: tuple[int, int, int],
) -> None:
    """Draws text centered around a point."""
    x, y = center
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    width = bbox[2] - bbox[0]
    draw.text(xy=(x - width // 2, y), text=text, font=font, fill=fill)


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: BoardFont, max_width: int) -> str:
    """Truncates text with an ellipsis suffix to fit the requested pixel width."""
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
