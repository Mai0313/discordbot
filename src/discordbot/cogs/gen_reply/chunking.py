"""Code-fence-aware Discord message chunking for streamed replies."""

import re
import unicodedata

from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT

# Appended when `surface.answer_capacity` caps an `/ask` reply.
TRUNCATED_NOTICE = "\n-# 回覆太長，這裡放不下後面的內容了"

_CODE_FENCE = "```"
# Bounded language tag carried onto reopened code blocks so splitting always makes progress.
_FENCE_LANGUAGE_RE = re.compile(pattern=r"[A-Za-z0-9_+\-.#]{0,20}")
# Combining, format, and emoji categories not to split beside when hard-cutting.
_CLINGING_CATEGORIES = frozenset({"Mn", "Mc", "Me", "Cf", "Sk", "So"})


def _in_code_block(text: str, cut: int) -> bool:
    """Whether `cut` falls inside a code block of `text`, pairing fences the way Discord does."""
    return text.count(_CODE_FENCE, 0, cut) % 2 == 1 and text.find(_CODE_FENCE, cut) != -1


def _last_break(text: str, end: int, floor: int, outside_code: bool) -> int:
    """Returns the cleanest break index in `text[:end]` at or after `floor`."""
    for separator in ("\n\n", "\n", " "):
        cut = text.rfind(separator, 0, end)
        while outside_code and cut >= floor and _in_code_block(text=text, cut=cut):
            cut = text.rfind(separator, 0, cut)
        if cut >= floor:
            return cut
    cut = end
    opening = text.rfind("<", 0, end)
    if opening >= floor and opening > text.rfind(">", 0, end):
        cut = opening
    while cut > floor and _CLINGING_CATEGORIES & {
        unicodedata.category(text[cut - 1]),
        unicodedata.category(text[cut]),
    }:
        cut -= 1
    return cut


def _cut_cleanly(text: str, budget: int, earliest: int) -> tuple[str, str]:
    """Splits `text` within `[earliest, budget]`, closing and reopening overrun code fences."""
    floor = max(budget // 2, earliest)
    cut = _last_break(text=text, end=budget, floor=floor, outside_code=True)
    if not _in_code_block(text=text, cut=cut):
        return text[:cut], text[cut:]
    opener = text.rfind(_CODE_FENCE, 0, cut)
    if opener >= earliest and text[:opener].strip():
        return text[:opener], text[opener:]
    language = text[opener + len(_CODE_FENCE) :].split("\n", 1)[0]
    fence = (
        _CODE_FENCE + language if _FENCE_LANGUAGE_RE.fullmatch(string=language) else _CODE_FENCE
    )
    # Reserve room for the closing fence here and reopening fence on the remainder.
    end = budget - len(_CODE_FENCE) - 1
    fenced_floor = max(floor, earliest + len(fence) + 1)
    if fenced_floor > end:
        return text[:cut], text[cut:]
    cut = _last_break(text=text, end=end, floor=fenced_floor, outside_code=False)
    head, rest = text[:cut], text[cut:]
    if not _in_code_block(text=text, cut=cut):
        return head, rest
    reopened = f"{fence}{rest}" if rest.startswith("\n") else f"{fence}\n{rest}"
    return f"{head}\n{_CODE_FENCE}", reopened


def split_reply_for_discord(
    content: str, footer: str, max_messages: int | None = None
) -> tuple[str, list[str]]:
    """Splits `content` and `footer` into parent and follow-up chunks within `max_messages`.

    Args:
        content (str): Reply body text before the usage footer.
        footer (str): Usage footer appended to the final message chunk.
        max_messages (int | None): Optional total message budget (parent plus follow-ups); when
            exceeded, truncates cleanly with `TRUNCATED_NOTICE`.

    Raises:
        ValueError: `footer` alone meets or exceeds `DISCORD_MESSAGE_LIMIT`.
    """
    if len(f"{content}{footer}") <= DISCORD_MESSAGE_LIMIT:
        return f"{content}{footer}", []
    if len(footer) >= DISCORD_MESSAGE_LIMIT:
        raise ValueError("Usage footer is too long for Discord message content")

    messages: list[str] = []
    remaining = content
    while len(remaining) + len(footer) > DISCORD_MESSAGE_LIMIT:
        left = None if max_messages is None else max_messages - len(messages)
        if left is not None and left <= 1:
            budget = DISCORD_MESSAGE_LIMIT - len(footer) - len(TRUNCATED_NOTICE)
            # Last allowed message: close an overrun code block instead of breaking before it.
            head, _ = _cut_cleanly(text=remaining, budget=budget, earliest=budget // 2)
            remaining = f"{head}{TRUNCATED_NOTICE}"
            break
        if not messages and len(remaining) <= DISCORD_MESSAGE_LIMIT:
            # An answer that fits one message is not split for its footer, which follows alone.
            messages.append(remaining)
            remaining = ""
            break
        # Keep text ahead of the footer on follow-ups so Discord does not strip
        # USAGE_FOOTER_RE's leading blank line.
        budget = (
            DISCORD_MESSAGE_LIMIT
            if len(remaining) > DISCORD_MESSAGE_LIMIT
            else DISCORD_MESSAGE_LIMIT - len(footer)
        )
        # Earliest cut that lets remaining text + footer fit in the remaining message slots.
        earliest = 0
        if left is not None:
            needed = len(remaining) + len(footer) - (left - 1) * DISCORD_MESSAGE_LIMIT
            earliest = needed if needed <= budget else 0
        head, remaining = _cut_cleanly(text=remaining, budget=budget, earliest=earliest)
        messages.append(head)
    messages.append(f"{remaining}{footer}")
    return messages[0], messages[1:]
