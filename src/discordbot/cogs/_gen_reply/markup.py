"""Parse the answer model's inline `<voice>` / `<image>` control tags out of a reply.

The answer model marks parts of its reply for special handling with tag pairs:

* `<voice>...</voice>` wraps the exact span it wants read aloud. Only that span is
  synthesized into a voice clip, but the wrapped text STAYS in the visible reply (only the
  tags are removed). This replaces the old whole-reply `</need-voice>` marker, so a long
  written reply can still speak just one conversational line.
* `<image>...</image>` wraps a rough description of an image to generate. The whole block
  (tags AND description) is REMOVED from the visible reply, so the raw generation prompt
  never leaks to the user; the description is handed to the image pipeline instead.

`extract_reply_segments` runs once on the finished reply to split it into the display text,
the spoken span, and the image prompt. `strip_tags_for_preview` does the same hiding live on
each streaming snapshot so a half-typed tag never flickers into the preview. Parsing tolerates
stray whitespace inside a tag, but a tag shown inside an inline-code span or a fenced code
block is treated as literal content, not a control tag, so a reply that demonstrates `<image>`
or `<voice>` as a code example is never corrupted or sent to generation. The whole module is
pure string work; it never touches Discord or the network.
"""

import re

from pydantic import Field, BaseModel

# Canonical tag literals shown to the model in the prompt; the regexes below accept lenient
# variants (inner whitespace, any case). A backticked or fenced tag is NOT a control tag: code
# regions are masked before parsing (see `_mask_code`), matching the prompt's "never backticked".
VOICE_OPEN_TAG = "<voice>"
VOICE_CLOSE_TAG = "</voice>"
IMAGE_OPEN_TAG = "<image>"
IMAGE_CLOSE_TAG = "</image>"

# Tag bodies, tolerant of stray whitespace the model may add around the name / slash.
_VOICE_OPEN = r"<\s*voice\s*>"
_VOICE_CLOSE = r"<\s*/\s*voice\s*>"
_IMAGE_OPEN = r"<\s*image\s*>"
_IMAGE_CLOSE = r"<\s*/\s*image\s*>"

# Single-tag matchers, for removal. No backtick handling here: a backticked tag is masked as
# code before these run (see `_mask_code`), so a real control tag is always the bare form.
_VOICE_OPEN_RE = re.compile(_VOICE_OPEN, re.IGNORECASE)
_VOICE_CLOSE_RE = re.compile(_VOICE_CLOSE, re.IGNORECASE)
_IMAGE_OPEN_RE = re.compile(_IMAGE_OPEN, re.IGNORECASE)

# Whole-block matchers; group 1 is the inner content (DOTALL so it spans newlines).
_VOICE_BLOCK_RE = re.compile(rf"{_VOICE_OPEN}(.*?){_VOICE_CLOSE}", re.IGNORECASE | re.DOTALL)
_IMAGE_BLOCK_RE = re.compile(rf"{_IMAGE_OPEN}(.*?){_IMAGE_CLOSE}", re.IGNORECASE | re.DOTALL)

# A dangling (unclosed) open tag to the end of the text; group 1 is the trailing content.
_VOICE_OPEN_TAIL_RE = re.compile(rf"{_VOICE_OPEN}(.*)\Z", re.IGNORECASE | re.DOTALL)
_IMAGE_OPEN_TAIL_RE = re.compile(rf"{_IMAGE_OPEN}(.*)\Z", re.IGNORECASE | re.DOTALL)

# Code regions (fenced ``` / ~~~ blocks and inline `...` spans) are masked with opaque
# placeholders before tag parsing, so a tag written as a code example is left untouched.
_CODE_SPAN_RE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]+`", re.DOTALL)
_CODE_PLACEHOLDER_RE = re.compile(r"\x01(\d+)\x01")

# Sentinel marking where an image block was removed, so whitespace healing touches only that
# seam and leaves the rest of the reply (code blocks, aligned tables, ASCII art) byte-for-byte.
_IMAGE_SEAM = "\x00"
# An inline seam with text on both sides collapses to one space; horizontal padding around the
# seam is absorbed, but newlines are not, so a block alone on its line keeps its blank lines.
_INLINE_SEAM_RE = re.compile(rf"(?<=\S)[^\S\n]*{_IMAGE_SEAM}[^\S\n]*(?=\S)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# All tags whose half-typed prefix should be trimmed off a live preview snapshot.
_PARTIAL_TAGS = (IMAGE_OPEN_TAG, IMAGE_CLOSE_TAG, VOICE_OPEN_TAG, VOICE_CLOSE_TAG)


class ReplySegments(BaseModel):
    """The answer reply split into its visible text and its extracted control spans.

    Attributes:
        display_text: The reply text to show, with voice tags removed (content kept) and
            image blocks removed entirely (tags and content).
        voice_text: The concatenated content of every `<voice>` span, or "" when none.
        image_prompt: The first `<image>` span's content (the rough draw request), or "".
        voice_requested: Whether a non-empty spoken span was found.
        image_requested: Whether a non-empty image span was found.
    """

    display_text: str = Field(..., description="Reply text to display, control tags removed.")
    voice_text: str = Field(
        default="", description="Concatenated spoken-span text, empty when none."
    )
    image_prompt: str = Field(
        default="", description="First image span's rough description, empty when none."
    )
    voice_requested: bool = Field(
        default=False, description="Whether a non-empty spoken span was present."
    )
    image_requested: bool = Field(
        default=False, description="Whether a non-empty image span was present."
    )


def _mask_code(text: str) -> tuple[str, list[str]]:
    """Replaces code spans/fences with opaque placeholders so tag parsing skips them."""
    spans: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"\x01{len(spans) - 1}\x01"

    return _CODE_SPAN_RE.sub(_stash, text), spans


def _restore_code(text: str, spans: list[str]) -> str:
    """Puts the masked code spans back exactly as they were."""
    return _CODE_PLACEHOLDER_RE.sub(lambda match: spans[int(match.group(1))], text)


def _heal_image_seams(text: str) -> str:
    """Closes the gap a removed image block left, touching only that seam.

    Each removed block is marked with `_IMAGE_SEAM`. An inline gap (text on both sides)
    becomes one space, a block alone on its line(s) collapses its surrounding blank lines, and
    a leading/trailing block leaves nothing. Everything else is left exactly as the model wrote
    it, so this never collapses spacing in code blocks, aligned tables, or ASCII art elsewhere.
    """
    text = _INLINE_SEAM_RE.sub(" ", text)
    text = text.replace(_IMAGE_SEAM, "")
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _trim_trailing_partial(text: str, tag: str) -> str | None:
    """Trims a trailing fragment that is a length>=2 prefix of `tag`, else returns None.

    Used on a live preview so a half-streamed tag (e.g. `<voi`, `</ima`) never flickers in.
    Fragments of length 1 (a bare `<`) are left alone so a literal `<` in the reply survives.
    """
    lowered = text.lower()
    tag_lowered = tag.lower()
    for length in range(len(tag) - 1, 1, -1):
        if lowered.endswith(tag_lowered[:length]):
            return text[: len(text) - length]
    return None


def extract_reply_segments(text: str) -> ReplySegments:
    """Splits a finished reply into display text plus its voice and image spans.

    Image blocks are removed whole (the description must not leak); voice spans keep their
    inner text in the display and only contribute it to the spoken clip. A dangling unclosed
    tag (the model forgot the closer) is still handled: an unclosed image drops to the end,
    an unclosed voice keeps its content visible and spoken.
    """
    # Mask code first so a tag shown as a code example is never parsed as a control tag.
    masked, code_spans = _mask_code(text)

    image_match = _IMAGE_BLOCK_RE.search(masked)
    image_prompt = image_match.group(1).strip() if image_match is not None else ""
    display = _IMAGE_BLOCK_RE.sub(_IMAGE_SEAM, masked)
    dangling_image = _IMAGE_OPEN_TAIL_RE.search(display)
    if dangling_image is not None:
        if not image_prompt:
            image_prompt = dangling_image.group(1).strip()
        display = _IMAGE_OPEN_TAIL_RE.sub(_IMAGE_SEAM, display)
    image_removed = _IMAGE_SEAM in display

    before_voice = display
    voice_parts = [match.group(1).strip() for match in _VOICE_BLOCK_RE.finditer(display)]
    display = _VOICE_BLOCK_RE.sub(r"\1", display)
    dangling_voice = _VOICE_OPEN_TAIL_RE.search(display)
    if dangling_voice is not None:
        voice_parts.append(dangling_voice.group(1).strip())
        display = _VOICE_OPEN_TAIL_RE.sub(r"\1", display)
    display = _VOICE_CLOSE_RE.sub("", display)
    voice_removed = display != before_voice

    # Only touch whitespace where markup was actually removed; a reply with no control tags is
    # returned exactly as written so code blocks, aligned tables, and ASCII art survive intact.
    if image_removed:
        display = _heal_image_seams(display)
    elif voice_removed:
        display = display.rstrip()

    voice_text = _restore_code("\n".join(part for part in voice_parts if part).strip(), code_spans)
    image_prompt = _restore_code(image_prompt, code_spans)
    return ReplySegments(
        display_text=_restore_code(display, code_spans),
        voice_text=voice_text,
        image_prompt=image_prompt,
        voice_requested=bool(voice_text),
        image_requested=bool(image_prompt),
    )


def strip_tags_for_preview(text: str) -> str:
    """Hides control tags (complete or mid-stream) from a live preview snapshot.

    Complete image blocks and any unclosed image tail are hidden so the raw draw prompt never
    shows; voice tags are dropped but their content stays; a trailing half-typed tag of any
    kind is trimmed so the control token never flickers into the preview before the final pass.
    """
    masked, code_spans = _mask_code(text)
    masked = _IMAGE_BLOCK_RE.sub("", masked)
    open_image = _IMAGE_OPEN_RE.search(masked)
    if open_image is not None:
        masked = masked[: open_image.start()]
    else:
        trimmed_image = _trim_trailing_partial(masked, IMAGE_OPEN_TAG)
        if trimmed_image is not None:
            masked = trimmed_image

    masked = _VOICE_BLOCK_RE.sub(r"\1", masked)
    masked = _VOICE_OPEN_RE.sub("", masked)
    masked = _VOICE_CLOSE_RE.sub("", masked)

    for tag in _PARTIAL_TAGS:
        trimmed = _trim_trailing_partial(masked, tag)
        if trimmed is not None:
            masked = trimmed
            break
    return _restore_code(masked, code_spans).strip()
