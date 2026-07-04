"""Inline reply markers: the answer model self-selects spoken segments, images, music, and video.

The answer model wraps the parts of its reply it wants read aloud in `<voice>...</voice>`: only
those segments are synthesized (concatenated into a single voice clip), but they STAY in the
visible reply (only the tags are stripped). It may also wrap short descriptions in
`<image>...</image>` to have images generated and attached, one `<music>...</music>` description
to have a music clip generated and attached, or one `<video>...</video>` description to have a
short video generated and attached; each such block (tags AND content) is REMOVED from the visible
reply so the generation prompt never leaks into chat. `ResponseStreamer` extracts them at finalize
time via `extract_inline_markers` and scrubs partial/complete tags from the live preview via
`scrub_markers_for_preview`, so none flickers mid-stream. The asymmetry is deliberate: voice
content is meant to stay visible, image / music / video content are meant to be pulled.
"""

import re

from pydantic import Field, BaseModel

# Tag literals are the single source of truth shared by the prompt instructions and this parser.
VOICE_OPEN = "<voice>"
VOICE_CLOSE = "</voice>"
IMAGE_OPEN = "<image>"
IMAGE_CLOSE = "</image>"
MUSIC_OPEN = "<music>"
MUSIC_CLOSE = "</music>"
VIDEO_OPEN = "<video>"
VIDEO_CLOSE = "</video>"
DEEP_RESEARCH_OPEN = "<deep-research>"
DEEP_RESEARCH_CLOSE = "</deep-research>"

# Hard cap on inline images per reply: a voice clip plus 9 images exactly fills Discord's
# 10-attachment ceiling. The prompt tells the model this limit; the streamer enforces it by
# dropping any extra blocks so a confused model never blows past the attachment cap. A reply
# may also carry one music clip and one video clip (each single per reply by design), so a rare
# voice + music + video + 9 images would be 12 attachments; the streamer's
# `[:DISCORD_ATTACHMENT_LIMIT]` clamp is the backstop.
MAX_INLINE_IMAGES = 9

# Complete blocks: non-greedy, DOTALL so a multi-line segment is captured, IGNORECASE so a
# stray-cased tag still matches.
_VOICE_BLOCK_RE = re.compile(r"<voice>(.*?)</voice>", re.IGNORECASE | re.DOTALL)
_IMAGE_BLOCK_RE = re.compile(r"<image>(.*?)</image>", re.IGNORECASE | re.DOTALL)
_MUSIC_BLOCK_RE = re.compile(r"<music>(.*?)</music>", re.IGNORECASE | re.DOTALL)
_VIDEO_BLOCK_RE = re.compile(r"<video>(.*?)</video>", re.IGNORECASE | re.DOTALL)
_DEEP_RESEARCH_BLOCK_RE = re.compile(
    r"<deep-research>(.*?)</deep-research>", re.IGNORECASE | re.DOTALL
)
# Bare tags, scrubbed so a stray/unpaired tag never leaks into the visible reply.
_VOICE_TAG_RE = re.compile(r"</?voice>", re.IGNORECASE)
_IMAGE_TAG_RE = re.compile(r"</?image>", re.IGNORECASE)
_MUSIC_TAG_RE = re.compile(r"</?music>", re.IGNORECASE)
_VIDEO_TAG_RE = re.compile(r"</?video>", re.IGNORECASE)
_DEEP_RESEARCH_TAG_RE = re.compile(r"</?deep-research>", re.IGNORECASE)
# An unclosed open tag and everything after it: the whole block is going to be pulled, so hide it
# the moment it starts streaming in (and tolerate the model forgetting to close it).
_TRAILING_IMAGE_OPEN_RE = re.compile(r"<image>.*\Z", re.IGNORECASE | re.DOTALL)
_TRAILING_MUSIC_OPEN_RE = re.compile(r"<music>.*\Z", re.IGNORECASE | re.DOTALL)
_TRAILING_VIDEO_OPEN_RE = re.compile(r"<video>.*\Z", re.IGNORECASE | re.DOTALL)
_TRAILING_DEEP_RESEARCH_OPEN_RE = re.compile(r"<deep-research>.*\Z", re.IGNORECASE | re.DOTALL)
_COLLAPSE_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Every tag whose half-streamed tail must be trimmed from a live preview so it never flickers in.
_ALL_TAGS = (
    IMAGE_OPEN,
    IMAGE_CLOSE,
    MUSIC_OPEN,
    MUSIC_CLOSE,
    VIDEO_OPEN,
    VIDEO_CLOSE,
    VOICE_OPEN,
    VOICE_CLOSE,
    DEEP_RESEARCH_OPEN,
    DEEP_RESEARCH_CLOSE,
)

# Markdown code regions (fenced ``` / ~~~ blocks and inline `...` spans) are masked to opaque
# sentinels BEFORE marker extraction and restored afterwards, so a literal tag the user needs to
# SEE is never mistaken for a generation marker and pulled from the reply. This matters most for
# the real HTML5 `<video>...</video>` element (the one marker whose name collides with common
# HTML — `<img>` / `<audio>` are the real tags, not `<image>` / `<music>`), and is sound because
# the prompts forbid a generation marker from ever being wrapped in backticks or a code block, so
# any tag inside a code region is example content by construction, never a marker. A NUL sentinel
# is used because it never appears in real Discord text and no marker/collapse regex matches it.
_CODE_REGION_RE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]+`", re.DOTALL)
_CODE_SENTINEL_RE = re.compile("\x00(\\d+)\x00")


def _mask_code_regions(text: str) -> tuple[str, list[str]]:
    """Replaces code regions with sentinels so marker regexes skip them; returns (masked, stash)."""
    stash: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        stash.append(match.group(0))
        return f"\x00{len(stash) - 1}\x00"

    return _CODE_REGION_RE.sub(_stash, text), stash


def _restore_code_regions(*, text: str, stash: list[str]) -> str:
    """Restores masked code regions from their sentinels (a no-op when nothing was masked)."""
    if not stash:
        return text
    return _CODE_SENTINEL_RE.sub(lambda match: stash[int(match.group(1))], text)


class InlineMarkers(BaseModel):
    """Markers extracted from a finished reply: the visible text plus its media requests."""

    cleaned_text: str = Field(
        ...,
        description="Reply text with image blocks removed and voice tags stripped (voice content kept).",
    )
    voice_text: str = Field(
        default="",
        description="Concatenated <voice> segments to synthesize aloud; empty when none.",
    )
    voice_requested: bool = Field(
        default=False, description="Whether the reply wrapped any segment in <voice>."
    )
    image_prompts: list[str] = Field(
        default_factory=list,
        description="Every <image> description to generate, in order; empty when none.",
    )
    music_prompt: str | None = Field(
        default=None,
        description="First <music> description to generate a single clip, or None when absent.",
    )
    video_prompt: str | None = Field(
        default=None,
        description="First <video> description to generate a single clip, or None when absent.",
    )
    research_brief: str | None = Field(
        default=None,
        description="First <deep-research> brief to launch a research thread, or None when absent.",
    )


def extract_inline_markers(*, text: str) -> InlineMarkers:
    """Splits a finished reply into visible text plus its voice / image / music / video requests.

    Image blocks (tags AND content) are removed entirely so the generation prompt never shows
    in chat; every non-empty one becomes an image request, in order. A `<music>` and a `<video>`
    block are pulled the same way, but only the first non-empty one of each is kept (one clip per
    reply by design). Voice tags are stripped but their inner content STAYS in the visible reply,
    and every wrapped segment is concatenated as the spoken-clip input. An unclosed trailing
    `<image>` / `<music>` / `<video>` (the model forgot to close it) is still pulled so its raw
    description never leaks, and any stray unpaired tag is scrubbed. Code regions (fenced blocks and
    inline spans) are masked first, so a literal HTML `<video>` example the user asked to see is
    preserved verbatim instead of being pulled as a generation request.
    """
    # Mask code regions so an example tag inside them (e.g. a real HTML `<video>` element) is never
    # mistaken for a marker; the extracted prompts and the cleaned reply are unmasked at the end.
    masked, stash = _mask_code_regions(text)
    image_prompts = [
        group for m in _IMAGE_BLOCK_RE.finditer(masked) if (group := m.group(1).strip())
    ]
    cleaned = _IMAGE_BLOCK_RE.sub("", masked)
    trailing_image = _TRAILING_IMAGE_OPEN_RE.search(cleaned)
    if trailing_image is not None:
        if trailing := trailing_image.group(0)[len(IMAGE_OPEN) :].strip():
            image_prompts.append(trailing)
        cleaned = _TRAILING_IMAGE_OPEN_RE.sub("", cleaned)

    # Music blocks are pulled like image blocks (tags AND content removed) so the generation
    # prompt never shows in chat; only the first non-empty one is kept (a single clip per reply).
    music_prompt = next(
        (group for m in _MUSIC_BLOCK_RE.finditer(cleaned) if (group := m.group(1).strip())), None
    )
    cleaned = _MUSIC_BLOCK_RE.sub("", cleaned)
    trailing_music = _TRAILING_MUSIC_OPEN_RE.search(cleaned)
    if trailing_music is not None:
        if music_prompt is None:
            music_prompt = trailing_music.group(0)[len(MUSIC_OPEN) :].strip() or None
        cleaned = _TRAILING_MUSIC_OPEN_RE.sub("", cleaned)

    # Video blocks are pulled like music blocks (tags AND content removed) so the generation
    # prompt never shows in chat; only the first non-empty one is kept (a single clip per reply).
    video_prompt = next(
        (group for m in _VIDEO_BLOCK_RE.finditer(cleaned) if (group := m.group(1).strip())), None
    )
    cleaned = _VIDEO_BLOCK_RE.sub("", cleaned)
    trailing_video = _TRAILING_VIDEO_OPEN_RE.search(cleaned)
    if trailing_video is not None:
        if video_prompt is None:
            video_prompt = trailing_video.group(0)[len(VIDEO_OPEN) :].strip() or None
        cleaned = _TRAILING_VIDEO_OPEN_RE.sub("", cleaned)

    # Deep-research blocks are pulled like image blocks (tags AND content removed) so the
    # research brief never shows in chat; the first non-empty one launches the research.
    research_brief = next(
        (
            group
            for m in _DEEP_RESEARCH_BLOCK_RE.finditer(cleaned)
            if (group := m.group(1).strip())
        ),
        None,
    )
    cleaned = _DEEP_RESEARCH_BLOCK_RE.sub("", cleaned)
    trailing_research = _TRAILING_DEEP_RESEARCH_OPEN_RE.search(cleaned)
    if trailing_research is not None:
        if research_brief is None:
            research_brief = trailing_research.group(0)[len(DEEP_RESEARCH_OPEN) :].strip() or None
        cleaned = _TRAILING_DEEP_RESEARCH_OPEN_RE.sub("", cleaned)

    voice_segments = [
        segment for m in _VOICE_BLOCK_RE.finditer(cleaned) if (segment := m.group(1).strip())
    ]
    # Strip the voice tags but keep their inner content in the visible reply.
    cleaned = _VOICE_BLOCK_RE.sub(r"\1", cleaned)
    # Scrub any stray unpaired tags the model may have left behind.
    cleaned = _IMAGE_TAG_RE.sub("", cleaned)
    cleaned = _MUSIC_TAG_RE.sub("", cleaned)
    cleaned = _VIDEO_TAG_RE.sub("", cleaned)
    cleaned = _VOICE_TAG_RE.sub("", cleaned)
    cleaned = _DEEP_RESEARCH_TAG_RE.sub("", cleaned)
    # Only tidy the gap a removed block leaves behind when marker processing actually changed
    # the text, so a marker-free reply (poetry, preformatted text, an exact code/output sample)
    # keeps its intentional blank lines and surrounding whitespace byte-for-byte.
    if cleaned != masked:
        cleaned = _COLLAPSE_BLANK_LINES_RE.sub("\n\n", cleaned).strip()

    # Restore code regions in every output: the cleaned reply keeps its examples, and a rare marker
    # description that itself contained backticks gets its literal text back (never a raw sentinel).
    return InlineMarkers(
        cleaned_text=_restore_code_regions(text=cleaned, stash=stash),
        voice_text=_restore_code_regions(text="\n".join(voice_segments), stash=stash),
        voice_requested=bool(voice_segments),
        image_prompts=[
            _restore_code_regions(text=prompt, stash=stash) for prompt in image_prompts
        ],
        music_prompt=_restore_code_regions(text=music_prompt, stash=stash)
        if music_prompt
        else None,
        video_prompt=_restore_code_regions(text=video_prompt, stash=stash)
        if video_prompt
        else None,
        research_brief=(
            _restore_code_regions(text=research_brief, stash=stash) if research_brief else None
        ),
    )


def scrub_markers_for_preview(*, text: str) -> str:
    """Hides complete or still-streaming markers from a live preview snapshot.

    Complete image / music / video blocks and an unclosed trailing `<image>` / `<music>` /
    `<video>` open are removed whole (the block is going to be pulled from the reply, so it must
    never flash in). Complete voice tags are stripped but their content stays visible. A trailing
    fragment that is a prefix of any marker tag (`<imag`, `</voic`, ...) is trimmed so a
    half-streamed tag never flickers. Code regions are masked first (like `extract_inline_markers`)
    so a literal HTML `<video>` example being typed inside a code block is not scrubbed from view.
    """
    masked, stash = _mask_code_regions(text)
    cleaned = _IMAGE_BLOCK_RE.sub("", masked)
    cleaned = _TRAILING_IMAGE_OPEN_RE.sub("", cleaned)
    cleaned = _MUSIC_BLOCK_RE.sub("", cleaned)
    cleaned = _TRAILING_MUSIC_OPEN_RE.sub("", cleaned)
    cleaned = _VIDEO_BLOCK_RE.sub("", cleaned)
    cleaned = _TRAILING_VIDEO_OPEN_RE.sub("", cleaned)
    cleaned = _DEEP_RESEARCH_BLOCK_RE.sub("", cleaned)
    cleaned = _TRAILING_DEEP_RESEARCH_OPEN_RE.sub("", cleaned)
    cleaned = _VOICE_BLOCK_RE.sub(r"\1", cleaned)
    cleaned = _VOICE_TAG_RE.sub("", cleaned)
    stripped = cleaned.rstrip()
    lowered = stripped.lower()
    trimmed = next(
        (
            stripped[:-cut].rstrip()
            for tag in _ALL_TAGS
            for cut in range(len(tag) - 1, 1, -1)
            if lowered.endswith(tag[:cut].lower())
        ),
        stripped,
    )
    return _restore_code_regions(text=trimmed, stash=stash)
