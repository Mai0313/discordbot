"""Inline reply markers: the answer model self-selects spoken segments, images, music, video, and what to remember.

The answer model wraps the parts of its reply it wants read aloud in `<generate-voice>...
</generate-voice>`: only those segments are synthesized (concatenated into a single voice clip),
but they STAY in the visible reply (only the tags are stripped). It may also wrap short
descriptions in `<generate-image>...</generate-image>` to have images generated and attached, one
`<generate-music>...</generate-music>` description to have a music clip generated and attached, or
one `<generate-video>...</generate-video>` description to have a short video generated and
attached, or one `<deep-research>...</deep-research>` brief to launch a research thread; each such
block (tags AND content) is REMOVED from the visible reply so the generation prompt never leaks
into chat. `ResponseStreamer` extracts them at finalize time via `extract_inline_markers` and
scrubs partial/complete tags from the live preview via `scrub_markers_for_preview`, so none flickers
mid-stream. The asymmetry is deliberate: voice content is meant to stay visible, image / music /
video content are meant to be pulled.

The memory tags (`<write-memory>`, `<forget-memory>`, `<write-server-memory>`) are pulled the same
way and carry one plain sentence each: what to remember about the person being replied to, what
they no longer want remembered, or what to remember about the community. They are the answer
model's half of the memory write path, replacing the separate extraction pass that used to re-read
the conversation afterwards (#596). Nothing here decides WHOSE memory is written or which
compartment it lands in: the scope comes from the message's author and guild in `cog.py`, so a
marker body can never name one.

The tags are deliberately hyphenated (`generate-*`, like `<deep-research>`) so none collides with a
real single-word HTML / SVG / SSML element — `<video>` is HTML5, `<image>` is SVG, `<voice>` is
SSML — which would otherwise let a reply that merely SHOWS such example markup be mistaken for a
generation request and stripped from the reply.
"""

import re

from pydantic import Field, BaseModel

# Tag literals are the single source of truth shared by the prompt instructions and this parser.
VOICE_OPEN = "<generate-voice>"
VOICE_CLOSE = "</generate-voice>"
IMAGE_OPEN = "<generate-image>"
IMAGE_CLOSE = "</generate-image>"
MUSIC_OPEN = "<generate-music>"
MUSIC_CLOSE = "</generate-music>"
VIDEO_OPEN = "<generate-video>"
VIDEO_CLOSE = "</generate-video>"
DEEP_RESEARCH_OPEN = "<deep-research>"
DEEP_RESEARCH_CLOSE = "</deep-research>"
WRITE_MEMORY_OPEN = "<write-memory>"
WRITE_MEMORY_CLOSE = "</write-memory>"
FORGET_MEMORY_OPEN = "<forget-memory>"
FORGET_MEMORY_CLOSE = "</forget-memory>"
WRITE_SERVER_MEMORY_OPEN = "<write-server-memory>"
WRITE_SERVER_MEMORY_CLOSE = "</write-server-memory>"

# Hard cap on inline images per reply: a voice clip plus 9 images exactly fills Discord's
# 10-attachment ceiling. The prompt tells the model this limit; the streamer enforces it by
# dropping any extra blocks so a confused model never blows past the attachment cap. A reply
# may also carry one music clip and one video clip (each single per reply by design), so a rare
# voice + music + video + 9 images would be 12 attachments; `MediaDeliveryPlanner.plan`'s
# attachment-count clamp is the backstop, dropping the trailing overflow.
MAX_INLINE_IMAGES = 9

# Hard cap on memory notes of one kind per reply. Unlike the image cap this is not a Discord
# limit but a sanity bound: a turn worth remembering produces one or two notes, and a model that
# emits twenty has misread the instruction rather than found twenty durable facts. Extra blocks
# are dropped, and the evaluator downstream still decides whether any of the kept ones survive.
MAX_MEMORY_NOTES = 5

# Complete blocks: non-greedy, DOTALL so a multi-line segment is captured, IGNORECASE so a
# stray-cased tag still matches.
_VOICE_BLOCK_RE = re.compile(r"<generate-voice>(.*?)</generate-voice>", re.IGNORECASE | re.DOTALL)
_IMAGE_BLOCK_RE = re.compile(r"<generate-image>(.*?)</generate-image>", re.IGNORECASE | re.DOTALL)
_MUSIC_BLOCK_RE = re.compile(r"<generate-music>(.*?)</generate-music>", re.IGNORECASE | re.DOTALL)
_VIDEO_BLOCK_RE = re.compile(r"<generate-video>(.*?)</generate-video>", re.IGNORECASE | re.DOTALL)
_DEEP_RESEARCH_BLOCK_RE = re.compile(
    r"<deep-research>(.*?)</deep-research>", re.IGNORECASE | re.DOTALL
)
_WRITE_MEMORY_BLOCK_RE = re.compile(
    r"<write-memory>(.*?)</write-memory>", re.IGNORECASE | re.DOTALL
)
_FORGET_MEMORY_BLOCK_RE = re.compile(
    r"<forget-memory>(.*?)</forget-memory>", re.IGNORECASE | re.DOTALL
)
_WRITE_SERVER_MEMORY_BLOCK_RE = re.compile(
    r"<write-server-memory>(.*?)</write-server-memory>", re.IGNORECASE | re.DOTALL
)
# Bare tags, scrubbed so a stray/unpaired tag never leaks into the visible reply.
_VOICE_TAG_RE = re.compile(r"</?generate-voice>", re.IGNORECASE)
_IMAGE_TAG_RE = re.compile(r"</?generate-image>", re.IGNORECASE)
_MUSIC_TAG_RE = re.compile(r"</?generate-music>", re.IGNORECASE)
_VIDEO_TAG_RE = re.compile(r"</?generate-video>", re.IGNORECASE)
_DEEP_RESEARCH_TAG_RE = re.compile(r"</?deep-research>", re.IGNORECASE)
_WRITE_MEMORY_TAG_RE = re.compile(r"</?write-memory>", re.IGNORECASE)
_FORGET_MEMORY_TAG_RE = re.compile(r"</?forget-memory>", re.IGNORECASE)
_WRITE_SERVER_MEMORY_TAG_RE = re.compile(r"</?write-server-memory>", re.IGNORECASE)
# An unclosed open tag and everything after it: the whole block is going to be pulled, so hide it
# the moment it starts streaming in (and tolerate the model forgetting to close it).
_TRAILING_IMAGE_OPEN_RE = re.compile(r"<generate-image>.*\Z", re.IGNORECASE | re.DOTALL)
_TRAILING_MUSIC_OPEN_RE = re.compile(r"<generate-music>.*\Z", re.IGNORECASE | re.DOTALL)
_TRAILING_VIDEO_OPEN_RE = re.compile(r"<generate-video>.*\Z", re.IGNORECASE | re.DOTALL)
_TRAILING_DEEP_RESEARCH_OPEN_RE = re.compile(r"<deep-research>.*\Z", re.IGNORECASE | re.DOTALL)
_TRAILING_WRITE_MEMORY_OPEN_RE = re.compile(r"<write-memory>.*\Z", re.IGNORECASE | re.DOTALL)
_TRAILING_FORGET_MEMORY_OPEN_RE = re.compile(r"<forget-memory>.*\Z", re.IGNORECASE | re.DOTALL)
_TRAILING_WRITE_SERVER_MEMORY_OPEN_RE = re.compile(
    r"<write-server-memory>.*\Z", re.IGNORECASE | re.DOTALL
)
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
    WRITE_MEMORY_OPEN,
    WRITE_MEMORY_CLOSE,
    FORGET_MEMORY_OPEN,
    FORGET_MEMORY_CLOSE,
    WRITE_SERVER_MEMORY_OPEN,
    WRITE_SERVER_MEMORY_CLOSE,
)


class InlineMarkers(BaseModel):
    """Markers extracted from a finished reply: the visible text plus its media requests."""

    cleaned_text: str = Field(
        ...,
        description="Reply text with image blocks removed and voice tags stripped (voice content kept).",
    )
    voice_text: str = Field(
        default="",
        description="Concatenated <generate-voice> segments to synthesize aloud; empty when none.",
    )
    voice_requested: bool = Field(
        default=False, description="Whether the reply wrapped any segment in <generate-voice>."
    )
    image_prompts: list[str] = Field(
        default_factory=list,
        description="Every <generate-image> description to generate, in order; empty when none.",
    )
    music_prompt: str | None = Field(
        default=None,
        description="First <generate-music> description to generate a single clip, or None when absent.",
    )
    video_prompt: str | None = Field(
        default=None,
        description="First <generate-video> description to generate a single clip, or None when absent.",
    )
    research_brief: str | None = Field(
        default=None,
        description="First <deep-research> brief to launch a research thread, or None when absent.",
    )
    memory_notes: list[str] = Field(
        default_factory=list,
        description="Every <write-memory> note about the message author, in order; empty when none.",
        examples=[["使用者希望回覆用繁體中文"]],
    )
    forget_notes: list[str] = Field(
        default_factory=list,
        description="Every <forget-memory> note naming what the author no longer wants remembered.",
        examples=[["使用者已經不住台中了"]],
    )
    server_memory_notes: list[str] = Field(
        default_factory=list,
        description="Every <write-server-memory> note about the community, in order; empty when none.",
    )


def extract_inline_markers(*, text: str) -> InlineMarkers:
    """Splits a finished reply into visible text plus its voice / image / music / video requests.

    Image blocks (tags AND content) are removed entirely so the generation prompt never shows
    in chat; every non-empty one becomes an image request, in order. A `<generate-music>` and a
    `<generate-video>` block are pulled the same way, but only the first non-empty one of each is
    kept (one clip per reply by design). Voice tags are stripped but their inner content STAYS in
    the visible reply, and every wrapped segment is concatenated as the spoken-clip input. An
    unclosed trailing `<generate-image>` / `<generate-music>` / `<generate-video>` (the model forgot
    to close it) is still pulled so its raw description never leaks, and any stray unpaired tag is
    scrubbed. The three memory tags are pulled the same way as image blocks, each keeping up to
    `MAX_MEMORY_NOTES` notes in the order they were written.
    """
    image_prompts = [
        group for m in _IMAGE_BLOCK_RE.finditer(text) if (group := m.group(1).strip())
    ]
    cleaned = _IMAGE_BLOCK_RE.sub("", text)
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

    # Memory notes are pulled like image blocks: the note is instruction to the memory pipeline,
    # never something the reader should see, and a reply that narrates what it just recorded reads
    # as the bot talking about itself instead of answering.
    memory_notes, cleaned = _pull_notes(
        text=cleaned,
        block_re=_WRITE_MEMORY_BLOCK_RE,
        trailing_re=_TRAILING_WRITE_MEMORY_OPEN_RE,
        open_tag=WRITE_MEMORY_OPEN,
    )
    forget_notes, cleaned = _pull_notes(
        text=cleaned,
        block_re=_FORGET_MEMORY_BLOCK_RE,
        trailing_re=_TRAILING_FORGET_MEMORY_OPEN_RE,
        open_tag=FORGET_MEMORY_OPEN,
    )
    server_memory_notes, cleaned = _pull_notes(
        text=cleaned,
        block_re=_WRITE_SERVER_MEMORY_BLOCK_RE,
        trailing_re=_TRAILING_WRITE_SERVER_MEMORY_OPEN_RE,
        open_tag=WRITE_SERVER_MEMORY_OPEN,
    )

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
    cleaned = _WRITE_SERVER_MEMORY_TAG_RE.sub("", cleaned)
    cleaned = _WRITE_MEMORY_TAG_RE.sub("", cleaned)
    cleaned = _FORGET_MEMORY_TAG_RE.sub("", cleaned)
    # Only tidy the gap a removed block leaves behind when marker processing actually changed
    # the text, so a marker-free reply (poetry, preformatted text, an exact code/output sample)
    # keeps its intentional blank lines and surrounding whitespace byte-for-byte.
    if cleaned != text:
        cleaned = _COLLAPSE_BLANK_LINES_RE.sub("\n\n", cleaned).strip()

    return InlineMarkers(
        cleaned_text=cleaned,
        voice_text="\n".join(voice_segments),
        voice_requested=bool(voice_segments),
        image_prompts=image_prompts,
        music_prompt=music_prompt,
        video_prompt=video_prompt,
        research_brief=research_brief,
        memory_notes=memory_notes,
        forget_notes=forget_notes,
        server_memory_notes=server_memory_notes,
    )


def _pull_notes(
    text: str, block_re: re.Pattern[str], trailing_re: re.Pattern[str], open_tag: str
) -> tuple[list[str], str]:
    """Pulls one kind of memory note out of a reply, returning the notes and what is left.

    Blocks are removed whole, an unclosed trailing open is taken as one last note (the model
    forgot to close it, and its body must not leak), and the result is capped at
    `MAX_MEMORY_NOTES`. Order is the order the model wrote them in, which is the order the
    evaluator downstream reads them in.
    """
    notes = [group for match in block_re.finditer(text) if (group := match.group(1).strip())]
    cleaned = block_re.sub("", text)
    trailing = trailing_re.search(cleaned)
    if trailing is not None:
        if note := trailing.group(0)[len(open_tag) :].strip():
            notes.append(note)
        cleaned = trailing_re.sub("", cleaned)
    return notes[:MAX_MEMORY_NOTES], cleaned


def scrub_markers_for_preview(*, text: str) -> str:
    """Hides complete or still-streaming markers from a live preview snapshot.

    Complete image / music / video / memory blocks and an unclosed trailing `<generate-image>` /
    `<generate-music>` / `<generate-video>` / `<write-memory>` open are removed whole (the block is
    going to be pulled from the reply, so it must never flash in). Complete voice tags are stripped
    but their content stays visible. A trailing fragment that is a prefix of any marker tag
    (`<generate-imag`, `</generate-voic`, ...) is trimmed so a half-streamed tag never flickers.
    """
    cleaned = _IMAGE_BLOCK_RE.sub("", text)
    cleaned = _TRAILING_IMAGE_OPEN_RE.sub("", cleaned)
    cleaned = _MUSIC_BLOCK_RE.sub("", cleaned)
    cleaned = _TRAILING_MUSIC_OPEN_RE.sub("", cleaned)
    cleaned = _VIDEO_BLOCK_RE.sub("", cleaned)
    cleaned = _TRAILING_VIDEO_OPEN_RE.sub("", cleaned)
    cleaned = _DEEP_RESEARCH_BLOCK_RE.sub("", cleaned)
    cleaned = _TRAILING_DEEP_RESEARCH_OPEN_RE.sub("", cleaned)
    cleaned = _WRITE_SERVER_MEMORY_BLOCK_RE.sub("", cleaned)
    cleaned = _TRAILING_WRITE_SERVER_MEMORY_OPEN_RE.sub("", cleaned)
    cleaned = _WRITE_MEMORY_BLOCK_RE.sub("", cleaned)
    cleaned = _TRAILING_WRITE_MEMORY_OPEN_RE.sub("", cleaned)
    cleaned = _FORGET_MEMORY_BLOCK_RE.sub("", cleaned)
    cleaned = _TRAILING_FORGET_MEMORY_OPEN_RE.sub("", cleaned)
    cleaned = _VOICE_BLOCK_RE.sub(r"\1", cleaned)
    cleaned = _VOICE_TAG_RE.sub("", cleaned)
    stripped = cleaned.rstrip()
    lowered = stripped.lower()
    for tag in _ALL_TAGS:
        for cut in range(len(tag) - 1, 1, -1):
            if lowered.endswith(tag[:cut].lower()):
                return stripped[:-cut].rstrip()
    return stripped
