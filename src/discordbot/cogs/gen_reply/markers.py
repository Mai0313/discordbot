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
compartment it lands in: the scope comes from the message's author and guild in `answer.py`, so a
marker body can never name one.

The tags are deliberately hyphenated (`generate-*`, like `<deep-research>`) so none collides with a
real single-word HTML / SVG / SSML element — `<video>` is HTML5, `<image>` is SVG, `<voice>` is
SSML — which would otherwise let a reply that merely SHOWS such example markup be mistaken for a
generation request and stripped from the reply.

Every tag's three patterns (the complete block, the bare tag, the unclosed trailing open) are
derived from its name by `_Marker` rather than written out per tag: they only ever differed by
that name, and eight hand-written triples is eight places for one of them to be spelled wrong.
"""

import re

from pydantic import Field, BaseModel

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


class _Marker:
    """One marker tag plus the three patterns every marker is read through.

    The block pattern is non-greedy and DOTALL so a multi-line body is captured whole, and every
    pattern is IGNORECASE because a stray-cased tag still has to be caught — a defusing or
    extraction pass stricter than the model's own output is no pass at all.
    """

    def __init__(self, name: str) -> None:
        self.open = f"<{name}>"
        self.close = f"</{name}>"
        self.block = re.compile(rf"<{name}>(.*?)</{name}>", re.IGNORECASE | re.DOTALL)
        # A bare, unpaired tag, scrubbed so it never leaks into the visible reply.
        self.tag = re.compile(rf"</?{name}>", re.IGNORECASE)
        # An unclosed open tag and everything after it: the whole block is going to be pulled,
        # so hide it the moment it starts streaming in (and tolerate a model that never closes).
        self.trailing = re.compile(rf"<{name}>.*\Z", re.IGNORECASE | re.DOTALL)

    def pull(self, text: str) -> tuple[list[str], str]:
        """Removes this marker's blocks from `text`, returning their bodies and what is left.

        Bodies come back in the order the model wrote them, with an unclosed trailing open taken
        as one last body so its raw description never leaks. Empty bodies are dropped.
        """
        bodies = [body for match in self.block.finditer(text) if (body := match.group(1).strip())]
        cleaned = self.block.sub("", text)
        trailing = self.trailing.search(cleaned)
        if trailing is not None:
            if body := trailing.group(0)[len(self.open) :].strip():
                bodies.append(body)
            cleaned = self.trailing.sub("", cleaned)
        return bodies, cleaned

    def hide(self, text: str) -> str:
        """Removes complete blocks and an unclosed trailing open, for a live preview."""
        return self.trailing.sub("", self.block.sub("", text))


_VOICE = _Marker("generate-voice")
_IMAGE = _Marker("generate-image")
_MUSIC = _Marker("generate-music")
_VIDEO = _Marker("generate-video")
_DEEP_RESEARCH = _Marker("deep-research")
_WRITE_MEMORY = _Marker("write-memory")
_FORGET_MEMORY = _Marker("forget-memory")
_WRITE_SERVER_MEMORY = _Marker("write-server-memory")

# Tag literals are the single source of truth shared by the prompt instructions and this parser.
VOICE_OPEN = _VOICE.open
VOICE_CLOSE = _VOICE.close
IMAGE_OPEN = _IMAGE.open
IMAGE_CLOSE = _IMAGE.close
MUSIC_OPEN = _MUSIC.open
MUSIC_CLOSE = _MUSIC.close
VIDEO_OPEN = _VIDEO.open
VIDEO_CLOSE = _VIDEO.close
DEEP_RESEARCH_OPEN = _DEEP_RESEARCH.open
DEEP_RESEARCH_CLOSE = _DEEP_RESEARCH.close
WRITE_MEMORY_OPEN = _WRITE_MEMORY.open
WRITE_MEMORY_CLOSE = _WRITE_MEMORY.close
FORGET_MEMORY_OPEN = _FORGET_MEMORY.open
FORGET_MEMORY_CLOSE = _FORGET_MEMORY.close
WRITE_SERVER_MEMORY_OPEN = _WRITE_SERVER_MEMORY.open
WRITE_SERVER_MEMORY_CLOSE = _WRITE_SERVER_MEMORY.close

# Every marker whose block is PULLED from the reply, in extraction order. Voice is not here: its
# content stays visible and only the tags come off, which is the one asymmetry in this module.
_PULLED = (
    _IMAGE,
    _MUSIC,
    _VIDEO,
    _DEEP_RESEARCH,
    _WRITE_MEMORY,
    _FORGET_MEMORY,
    _WRITE_SERVER_MEMORY,
)

# Every tag, for the half-streamed-tail trim in a live preview.
_ALL_TAGS = tuple(
    literal for marker in (*_PULLED, _VOICE) for literal in (marker.open, marker.close)
)

_COLLAPSE_BLANK_LINES_RE = re.compile(r"\n{3,}")


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
    kept (one clip per reply by design), as is the first `<deep-research>` brief. Voice tags are
    stripped but their inner content STAYS in the visible reply, and every wrapped segment is
    concatenated as the spoken-clip input. An unclosed trailing open (the model forgot to close
    it) is still pulled so its raw description never leaks, and any stray unpaired tag is
    scrubbed. The three memory tags are pulled like image blocks, each keeping up to
    `MAX_MEMORY_NOTES` notes in the order they were written — which is the order the evaluator
    downstream reads them in.
    """
    cleaned = text
    pulled: dict[str, list[str]] = {}
    for marker in _PULLED:
        pulled[marker.open], cleaned = marker.pull(cleaned)

    # Complete blocks only, unlike every pulled marker: an unclosed `<generate-voice>` leaves its
    # content visible (the bare tag is scrubbed below), so speaking it would put words in the clip
    # that the reply never marked as spoken.
    voice_segments = [
        segment for match in _VOICE.block.finditer(cleaned) if (segment := match.group(1).strip())
    ]
    # Strip the voice tags but keep their inner content in the visible reply.
    cleaned = _VOICE.block.sub(r"\1", cleaned)
    # Scrub any stray unpaired tags the model may have left behind.
    for marker in (*_PULLED, _VOICE):
        cleaned = marker.tag.sub("", cleaned)
    # Only tidy the gap a removed block leaves behind when marker processing actually changed
    # the text, so a marker-free reply (poetry, preformatted text, an exact code/output sample)
    # keeps its intentional blank lines and surrounding whitespace byte-for-byte.
    if cleaned != text:
        cleaned = _COLLAPSE_BLANK_LINES_RE.sub("\n\n", cleaned).strip()

    return InlineMarkers(
        cleaned_text=cleaned,
        voice_text="\n".join(voice_segments),
        voice_requested=bool(voice_segments),
        image_prompts=pulled[IMAGE_OPEN],
        music_prompt=next(iter(pulled[MUSIC_OPEN]), None),
        video_prompt=next(iter(pulled[VIDEO_OPEN]), None),
        research_brief=next(iter(pulled[DEEP_RESEARCH_OPEN]), None),
        memory_notes=pulled[WRITE_MEMORY_OPEN][:MAX_MEMORY_NOTES],
        forget_notes=pulled[FORGET_MEMORY_OPEN][:MAX_MEMORY_NOTES],
        server_memory_notes=pulled[WRITE_SERVER_MEMORY_OPEN][:MAX_MEMORY_NOTES],
    )


def scrub_markers_for_preview(*, text: str) -> str:
    """Hides complete or still-streaming markers from a live preview snapshot.

    Complete image / music / video / research / memory blocks and their unclosed trailing opens
    are removed whole (the block is going to be pulled from the reply, so it must never flash
    in). Complete voice tags are stripped but their content stays visible. A trailing fragment
    that is a prefix of any marker tag (`<generate-imag`, `</generate-voic`, ...) is trimmed so a
    half-streamed tag never flickers.
    """
    cleaned = text
    for marker in _PULLED:
        cleaned = marker.hide(cleaned)
    cleaned = _VOICE.tag.sub("", _VOICE.block.sub(r"\1", cleaned))
    stripped = cleaned.rstrip()
    lowered = stripped.lower()
    for tag in _ALL_TAGS:
        for cut in range(len(tag) - 1, 1, -1):
            if lowered.endswith(tag[:cut].lower()):
                return stripped[:-cut].rstrip()
    return stripped
