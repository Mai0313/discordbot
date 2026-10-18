"""Inline reply markers for spoken segments, media generation, research, and memory notes.

The answer model wraps segments to read aloud in `<generate-voice>...</generate-voice>`: only
those segments are synthesized into a single voice clip, and their inner text stays visible in
the reply with only the tags stripped. By contrast, `<generate-image>`, `<generate-music>`,
`<generate-video>`, `<deep-research>`, `<write-memory>`, `<forget-memory>`, and
`<write-server-memory>` blocks are pulled in full (tags and content) so generation prompts,
research briefs, and memory notes never appear in chat. Marker bodies carry no target scope:
whose memory or server a note writes to is determined by the caller from the turn's message.

Tags are hyphenated (`generate-*`, `<deep-research>`) so they never collide with single-word
HTML, SVG, or SSML elements such as `<video>`, `<image>`, or `<voice>` in example markup.
Every marker is registered through `_declare` so extraction, preview scrubbing, and tag-name
exports cover the same set.
"""

import re
from typing import Final
from functools import cached_property

from pydantic import Field, BaseModel

# Hard cap on inline images per reply: a voice clip plus 9 images fills Discord's 10-attachment
# ceiling; `MediaDeliveryPlanner.plan` clamps the rare reply that also adds music or video.
MAX_INLINE_IMAGES = 9

# Sanity cap on memory notes of one kind per reply before downstream review.
MAX_MEMORY_NOTES = 5


class _Marker(BaseModel):
    """One marker tag and its compiled regex patterns.

    Block matching is non-greedy across newlines, and all patterns are case-insensitive so
    stray-cased model output is still caught.
    """

    name: str = Field(
        ..., description="The tag name the model writes.", examples=["generate-image"]
    )

    @property
    def open(self) -> str:
        """The opening tag."""
        return f"<{self.name}>"

    @property
    def close(self) -> str:
        """The closing tag."""
        return f"</{self.name}>"

    @cached_property
    def block(self) -> re.Pattern[str]:
        """A complete block, capturing its body."""
        return re.compile(rf"<{self.name}>(.*?)</{self.name}>", re.IGNORECASE | re.DOTALL)

    @cached_property
    def tag(self) -> re.Pattern[str]:
        """A bare, unpaired tag, scrubbed so it never leaks into the visible reply."""
        return re.compile(rf"</?{self.name}>", re.IGNORECASE)

    @cached_property
    def trailing(self) -> re.Pattern[str]:
        """An unclosed opening tag and everything after it, hidden once it starts streaming."""
        return re.compile(rf"<{self.name}>.*\Z", re.IGNORECASE | re.DOTALL)

    def pull(self, text: str) -> tuple[list[str], str]:
        """Removes this marker's blocks from `text`, returning non-empty bodies and the remaining text.

        An unclosed trailing open tag is treated as a final block so its raw body never leaks.
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


_MARKERS: Final[list[_Marker]] = []


def _declare(name: str) -> _Marker:
    """Declares one marker, registering it so a pass over every marker cannot miss it."""
    marker = _Marker(name=name)
    _MARKERS.append(marker)
    return marker


_VOICE = _declare(name="generate-voice")
_IMAGE = _declare(name="generate-image")
_MUSIC = _declare(name="generate-music")
_VIDEO = _declare(name="generate-video")
_DEEP_RESEARCH = _declare(name="deep-research")
_WRITE_MEMORY = _declare(name="write-memory")
_FORGET_MEMORY = _declare(name="forget-memory")
_WRITE_SERVER_MEMORY = _declare(name="write-server-memory")

MARKER_TAG_NAMES: Final[tuple[str, ...]] = tuple(marker.name for marker in _MARKERS)

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

# Markers whose entire block is removed from the reply; voice keeps its inner text visible.
_PULLED = tuple(marker for marker in _MARKERS if marker is not _VOICE)

_ALL_TAGS = tuple(
    literal for marker in (*_PULLED, _VOICE) for literal in (marker.open, marker.close)
)

_COLLAPSE_BLANK_LINES_RE = re.compile(r"\n{3,}")


class InlineMarkers(BaseModel):
    """Markers extracted from a finished reply: the visible text plus its media requests."""

    cleaned_text: str = Field(
        ...,
        description=(
            "Reply text with every pulled block removed and voice tags stripped "
            "(voice content kept)."
        ),
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
        description=(
            "The <generate-image> descriptions to generate, in order, up to MAX_INLINE_IMAGES; "
            "empty when none."
        ),
    )
    image_requests: int = Field(
        default=0,
        description="How many <generate-image> descriptions the reply wrote, past the cap too.",
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

    @property
    def media_requested(self) -> bool:
        """Returns whether the reply asked for any spoken clip, image, song, or video."""
        return bool(
            self.voice_requested or self.image_prompts or self.music_prompt or self.video_prompt
        )


def extract_inline_markers(text: str) -> InlineMarkers:
    """Returns `text` split into visible reply text and extracted inline marker payloads.

    Pulled blocks (`<generate-image>`, `<generate-music>`, `<generate-video>`, `<deep-research>`,
    and the memory tags) are removed completely, including unclosed trailing opens. Image prompts
    are capped at `MAX_INLINE_IMAGES`, memory notes at `MAX_MEMORY_NOTES` per kind, and music,
    video, and research keep only their first non-empty block. `<generate-voice>` tags are
    stripped while leaving their inner text in place, and only complete voice blocks are
    concatenated into `voice_text`. Stray unpaired tags are scrubbed.

    Args:
        text (str): Raw reply text from the answer model.
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
    cleaned = _VOICE.block.sub(r"\1", cleaned)
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
        image_prompts=pulled[IMAGE_OPEN][:MAX_INLINE_IMAGES],
        image_requests=len(pulled[IMAGE_OPEN]),
        music_prompt=next(iter(pulled[MUSIC_OPEN]), None),
        video_prompt=next(iter(pulled[VIDEO_OPEN]), None),
        research_brief=next(iter(pulled[DEEP_RESEARCH_OPEN]), None),
        memory_notes=pulled[WRITE_MEMORY_OPEN][:MAX_MEMORY_NOTES],
        forget_notes=pulled[FORGET_MEMORY_OPEN][:MAX_MEMORY_NOTES],
        server_memory_notes=pulled[WRITE_SERVER_MEMORY_OPEN][:MAX_MEMORY_NOTES],
    )


def scrub_markers_for_preview(text: str) -> str:
    """Returns `text` with complete, unclosed, or partially streamed markers hidden for preview.

    Pulled blocks and their unclosed trailing opens are removed in `_PULLED` order to match
    `extract_inline_markers`, `<generate-voice>` tags are stripped while keeping their inner text,
    and any trailing prefix of a known marker tag is trimmed.

    Args:
        text (str): Accumulated streaming reply text.
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
