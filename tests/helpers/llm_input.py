"""Structural extractors over recorded Responses API inputs.

The reply pipeline records each ``responses.create`` call's ``input`` (a
``ResponseInputParam`` list of role/content items). Tests used to assert on
these by serializing the whole list with ``str(...)`` and substring-matching a
magic sentinel, which is brittle and coupling to incidental ordering. These
helpers walk the role/content structure instead, keyed on the production block
headers and the ``[id: N]`` markers the memory blocks emit, so a test asserts on
*which user's memory reached which role* rather than on an arbitrary literal.

The block-header anchors are derived from the production renderers and the
link-source separators at import time, so a wording change in ``recall.py``
or in a ``link_sources`` module is tracked automatically rather than silently
breaking these extractors.
"""

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, cast
from importlib import import_module
from collections.abc import Mapping, Iterator, Sequence

from pydantic import Field, BaseModel
from openai.types.responses import ResponseInputParam

from discordbot.cogs.gen_reply.recall import (
    render_tone_block,
    render_server_memory_block,
    render_callable_users_block,
    render_memory_context_block,
)
from discordbot.cogs.gen_reply.context import current_header, reference_header
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES

if TYPE_CHECKING:
    from nextcord import Message


class RecordedResponses(Protocol):
    """The recording surface a fake Responses resource exposes to tests.

    Mirrors the attributes the test double accumulates per ``create`` call, so
    helpers can be typed against the recorder without importing the test module.
    """

    create_streams: list[bool]
    create_inputs: list[ResponseInputParam | str]


def _content_to_text(content: object) -> str:
    """Flattens a message item's content to plain text.

    Handles both shapes the pipeline emits: a bare string, or a list of
    ``input_text`` parts whose ``text`` fields are concatenated.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _head(text: str) -> str:
    """Returns a block text's first line, which is what identifies the block."""
    return text.split("\n", 1)[0]


def _header_line(block: Mapping[str, object]) -> str:
    """Returns the first line of a rendered block's text (its stable header)."""
    return _head(text=_content_to_text(content=block.get("content")))


_PARTICIPANT_HEADER = _header_line(block=render_memory_context_block(memories=[]))
_SERVER_HEADER = _header_line(block=render_server_memory_block(memory=""))
_TONE_HEADER = _header_line(block=render_tone_block(tone=""))
_CALLABLE_HEADER = _header_line(block=render_callable_users_block(allowed={}))

# The Reference and Current Message separators name their author, so only the text ahead of the
# name identifies them; a sentinel author marks where that text ends.
_AUTHOR_SENTINEL = "\x00"
_SENTINEL_MESSAGE = cast(
    "Message",
    SimpleNamespace(
        author=SimpleNamespace(display_name=_AUTHOR_SENTINEL, name=_AUTHOR_SENTINEL, id=0)
    ),
)
_REFERENCE_HEAD = _header_line(block=reference_header(ref=_SENTINEL_MESSAGE)).split(
    _AUTHOR_SENTINEL, 1
)[0]
_CURRENT_HEAD = _header_line(
    block=current_header(message=_SENTINEL_MESSAGE, has_reference=False)
).split(_AUTHOR_SENTINEL, 1)[0]


# Kind -> the role and leading text of the answer-input block it names.
_BLOCK_HEADS: dict[str, tuple[str, str]] = {
    "server_memory": ("assistant", _SERVER_HEADER),
    "memory": ("assistant", _PARTICIPANT_HEADER),
    "tone": ("assistant", _TONE_HEADER),
    "reference": ("system", _REFERENCE_HEAD),
    "current": ("system", _CURRENT_HEAD),
    "callable": ("system", _CALLABLE_HEADER),
}


class LinkSourceBlocks(BaseModel):
    """The top-level texts one link source injects into the answer input."""

    separators: tuple[str, ...] = Field(
        ..., description="System separators a fetched post's own block follows."
    )
    notices: tuple[str, ...] = Field(
        ..., description="Notices a failed or timed-out read injects instead of the post."
    )
    timeout_notice: str = Field(
        ..., description="The one of `notices` a build that outran the post-route grace injects."
    )


def _link_source_blocks(name: str) -> LinkSourceBlocks:
    """Reads one registered source's top-level separators and notices off its module.

    Only `====` texts count: a `----` notice is a section inside a post's own block, not a
    block of its own.
    """
    module = import_module(name=f"discordbot.cogs.gen_reply.link_sources.{name}")
    texts = {
        key: value
        for key, value in vars(module).items()
        if isinstance(value, str) and value.startswith("====")
    }
    (timeout_notice,) = (value for key, value in texts.items() if key.endswith("_TIMEOUT_NOTICE"))
    return LinkSourceBlocks(
        separators=tuple(value for key, value in texts.items() if key.endswith("_SEPARATOR")),
        notices=tuple(value for key, value in texts.items() if key.endswith("_NOTICE")),
        timeout_notice=timeout_notice,
    )


# Keyed by registry name. Read off each source's own module, since a text missing here reads as
# "no block" in every `has_link_context_block` check and would leave a negative assertion vacuous.
LINK_SOURCE_BLOCKS: dict[str, LinkSourceBlocks] = {
    source.name: _link_source_blocks(name=source.name) for source in LINK_CONTEXT_SOURCES
}

_ID_SECTION = re.compile(r"\[id: (\d+)\][^\n]*\n(.*?)(?=\n\n\[id: |\Z)", re.DOTALL)
_ID_MARKER = re.compile(r"\[id: (\d+)\]")


def iter_text_blocks(request: ResponseInputParam | str) -> Iterator[tuple[str, str]]:
    """Yields ``(role, text)`` for each role-bearing item in a recorded input."""
    if isinstance(request, str):
        return
    for item in request:
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        if isinstance(role, str):
            yield role, _content_to_text(content=item.get("content"))


def _block_text(request: ResponseInputParam | str, kind: str) -> str | None:
    """Returns the text of the first block of `kind` (a `_BLOCK_HEADS` key), or None if absent."""
    role, head = _BLOCK_HEADS[kind]
    for item_role, text in iter_text_blocks(request=request):
        if item_role == role and text.startswith(head):
            return text
    return None


def extract_memory_context_block(request: ResponseInputParam | str) -> str | None:
    """Returns the participant-memory assistant block's text, or None if absent."""
    return _block_text(request=request, kind="memory")


def has_memory_context_block(request: ResponseInputParam | str) -> bool:
    """Whether the input carries an injected participant-memory block."""
    return extract_memory_context_block(request=request) is not None


def extract_user_memory_blocks(request: ResponseInputParam | str) -> dict[int, str]:
    """Maps each injected user id to its memory body within the memory block.

    Empty when no memory block is present, so a leak check reads as
    ``user_id not in extract_user_memory_blocks(request=...)``.
    """
    block = extract_memory_context_block(request=request)
    if block is None:
        return {}
    body = block.split("\n", 1)[1] if "\n" in block else ""
    return {int(match.group(1)): match.group(2).strip() for match in _ID_SECTION.finditer(body)}


def extract_server_memory_block(request: ResponseInputParam | str) -> str | None:
    """Returns the server-memory assistant block's text, or None if absent."""
    return _block_text(request=request, kind="server_memory")


def extract_tone_block(request: ResponseInputParam | str) -> str | None:
    """Returns the tone-note assistant block's text, or None if absent."""
    return _block_text(request=request, kind="tone")


def extract_callable_user_ids(request: ResponseInputParam | str) -> set[int]:
    """Returns the ids offered for optional oblique-reference selection.

    This is the narrowed per-request allowlist boundary: it contains only absent
    public nickname-table members, never deterministic participants.
    """
    text = _block_text(request=request, kind="callable")
    return set() if text is None else {int(match) for match in _ID_MARKER.findall(text)}


def extract_link_context_block(request: ResponseInputParam | str, source: str) -> str | None:
    """Returns the text of the block following `source`'s separator, or None if absent.

    The builder emits a ``role="system"`` separator immediately followed by the
    ``role="user"`` message carrying the post's text and media; this anchors on the
    separator's header line and returns that next block's text.
    """
    separators = {_head(text=text) for text in LINK_SOURCE_BLOCKS[source].separators}
    items = list(iter_text_blocks(request=request))
    for index, (role, text) in enumerate(items):
        if role == "system" and _head(text=text) in separators:
            return items[index + 1][1] if index + 1 < len(items) else ""
    return None


def has_link_context_block(request: ResponseInputParam | str, source: str) -> bool:
    """Whether the input carries any separator or notice block of `source`."""
    blocks = LINK_SOURCE_BLOCKS[source]
    heads = {_head(text=text) for text in blocks.separators + blocks.notices}
    return any(_head(text=text) in heads for _role, text in iter_text_blocks(request=request))


def has_timeout_notice(request: ResponseInputParam | str, source: str) -> bool:
    """Whether the input carries `source`'s notice that its build outran the grace."""
    head = _head(text=LINK_SOURCE_BLOCKS[source].timeout_notice)
    return any(
        role == "system" and _head(text=text) == head
        for role, text in iter_text_blocks(request=request)
    )


def block_index(request: ResponseInputParam | str, kind: str) -> int:
    """Returns the position of the first block of `kind` among the input's role-bearing items.

    `kind` is a `_BLOCK_HEADS` key, or a link source's registry name for the separator its post
    opens with. Positions count what `iter_text_blocks` yields, so they compare with each other.

    Raises:
        AssertionError: No block of that kind is in the input.
    """
    if kind in LINK_SOURCE_BLOCKS:
        role = "system"
        heads = tuple(_head(text=text) for text in LINK_SOURCE_BLOCKS[kind].separators)
    else:
        role, head = _BLOCK_HEADS[kind]
        heads = (head,)
    for index, (item_role, text) in enumerate(iter_text_blocks(request=request)):
        if item_role == role and text.startswith(heads):
            return index
    raise AssertionError(f"no {kind} block in the request")


def request_index(responses: RecordedResponses) -> int:
    """Returns the recorded ``create`` index of the answer: the last streaming call."""
    streams = responses.create_streams
    for index in range(len(streams) - 1, -1, -1):
        if streams[index]:
            return index
    raise AssertionError("no streaming answer request was recorded")


def request_input(responses: RecordedResponses) -> ResponseInputParam | str:
    """Returns the recorded input of the answer request."""
    return responses.create_inputs[request_index(responses=responses)]
