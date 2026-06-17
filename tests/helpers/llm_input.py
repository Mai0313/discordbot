"""Structural extractors over recorded Responses API inputs.

The reply pipeline records each ``responses.create`` call's ``input`` (a
``ResponseInputParam`` list of role/content items). Tests used to assert on
these by serializing the whole list with ``str(...)`` and substring-matching a
magic sentinel, which is brittle and coupling to incidental ordering. These
helpers walk the role/content structure instead, keyed on the production block
headers and the ``[id: N]`` markers the memory blocks emit, so a test asserts on
*which user's memory reached which role* rather than on an arbitrary literal.

The block-header anchors are derived from the production renderers at import
time, so a wording change in ``memory_tool.py`` is tracked automatically rather
than silently breaking these extractors.
"""

import re
from typing import Literal, Protocol
from collections.abc import Mapping, Iterator, Sequence

from openai.types.responses import ResponseInputParam

from discordbot.cogs._gen_reply.memory_tool import (
    render_server_memory_block,
    render_callable_users_block,
    render_memory_context_block,
)
from discordbot.cogs._parse_threads.builder import (
    THREADS_TIMEOUT_NOTICE,
    THREADS_CONTEXT_SEPARATOR,
    THREADS_UNAVAILABLE_NOTICE,
    THREADS_TEXT_ONLY_SEPARATOR,
)


class RecordedResponses(Protocol):
    """The recording surface a fake Responses resource exposes to tests.

    Mirrors the attributes the test double accumulates per ``create`` call, so
    helpers can be typed against the recorder without importing the test module.
    """

    create_streams: list[bool]
    create_tools: list[list[object] | None]
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


def _header_line(block: Mapping[str, object]) -> str:
    """Returns the first line of a rendered block's text (its stable header)."""
    return _content_to_text(content=block.get("content")).split("\n", 1)[0]


_PARTICIPANT_HEADER = _header_line(block=render_memory_context_block(memories=[]))
_SERVER_HEADER = _header_line(block=render_server_memory_block(memory=""))
_CALLABLE_HEADER = _header_line(block=render_callable_users_block(allowed={}))
_THREADS_SEPARATOR_HEADS = (
    THREADS_CONTEXT_SEPARATOR.split("\n", 1)[0],
    THREADS_TEXT_ONLY_SEPARATOR.split("\n", 1)[0],
)
_THREADS_NOTICE_HEADS = (
    THREADS_UNAVAILABLE_NOTICE.split("\n", 1)[0],
    THREADS_TIMEOUT_NOTICE.split("\n", 1)[0],
)

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


def extract_memory_context_block(request: ResponseInputParam | str) -> str | None:
    """Returns the participant-memory assistant block's text, or None if absent."""
    for role, text in iter_text_blocks(request=request):
        if role == "assistant" and text.split("\n", 1)[0] == _PARTICIPANT_HEADER:
            return text
    return None


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
    for role, text in iter_text_blocks(request=request):
        if role == "assistant" and text.split("\n", 1)[0] == _SERVER_HEADER:
            return text
    return None


def extract_callable_user_ids(request: ResponseInputParam | str) -> set[int]:
    """Returns the ids the selection phase offered in its callable-users block.

    This is the per-request allowlist boundary: a private channel that does not
    widen via the nickname table yields only the conversation participants here.
    """
    for role, text in iter_text_blocks(request=request):
        if role == "system" and text.split("\n", 1)[0] == _CALLABLE_HEADER:
            return {int(match) for match in _ID_MARKER.findall(text)}
    return set()


def extract_threads_context_block(request: ResponseInputParam | str) -> str | None:
    """Returns the text of the block following the Threads separator, or None if absent.

    The builder emits a ``role="system"`` separator immediately followed by the
    ``role="user"`` message carrying the post's text and media; this anchors on the
    separator's header line and returns that next block's text.
    """
    items = list(iter_text_blocks(request=request))
    for index, (role, text) in enumerate(items):
        if role == "system" and text.split("\n", 1)[0] in _THREADS_SEPARATOR_HEADS:
            return items[index + 1][1] if index + 1 < len(items) else ""
    return None


def has_threads_context_block(request: ResponseInputParam | str) -> bool:
    """Whether the input carries an injected Threads separator or notice block."""
    for _role, text in iter_text_blocks(request=request):
        head = text.split("\n", 1)[0]
        if head in _THREADS_SEPARATOR_HEADS or head in _THREADS_NOTICE_HEADS:
            return True
    return False


def tool_names_for_call(responses: RecordedResponses, n: int) -> list[str]:
    """Returns the tool names offered on the nth recorded ``create`` call."""
    names: list[str] = []
    for tool in responses.create_tools[n] or []:
        if isinstance(tool, Mapping):
            name = tool.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def request_index(responses: RecordedResponses, phase: Literal["selection", "answer"]) -> int:
    """Maps a semantic pipeline phase to its recorded ``create`` index.

    Selection is the first non-streaming call; the answer is the last streaming
    call. Lets tests reference phases by name instead of hardcoding positions.
    """
    streams = responses.create_streams
    if phase == "selection":
        return streams.index(False)
    for index in range(len(streams) - 1, -1, -1):
        if streams[index]:
            return index
    raise AssertionError("no streaming answer request was recorded")


def request_input(
    responses: RecordedResponses, phase: Literal["selection", "answer"]
) -> ResponseInputParam | str:
    """Returns the recorded input for a semantic pipeline phase."""
    return responses.create_inputs[request_index(responses=responses, phase=phase)]
