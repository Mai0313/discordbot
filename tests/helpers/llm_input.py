"""Structural extractors over recorded Responses API inputs.

The reply pipeline records each ``responses.create`` call's ``input`` (a
``ResponseInputParam`` list of role/content items). Tests used to assert on
these by serializing the whole list with ``str(...)`` and substring-matching a
magic sentinel, which is brittle and coupling to incidental ordering. These
helpers walk the role/content structure instead, keyed on the production block
headers and the ``[id: N]`` markers the memory blocks emit, so a test asserts on
*which user's memory reached which role* rather than on an arbitrary literal.

Three groups live here. ``iter_text_blocks`` flattens one recorded input into
``(role, text)`` pairs and everything else is built on it. The block extractors
answer what the pipeline injected and where: participant memory keyed by user
id, server memory, the tone note, the narrowed oblique-reference allowlist, and
each link source's context block. ``request_index`` / ``request_input`` /
``tool_names_for_call`` read the recorder itself, so a test names a pipeline
phase instead of a position in the call log.

Two conventions run through the link-source pair. ``extract_*_context_block``
anchors on the separator and returns the block AFTER it, because the builder
emits the framing as its own ``role="system"`` item and the post's text and
media as the ``role="user"`` item behind it; ``has_*_context_block`` matches the
failure notices as well, which ``extract_*`` deliberately does not, so a test
can assert the pipeline said something about the link without caring whether the
fetch worked.

The anchors are all derived at import time from the production strings, the
renderers in ``memory_tool.py`` and the separators and notices in
``link_sources/``, so a wording change there is tracked automatically rather
than silently turning every assertion built on it into a no-op.
"""

import re
from typing import Literal, Protocol
from collections.abc import Mapping, Iterator, Sequence

from openai.types.responses import ResponseInputParam

from discordbot.cogs.gen_reply.memory_tool import (
    render_tone_block,
    render_server_memory_block,
    render_callable_users_block,
    render_memory_context_block,
)
from discordbot.cogs.gen_reply.link_sources.douyin import (
    DOUYIN_BLOCKED_NOTICE,
    DOUYIN_TIMEOUT_NOTICE,
    DOUYIN_CONTEXT_SEPARATOR,
    DOUYIN_UNAVAILABLE_NOTICE,
    DOUYIN_TEXT_ONLY_SEPARATOR,
)
from discordbot.cogs.gen_reply.link_sources.threads import (
    THREADS_TIMEOUT_NOTICE,
    THREADS_CONTEXT_SEPARATOR,
    THREADS_UNAVAILABLE_NOTICE,
    THREADS_TEXT_ONLY_SEPARATOR,
)
from discordbot.cogs.gen_reply.link_sources.bilibili import (
    BILIBILI_TIMEOUT_NOTICE,
    BILIBILI_CONTEXT_SEPARATOR,
    BILIBILI_UNREADABLE_NOTICE,
    BILIBILI_TOO_LONG_SEPARATOR,
    BILIBILI_TEXT_ONLY_SEPARATOR,
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

    Handles both shapes the pipeline emits: a bare string, or a list of parts whose ``text``
    fields are joined with newlines. A part carrying no ``text`` (an uploaded ``input_file``,
    an ``input_image``) contributes nothing rather than a blank line, and an unrecognised
    content shape flattens to the empty string instead of raising.

    Args:
        content (object): The item's ``content`` field, exactly as it was recorded.

    Returns:
        The flattened text, empty when the content carries none.
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
    """Returns the first line of a rendered block's text, which is its stable header.

    Every production renderer opens with a fixed framing line and varies only below it, so that
    line is the anchor the extractors match on.

    Args:
        block (Mapping[str, object]): A rendered input item, as its production renderer returns
            it.

    Returns:
        The block's first line.
    """
    return _content_to_text(content=block.get("content")).split("\n", 1)[0]


_PARTICIPANT_HEADER = _header_line(block=render_memory_context_block(memories=[]))
_SERVER_HEADER = _header_line(block=render_server_memory_block(memory=""))
_TONE_HEADER = _header_line(block=render_tone_block(tone=""))
_CALLABLE_HEADER = _header_line(block=render_callable_users_block(allowed={}))
_THREADS_SEPARATOR_HEADS = (
    THREADS_CONTEXT_SEPARATOR.split("\n", 1)[0],
    THREADS_TEXT_ONLY_SEPARATOR.split("\n", 1)[0],
)
_THREADS_NOTICE_HEADS = (
    THREADS_UNAVAILABLE_NOTICE.split("\n", 1)[0],
    THREADS_TIMEOUT_NOTICE.split("\n", 1)[0],
)
_DOUYIN_SEPARATOR_HEADS = (
    DOUYIN_CONTEXT_SEPARATOR.split("\n", 1)[0],
    DOUYIN_TEXT_ONLY_SEPARATOR.split("\n", 1)[0],
)
_DOUYIN_NOTICE_HEADS = (
    DOUYIN_UNAVAILABLE_NOTICE.split("\n", 1)[0],
    DOUYIN_BLOCKED_NOTICE.split("\n", 1)[0],
    DOUYIN_TIMEOUT_NOTICE.split("\n", 1)[0],
)
_BILIBILI_SEPARATOR_HEADS = (
    BILIBILI_CONTEXT_SEPARATOR.split("\n", 1)[0],
    BILIBILI_TEXT_ONLY_SEPARATOR.split("\n", 1)[0],
    BILIBILI_TOO_LONG_SEPARATOR.split("\n", 1)[0],
)
_BILIBILI_NOTICE_HEADS = (
    BILIBILI_UNREADABLE_NOTICE.split("\n", 1)[0],
    BILIBILI_TIMEOUT_NOTICE.split("\n", 1)[0],
)

# A section runs to the blank line before the next `[id: N]`, which is how
# `render_memory_context_block` joins them, so a marker quoted inside a body never splits it.
_ID_SECTION = re.compile(r"\[id: (\d+)\][^\n]*\n(.*?)(?=\n\n\[id: |\Z)", re.DOTALL)
_ID_MARKER = re.compile(r"\[id: (\d+)\]")


def iter_text_blocks(request: ResponseInputParam | str) -> Iterator[tuple[str, str]]:
    """Yields ``(role, text)`` for each role-bearing item in a recorded input.

    A bare string input yields nothing and an item carrying no ``role`` is skipped, so no caller
    has to branch on which shape the pipeline happened to record.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Yields:
        The role and flattened text of each role-bearing item, in the order the pipeline
        assembled them.
    """
    if isinstance(request, str):
        return
    for item in request:
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        if isinstance(role, str):
            yield role, _content_to_text(content=item.get("content"))


def extract_memory_context_block(request: ResponseInputParam | str) -> str | None:
    """Returns the participant-memory assistant block's text, or None if absent.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        The whole block, framing header included, or None when no participant memory was
        injected.
    """
    for role, text in iter_text_blocks(request=request):
        if role == "assistant" and text.split("\n", 1)[0] == _PARTICIPANT_HEADER:
            return text
    return None


def has_memory_context_block(request: ResponseInputParam | str) -> bool:
    """Whether the input carries an injected participant-memory block.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        True when the participant-memory block is present.
    """
    return extract_memory_context_block(request=request) is not None


def extract_user_memory_blocks(request: ResponseInputParam | str) -> dict[int, str]:
    """Maps each injected user id to its memory body within the memory block.

    Empty when no memory block is present, so a leak check reads as
    ``user_id not in extract_user_memory_blocks(request=...)``.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        Each ``[id: N]`` marker's user id to the body rendered under it, stripped.
    """
    block = extract_memory_context_block(request=request)
    if block is None:
        return {}
    body = block.split("\n", 1)[1] if "\n" in block else ""
    return {int(match.group(1)): match.group(2).strip() for match in _ID_SECTION.finditer(body)}


def extract_server_memory_block(request: ResponseInputParam | str) -> str | None:
    """Returns the server-memory assistant block's text, or None if absent.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        The whole block, framing header included, or None when no server memory was injected.
    """
    for role, text in iter_text_blocks(request=request):
        if role == "assistant" and text.split("\n", 1)[0] == _SERVER_HEADER:
            return text
    return None


def extract_tone_block(request: ResponseInputParam | str) -> str | None:
    """Returns the tone-note assistant block's text, or None if absent.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        The whole block, framing header included, or None when no tone note was injected.
    """
    for role, text in iter_text_blocks(request=request):
        if role == "assistant" and text.split("\n", 1)[0] == _TONE_HEADER:
            return text
    return None


def extract_callable_user_ids(request: ResponseInputParam | str) -> set[int]:
    """Returns the ids offered for optional oblique-reference selection.

    This is the narrowed per-request allowlist boundary: it contains only absent
    public nickname-table members, never deterministic participants. Read off the
    ``role="system"`` candidate block alone, so an id the request mentions anywhere else does
    not count as offered.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        The candidate ids, empty when no selection block was injected.
    """
    for role, text in iter_text_blocks(request=request):
        if role == "system" and text.split("\n", 1)[0] == _CALLABLE_HEADER:
            return {int(match) for match in _ID_MARKER.findall(text)}
    return set()


def extract_threads_context_block(request: ResponseInputParam | str) -> str | None:
    """Returns the text of the block following the Threads separator, or None if absent.

    The builder emits a ``role="system"`` separator immediately followed by the
    ``role="user"`` message carrying the post's text and media; this anchors on the
    separator's header line and returns that next block's text. A failure notice is not a
    separator, so it reads as absent here and only ``has_threads_context_block`` sees it.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        The following block's text, empty when the separator was the last item, or None when no
        Threads separator was injected.
    """
    items = list(iter_text_blocks(request=request))
    for index, (role, text) in enumerate(items):
        if role == "system" and text.split("\n", 1)[0] in _THREADS_SEPARATOR_HEADS:
            return items[index + 1][1] if index + 1 < len(items) else ""
    return None


def has_threads_context_block(request: ResponseInputParam | str) -> bool:
    """Whether the input carries an injected Threads separator or notice block.

    Counts the failure notices too, so a test asserting the pipeline said something about the
    link does not have to know whether the fetch succeeded.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        True when a Threads separator or failure notice is present.
    """
    for _role, text in iter_text_blocks(request=request):
        head = text.split("\n", 1)[0]
        if head in _THREADS_SEPARATOR_HEADS or head in _THREADS_NOTICE_HEADS:
            return True
    return False


def tool_names_for_call(responses: RecordedResponses, n: int) -> list[str]:
    """Returns the tool names offered on the nth recorded ``create`` call.

    Only a function tool carries a ``name``; a built-in such as ``{"googleSearch": {}}`` has
    none and is skipped, so an empty list means no function tool was offered rather than no
    tools at all.

    Args:
        responses (RecordedResponses): The recorder the pipeline was driven against.
        n (int): Index into the recorded calls.

    Returns:
        The function tool names, in the order they were offered.
    """
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

    Args:
        responses (RecordedResponses): The recorder the pipeline was driven against.
        phase (Literal["selection", "answer"]): Which phase to locate.

    Returns:
        The index into the recorder's per-call lists.

    Raises:
        AssertionError: `phase` is "answer" and no streaming call was recorded.
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
    """Returns the recorded input for a semantic pipeline phase.

    Args:
        responses (RecordedResponses): The recorder the pipeline was driven against.
        phase (Literal["selection", "answer"]): Which phase's input to read.

    Returns:
        The input recorded for that phase.
    """
    return responses.create_inputs[request_index(responses=responses, phase=phase)]


def extract_douyin_context_block(request: ResponseInputParam | str) -> str | None:
    """Returns the text of the block following the Douyin separator, or None if absent.

    Same anchoring as the Threads twin: the separator is its own ``role="system"`` item and the
    caption plus media ride the item behind it. A failure notice is not a separator, so it reads
    as absent here and only ``has_douyin_context_block`` sees it.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        The following block's text, empty when the separator was the last item, or None when no
        Douyin separator was injected.
    """
    items = list(iter_text_blocks(request=request))
    for index, (role, text) in enumerate(items):
        if role == "system" and text.split("\n", 1)[0] in _DOUYIN_SEPARATOR_HEADS:
            return items[index + 1][1] if index + 1 < len(items) else ""
    return None


def has_douyin_context_block(request: ResponseInputParam | str) -> bool:
    """Whether the input carries an injected Douyin separator or notice block.

    Counts the failure notices too, the WAF-block one included, so a test asserting the pipeline
    said something about the link does not have to know which outcome Douyin returned.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        True when a Douyin separator or failure notice is present.
    """
    for _role, text in iter_text_blocks(request=request):
        head = text.split("\n", 1)[0]
        if head in _DOUYIN_SEPARATOR_HEADS or head in _DOUYIN_NOTICE_HEADS:
            return True
    return False


def extract_bilibili_context_block(request: ResponseInputParam | str) -> str | None:
    """Returns the text of the block following the Bilibili separator, or None if absent.

    Same anchoring as the Threads twin, and the too-long separator counts as one: a video the
    builder deliberately did not download still ships its title and description in the block
    behind it. A failure notice is not a separator, so it reads as absent here and only
    ``has_bilibili_context_block`` sees it.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        The following block's text, empty when the separator was the last item, or None when no
        Bilibili separator was injected.
    """
    items = list(iter_text_blocks(request=request))
    for index, (role, text) in enumerate(items):
        if role == "system" and text.split("\n", 1)[0] in _BILIBILI_SEPARATOR_HEADS:
            return items[index + 1][1] if index + 1 < len(items) else ""
    return None


def has_bilibili_context_block(request: ResponseInputParam | str) -> bool:
    """Whether the input carries an injected Bilibili separator or notice block.

    Counts the failure notices too, so a test asserting the pipeline said something about the
    link does not have to know whether yt-dlp could read it.

    Args:
        request (ResponseInputParam | str): One recorded ``create`` input.

    Returns:
        True when a Bilibili separator or failure notice is present.
    """
    for _role, text in iter_text_blocks(request=request):
        head = text.split("\n", 1)[0]
        if head in _BILIBILI_SEPARATOR_HEADS or head in _BILIBILI_NOTICE_HEADS:
            return True
    return False
