"""Tests for the request extractors in `tests/helpers/llm_input.py`.

The redesigned suite leans on these extractors, so they are pinned here against the real
production renderers.
"""

from openai.types.responses import ResponseInputParam, EasyInputMessageParam

from discordbot.cogs.gen_reply.recall import (
    UserMemory,
    RecallCandidate,
    render_server_memory_block,
    render_callable_users_block,
    render_memory_context_block,
)
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES

from tests.helpers.llm_input import (
    LINK_SOURCE_BLOCKS,
    request_index,
    request_input,
    iter_text_blocks,
    has_memory_context_block,
    extract_callable_user_ids,
    extract_user_memory_blocks,
    extract_server_memory_block,
)


def _answer_request(
    memory_ids: dict[int, str] | None = None,
    server_memory: str | None = None,
    callable_ids: dict[int, str] | None = None,
) -> ResponseInputParam:
    """Assembles a request from whichever blocks a case needs, in this helper's own order."""
    request: ResponseInputParam = []
    if callable_ids is not None:
        request.append(
            render_callable_users_block(
                allowed={
                    uid: RecallCandidate(prompt_label=label) for uid, label in callable_ids.items()
                }
            )
        )
    if server_memory is not None:
        request.append(render_server_memory_block(memory=server_memory))
    if memory_ids is not None:
        memories = [
            UserMemory(
                user_id=str(uid), prompt_label=f"u{uid}", credit_label=f"u{uid}", memory=body
            )
            for uid, body in memory_ids.items()
        ]
        request.append(render_memory_context_block(memories=memories))
    request.append(EasyInputMessageParam(role="user", content="hi"))
    return request


def test_extract_user_memory_blocks_keys_by_id() -> None:
    """Each injected user's memory body is recovered keyed by id."""
    request = _answer_request(memory_ids={1: "喜歡阿狗", 42: "李董的祕密"})
    blocks = extract_user_memory_blocks(request=request)
    assert blocks == {1: "喜歡阿狗", 42: "李董的祕密"}
    assert has_memory_context_block(request=request)


def test_extract_user_memory_blocks_empty_without_block() -> None:
    """A request with no memory block yields no injected ids."""
    request = _answer_request()
    assert extract_user_memory_blocks(request=request) == {}
    assert not has_memory_context_block(request=request)


def test_extract_server_memory_block_present_and_absent() -> None:
    """The server block is found by its production header, else None."""
    with_server = _answer_request(server_memory="這個社群很愛嘴")
    assert extract_server_memory_block(request=with_server) is not None
    assert extract_server_memory_block(request=_answer_request()) is None


def test_extract_callable_user_ids_from_selection_block() -> None:
    """Callable ids are parsed from the selection allowlist block."""
    request = _answer_request(callable_ids={1: "Alice (alice)", 42: "Boss (boss)"})
    assert extract_callable_user_ids(request=request) == {1, 42}


def test_iter_text_blocks_yields_role_text_pairs() -> None:
    """Every role-bearing item flattens to a (role, text) pair."""
    request = _answer_request(memory_ids={1: "m"})
    roles = [role for role, _ in iter_text_blocks(request=request)]
    assert "assistant" in roles
    assert roles[-1] == "user"


class _Recorder:
    """Minimal stand-in for the recording fake Responses resource."""

    def __init__(self) -> None:
        """Initializes the recorded per-call lists."""
        self.create_streams: list[bool] = [False, True]
        self.create_inputs: list[ResponseInputParam | str] = [
            [EasyInputMessageParam(role="user", content="director")],
            [EasyInputMessageParam(role="user", content="answer")],
        ]


def test_request_index_finds_the_answer() -> None:
    """The answer is the last streaming call, not a non-streaming one before it."""
    recorder = _Recorder()
    assert request_index(responses=recorder) == 1
    assert request_input(responses=recorder) == [
        EasyInputMessageParam(role="user", content="answer")
    ]


def test_the_table_names_the_timeout_notice_each_source_injects() -> None:
    """The table finds a timeout notice by its constant's name; the registry wires the one sent.

    `has_timeout_notice` reads the table, so a source whose registry entry injects another text
    would leave `assert not has_timeout_notice(...)` passing on a request that carries one.
    """
    for source in LINK_CONTEXT_SOURCES:
        assert LINK_SOURCE_BLOCKS[source.name].timeout_notice == source.timeout_notice, source.name
