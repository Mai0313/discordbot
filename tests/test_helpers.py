"""Tests for the shared test helpers themselves.

The redesigned suite leans on these extractors and invariant asserts, so they
are pinned here against the real production renderers and database helpers.
"""

import pytest
from sqlalchemy import Update, update
from openai.types.responses import ResponseInputParam, EasyInputMessageParam

from discordbot.cogs.gen_reply.recall import (
    UserMemory,
    RecallCandidate,
    render_server_memory_block,
    render_callable_users_block,
    render_memory_context_block,
)
from discordbot.services.economy.database import (
    UserWallet,
    CasinoLedger,
    CasinoAccount,
    open_session,
    adjust_balance,
    apply_blackjack_settlement,
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
    extract_memory_context_block,
)
from tests.helpers.economy_invariants import (
    assert_wallet_consistent,
    assert_daily_casino_stats,
    assert_casino_ledger_consistent,
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


# --- llm_input ---------------------------------------------------------------


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


def test_extract_user_memory_blocks_handles_bare_string_input() -> None:
    """A bare string input has no role structure and leaks nothing."""
    assert extract_user_memory_blocks(request="just text") == {}
    assert extract_callable_user_ids(request="just text") == set()


def test_extract_server_memory_block_present_and_absent() -> None:
    """The server block is found by its production header, else None."""
    with_server = _answer_request(server_memory="這個社群很愛嘴")
    assert extract_server_memory_block(request=with_server) is not None
    assert extract_server_memory_block(request=_answer_request()) is None


def test_extract_callable_user_ids_from_selection_block() -> None:
    """Callable ids are parsed from the selection allowlist block."""
    request = _answer_request(callable_ids={1: "Alice (alice)", 42: "Boss (boss)"})
    assert extract_callable_user_ids(request=request) == {1, 42}


def test_extract_memory_context_block_returns_full_text() -> None:
    """The raw block text is returned for callers needing the framing."""
    request = _answer_request(memory_ids={7: "x"})
    block = extract_memory_context_block(request=request)
    assert block is not None
    assert "[id: 7]" in block


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


# --- economy_invariants ------------------------------------------------------


async def _write(statement: Update) -> None:
    """Writes straight to the ledger, past every helper that keeps its identities."""
    async with open_session() as session:
        await session.execute(statement=statement)
        await session.commit()


async def test_assert_wallet_consistent_fails_a_broken_wallet() -> None:
    """A clean credit passes, and totals that no longer add up to the balance fail."""
    await adjust_balance(user_id=1, name="alice", delta=250)
    await assert_wallet_consistent(user_id=1, expected_balance=250)

    await _write(statement=update(UserWallet).values(total_spent=1))

    with pytest.raises(AssertionError, match="wallet identity broken"):
        await assert_wallet_consistent(user_id=1)


async def test_the_casino_asserts_fail_a_broken_counter() -> None:
    """Settled play passes, and a ledger or daily counter off its identity fails.

    Each broken call passes the stored values as expected, so only the identity can fail it.
    """
    await adjust_balance(user_id=1, name="alice", delta=100)
    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-10, casino_delta=10
    )
    await assert_casino_ledger_consistent(expected_balance=10)
    await assert_daily_casino_stats(user_id=1, loss=10, win=0, net=-10)

    await _write(statement=update(CasinoLedger).values(total_spent=1))
    await _write(statement=update(CasinoAccount).values(daily_net=0))

    with pytest.raises(AssertionError, match="casino ledger identity broken"):
        await assert_casino_ledger_consistent(expected_balance=10)
    with pytest.raises(AssertionError, match="daily net identity broken"):
        await assert_daily_casino_stats(user_id=1, loss=10, win=0, net=0)
