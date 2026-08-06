"""Pins the shared `tests/helpers/` doubles, extractors and invariant asserts against production.

Everything under `tests/helpers/` is what the rest of the suite asserts *through*, so a helper
that quietly stops finding what it looks for turns every assertion built on it into a vacuous
pass rather than a failure. That is not a hypothetical for the `llm_input` extractors: their
anchors are derived at import time from the production renderers' own framing lines, so a header
the pipeline stops emitting simply makes `extract_user_memory_blocks` return `{}` — which is
precisely what the memory-leak checks spelt `42 not in extract_user_memory_blocks(...)` in
`tests/test_gen_reply.py` assert. Each extractor therefore needs a positive control, and this
file is it.

The sections follow the four helper modules. `llm_input`: the block extractors, driven off inputs
built by calling the real `render_*_block` functions, plus the phase-to-call-index readers over a
stand-in recorder. `embeds`: both outcomes of the field assert and the title-marker one.
`discord_mocks`: that the interaction doubles record what a cog sent and hand back a message
double that records in turn. `economy_invariants`: that the accounting identities hold against
real database state, which is why those two tests take `economy_isolated_db`.

Production *wording* is deliberately not pinned here. The tests render their inputs with the same
functions the extractors anchor on, so a reworded header stays a matched pair and only a
structural break fails — a lost role, a changed `[id: N]` marker, a renderer that stops opening
with a stable header line.
"""

import pytest
import nextcord
from openai.types.responses import ResponseInputParam, EasyInputMessageParam

from discordbot.services.economy.database import adjust_balance
from discordbot.cogs.gen_reply.memory_tool import (
    UserMemory,
    render_server_memory_block,
    render_callable_users_block,
    render_memory_context_block,
)

from tests.helpers.embeds import assert_embed_has_field, assert_embed_title_prefix
from tests.helpers.llm_input import (
    request_index,
    request_input,
    iter_text_blocks,
    tool_names_for_call,
    has_memory_context_block,
    extract_callable_user_ids,
    extract_user_memory_blocks,
    extract_server_memory_block,
    extract_memory_context_block,
)
from tests.helpers.discord_mocks import FakeUser, FakeResponse, FakeInteraction
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
    """Builds a recorded input carrying whichever production blocks a case needs.

    Each block is rendered by the production renderer rather than written out here, so an
    extractor that has drifted from its anchor fails instead of matching a stale literal. A None
    argument omits its block entirely, which is what gives the absence cases a request that is
    otherwise identical.

    The callable-users block rides the selection request in production, not the answer one;
    combining all three here is harmless because every extractor anchors on its own header
    independently.

    Returns:
        The blocks in render order, closed by the triggering `role="user"` message.
    """
    request: ResponseInputParam = []
    if callable_ids is not None:
        request.append(render_callable_users_block(allowed=callable_ids))
    if server_memory is not None:
        request.append(render_server_memory_block(memory=server_memory))
    if memory_ids is not None:
        memories = [
            UserMemory(user_id=str(uid), username=f"u{uid}", memory=body)
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
    """A bare string input yields no memory bodies and no callable ids."""
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
    """The whole participant block comes back, framing and `[id: N]` markers intact."""
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
        """Prefills one non-streaming selection call and one streaming answer call."""
        self.create_streams: list[bool] = [False, True]
        self.create_tools: list[list[object] | None] = [
            [{"name": "get_user_memory"}],
            [{"type": "web_search"}],
        ]
        self.create_inputs: list[ResponseInputParam | str] = [
            [EasyInputMessageParam(role="user", content="selection")],
            [EasyInputMessageParam(role="user", content="answer")],
        ]


def test_request_index_maps_phase_to_position() -> None:
    """A phase name resolves to its recorded call index, and to that call's own input."""
    recorder = _Recorder()
    assert request_index(responses=recorder, phase="selection") == 0
    assert request_index(responses=recorder, phase="answer") == 1
    assert request_input(responses=recorder, phase="answer") == [
        EasyInputMessageParam(role="user", content="answer")
    ]


def test_tool_names_for_call_reads_offered_tools() -> None:
    """Tool names are read structurally, ignoring non-function builtins."""
    recorder = _Recorder()
    assert tool_names_for_call(responses=recorder, n=0) == ["get_user_memory"]
    assert tool_names_for_call(responses=recorder, n=1) == []


# --- embeds ------------------------------------------------------------------


def test_assert_embed_has_field_returns_field() -> None:
    """A present field is returned so the caller can check its value."""
    embed = nextcord.Embed(title="💰 財務總覽")
    embed.add_field(name="現金", value="100")
    field = assert_embed_has_field(embed=embed, name="現金")
    assert field.value == "100"


def test_assert_embed_has_field_raises_when_missing() -> None:
    """A missing field raises with the available names for diagnosis."""
    embed = nextcord.Embed(title="x")
    with pytest.raises(AssertionError, match="現金"):
        assert_embed_has_field(embed=embed, name="現金")


def test_assert_embed_title_prefix_checks_marker() -> None:
    """The title is matched on its leading category marker only."""
    embed = nextcord.Embed(title="💰 財務總覽")
    assert_embed_title_prefix(embed=embed, prefix="💰")


# --- discord_mocks -----------------------------------------------------------


async def test_fake_response_records_defer_and_send() -> None:
    """The response double records deferral, sends, and done-state."""
    response = FakeResponse()
    assert not response.is_done()
    await response.defer(ephemeral=True)
    assert response.deferred_ephemeral is True
    await response.send_message(content="hi", ephemeral=True)
    assert response.is_done()
    assert response.sent[0]["content"] == "hi"


async def test_fake_interaction_defaults_and_followup() -> None:
    """The interaction double defaults a user, and its followup returns a replyable message."""
    interaction = FakeInteraction(user=FakeUser(user_id=5, name="bob"))
    assert interaction.user.id == 5
    message = await interaction.followup.send(content="done")
    assert interaction.followup.sent[0]["content"] == "done"
    await message.reply(content="re")
    assert message.replies[0]["content"] == "re"


# --- economy_invariants ------------------------------------------------------


async def test_assert_wallet_consistent_passes_on_clean_credit(economy_isolated_db: None) -> None:
    """A simple credit keeps the wallet identity and matches the expected balance."""
    del economy_isolated_db
    await adjust_balance(user_id=1, name="alice", delta=250)
    account = await assert_wallet_consistent(user_id=1, expected_balance=250)
    assert account.total_earned == 250


async def test_assert_casino_and_daily_stats_zero_baseline(economy_isolated_db: None) -> None:
    """A fresh ledger and an inactive user satisfy the accounting identities."""
    del economy_isolated_db
    await assert_casino_ledger_consistent(expected_balance=0)
    await assert_daily_casino_stats(user_id=1, loss=0, win=0, net=0)
