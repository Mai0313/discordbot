"""Tests for the deep-research feature: marker extraction, delivery, agent helpers, and store."""

from types import SimpleNamespace
import base64

from discordbot.cogs import research as research_cog
from discordbot.cogs._research import agent
from discordbot.cogs._research import database as rdb
from discordbot.cogs._gen_reply.markers import extract_inline_markers, scrub_markers_for_preview
from discordbot.cogs._research.delivery import split_report

# ----- marker extraction --------------------------------------------------------------------


def test_deep_research_block_is_pulled_and_brief_captured() -> None:
    markers = extract_inline_markers(
        text="好喔幫你查 <deep-research>研究 TPU 的競爭格局</deep-research> 等等貼到 thread"
    )
    assert markers.research_brief == "研究 TPU 的競爭格局"
    assert "TPU" not in markers.cleaned_text
    assert "thread" in markers.cleaned_text


def test_unclosed_trailing_deep_research_is_still_pulled() -> None:
    markers = extract_inline_markers(text="開查囉 <deep-research>研究量子計算最新進展")
    assert markers.research_brief == "研究量子計算最新進展"
    assert "量子" not in markers.cleaned_text


def test_deep_research_coexists_with_voice() -> None:
    markers = extract_inline_markers(
        text="<voice>馬上幫你查</voice> <deep-research>研究 X</deep-research>"
    )
    assert markers.voice_requested
    assert "馬上幫你查" in markers.cleaned_text
    assert markers.research_brief == "研究 X"
    assert "X" not in markers.cleaned_text


def test_scrub_hides_deep_research_mid_stream() -> None:
    assert "TPU" not in scrub_markers_for_preview(text="好喔 <deep-research>研究 TPU")


def test_no_marker_leaves_text_and_brief_untouched() -> None:
    markers = extract_inline_markers(text="這只是一般回覆,沒有任何 marker")
    assert markers.research_brief is None
    assert markers.cleaned_text == "這只是一般回覆,沒有任何 marker"


# ----- delivery splitting -------------------------------------------------------------------


def test_split_report_keeps_short_text_as_one_chunk() -> None:
    assert split_report(text="short report") == ["short report"]


def test_split_report_prefers_paragraph_boundaries() -> None:
    para_a = "A" * 1200
    para_b = "B" * 1200
    chunks = split_report(text=f"{para_a}\n\n{para_b}")
    assert len(chunks) == 2
    assert chunks[0] == para_a
    assert chunks[1] == para_b


def test_split_report_hard_cuts_an_oversized_line() -> None:
    chunks = split_report(text="C" * 5000, limit=2000)
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert "".join(chunks) == "C" * 5000


# ----- agent helpers ------------------------------------------------------------------------


def test_deep_research_agent_config_shape() -> None:
    plan_config = agent._deep_research_agent_config(collaborative_planning=True)
    assert plan_config["type"] == "deep-research"
    assert plan_config["collaborative_planning"] is True
    run_config = agent._deep_research_agent_config(collaborative_planning=False)
    assert run_config["collaborative_planning"] is False


def test_to_result_extracts_text_image_and_usage() -> None:
    image_b64 = base64.b64encode(b"PNGBYTES").decode()
    interaction = SimpleNamespace(
        id="int_123",
        status="completed",
        output_text="# Report\nbody",
        usage=SimpleNamespace(total_input_tokens=250000, total_output_tokens=60000),
        steps=[
            SimpleNamespace(
                type="model_output", content=[SimpleNamespace(type="image", data=image_b64)]
            )
        ],
    )
    result = agent._to_result(interaction=interaction)
    assert result.interaction_id == "int_123"
    assert result.ok is True
    assert result.report_text.startswith("# Report")
    assert result.image_bytes == b"PNGBYTES"
    assert result.input_tokens == 250000
    assert result.output_tokens == 60000


def test_to_result_handles_failure_and_missing_fields() -> None:
    interaction = SimpleNamespace(id="int_x", status="failed")
    result = agent._to_result(interaction=interaction)
    assert result.ok is False
    assert result.report_text == ""
    assert result.image_bytes is None
    assert result.input_tokens == 0


def test_latest_thought_returns_last_summary() -> None:
    interaction = SimpleNamespace(
        steps=[
            SimpleNamespace(content=[SimpleNamespace(type="thought_summary", text="first")]),
            SimpleNamespace(content=[SimpleNamespace(type="thought_summary", text="second")]),
        ]
    )
    assert agent._latest_thought(interaction=interaction) == "second"
    assert agent._latest_thought(interaction=SimpleNamespace()) is None


# ----- research module helpers --------------------------------------------------------------


def test_fallback_thread_name_uses_first_line() -> None:
    name = research_cog._fallback_thread_name(brief="研究 TPU 的歷史與競爭格局\n更多細節")
    assert name.startswith("研究 TPU")
    assert "\n" not in name
    assert research_cog._fallback_thread_name(brief="   ") == "深度研究"


def test_tier_label_maps_agent_strings() -> None:
    assert research_cog._tier_label(agent="antigravity-preview-05-2026") == "Antigravity"
    assert research_cog._tier_label(agent="deep-research-preview-04-2026") == "Deep Research"
    assert (
        research_cog._tier_label(agent="deep-research-max-preview-04-2026") == "Deep Research Max"
    )


def test_terminal_phase_mapping() -> None:
    assert research_cog._terminal_phase(status="completed") == "done"
    assert research_cog._terminal_phase(status="cancelled") == "cancelled"
    assert research_cog._terminal_phase(status="budget_exceeded") == "failed"


def test_owner_id_from_mention_parses_digits() -> None:
    assert research_cog._owner_id_from_mention(mention="<@123456789>") == 123456789
    assert research_cog._owner_id_from_mention(mention="<@!42>") == 42
    assert research_cog._owner_id_from_mention(mention="nobody") == 0


def test_failure_text_distinguishes_budget() -> None:
    assert "成本上限" in research_cog._failure_text(status="budget_exceeded")
    assert "取消" in research_cog._failure_text(status="cancelled")
    assert research_cog._failure_text(status="failed")


# ----- persistence (reply.db) ---------------------------------------------------------------


async def test_session_round_trip(research_isolated_db: None) -> None:
    await rdb.upsert_session(
        thread_id=1,
        owner_id=99,
        channel_id=7,
        guild_id=5,
        source_message_id=3,
        agent="antigravity-preview-05-2026",
        interaction_id=None,
        brief="研究 X",
        phase="researching",
    )
    session = await rdb.get_session(thread_id=1)
    assert session is not None
    assert session.owner_id == 99
    assert session.brief == "研究 X"
    assert session.phase == "researching"
    assert session.interaction_id is None
    assert await rdb.get_session(thread_id=999) is None


async def test_set_interaction_and_phase(research_isolated_db: None) -> None:
    await rdb.upsert_session(
        thread_id=2,
        owner_id=1,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="antigravity-preview-05-2026",
        interaction_id=None,
        brief="b",
        phase="researching",
    )
    await rdb.set_interaction(
        thread_id=2,
        interaction_id="int_abc",
        agent="deep-research-preview-04-2026",
        phase="planning",
    )
    session = await rdb.get_session(thread_id=2)
    assert session is not None
    assert session.interaction_id == "int_abc"
    assert session.agent == "deep-research-preview-04-2026"
    assert session.phase == "planning"
    await rdb.set_phase(thread_id=2, phase="done")
    refreshed = await rdb.get_session(thread_id=2)
    assert refreshed is not None
    assert refreshed.phase == "done"


async def test_active_thread_for_owner_excludes_terminal(research_isolated_db: None) -> None:
    await rdb.upsert_session(
        thread_id=10,
        owner_id=500,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="antigravity-preview-05-2026",
        interaction_id=None,
        brief="b",
        phase="researching",
    )
    assert await rdb.active_thread_for_owner(owner_id=500) == 10
    await rdb.set_phase(thread_id=10, phase="done")
    assert await rdb.active_thread_for_owner(owner_id=500) is None
    assert await rdb.active_thread_for_owner(owner_id=12345) is None


async def test_list_resumable_only_returns_researching(research_isolated_db: None) -> None:
    for thread_id, phase in ((20, "researching"), (21, "planning"), (22, "done")):
        await rdb.upsert_session(
            thread_id=thread_id,
            owner_id=thread_id,
            channel_id=1,
            guild_id=1,
            source_message_id=1,
            agent="deep-research-preview-04-2026",
            interaction_id="int_x",
            brief="b",
            phase=phase,  # type: ignore[arg-type]  # the test deliberately exercises each phase
        )
    resumable = await rdb.list_resumable()
    assert {session.thread_id for session in resumable} == {20}


def test_cast_phase_defaults_unknown_to_failed() -> None:
    assert rdb.cast_phase(value="researching") == "researching"
    assert rdb.cast_phase(value="bogus") == "failed"
