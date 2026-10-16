"""Tests for the `/memory` commands."""

import time
from pathlib import Path

import pytest
from nextcord import Embed

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.memory.cog import MemoryCogs
from discordbot.services.memory import regeneration
from discordbot.cogs.memory.views import (
    MEMORY_PAGE_MAX_CHARS,
    MemoryPagesView,
    MemoryClearConfirmView,
)
from discordbot.services.memory.store import (
    write_fact,
    write_tone,
    append_detail,
    append_raw_entry,
)
from discordbot.services.memory.prompts import PHASE2_PROMPT, PHASE2_COMPACTION_BLOCK

from tests.helpers.memory import FakeMemoryClient, make_memory_cog
from tests.helpers.casting import as_interaction
from tests.helpers.user_memory import (
    USER_ID,
    USER_SCOPE,
    _GUILD_222,
    DETAIL_EVIDENCE,
    _memory_text,
    _stored_fact,
)
from tests.helpers.discord_mocks import FakeUser, FakeInteraction
from tests.helpers.server_memory import GUILD_ID, SERVER_SCOPE, _server_fact


def _interaction() -> FakeInteraction:
    """Builds an interaction invoked by the test user."""
    return FakeInteraction(user=FakeUser(user_id=USER_ID))


async def test_memory_show_displays_stored_memory(memory_isolated_dir: Path) -> None:
    """The owner's view leads each compartment with who can see it, then its facts."""
    write_fact(scope=USER_SCOPE, fact=_stored_fact(section="profile", text="愛開玩笑"))
    cog = make_memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=interaction))
    assert interaction.response.sent[-1]["ephemeral"] is True
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    description = embed.description or ""
    assert "愛開玩笑" in description
    # Provenance is the directory now, so showing it is free and tells the owner exactly
    # where each thing they told the bot can come back up.
    assert description.startswith("# 全部聊天都看得到")
    assert "## 使用者輪廓" in description
    # A memory that fits one embed keeps the original no-view behavior.
    assert "view" not in interaction.response.sent[-1]


async def test_memory_show_separates_a_guild_compartment_from_the_shared_one(
    memory_isolated_dir: Path,
) -> None:
    """A fact locked to one server is shown under its own heading, never merged into global."""
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="全域事實"))
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(fact_id="1" * 16, compartment=_GUILD_222, text="本群事實"),
    )
    cog = make_memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    description = embed.description or ""
    assert description.index("# 全部聊天都看得到") < description.index("# 只有伺服器 222 看得到")
    assert description.index("全域事實") < description.index("# 只有伺服器 222 看得到")


async def test_memory_show_paginates_oversized_memory(memory_isolated_dir: Path) -> None:
    for index in range(80):
        write_fact(
            scope=USER_SCOPE,
            fact=_stored_fact(fact_id=f"{index:016x}", text=f"記憶條目 {index} " + "內" * 80),
        )
    cog = make_memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=interaction))
    sent = interaction.response.sent[-1]
    assert sent["ephemeral"] is True
    view = sent["view"]
    assert isinstance(view, MemoryPagesView)
    assert len(view.pages) > 1
    embed = sent["embed"]
    assert isinstance(embed, Embed)
    assert len(embed.description or "") <= MEMORY_PAGE_MAX_CHARS
    assert (embed.description or "").startswith("# 全部聊天都看得到")
    assert embed.footer is not None
    assert f"第 1/{len(view.pages)} 頁" in (embed.footer.text or "")


async def test_memory_show_handles_empty_memory(memory_isolated_dir: Path) -> None:
    cog = make_memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=interaction))
    assert interaction.response.sent[-1]["ephemeral"] is True
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "還沒有任何記憶" in (embed.description or "")


def _record_schedules(
    monkeypatch: pytest.MonkeyPatch, scheduled: bool = True
) -> dict[str, object]:
    """Stands in for the cog's rebuild scheduler; returns what a call handed it, if any."""
    calls: dict[str, object] = {}

    def fake_schedule(scope: str, writer: object, identity: str) -> bool:
        calls.update(scope=scope, writer=writer, identity=identity)
        return scheduled

    monkeypatch.setattr("discordbot.cogs.memory.cog.schedule_memory_regeneration", fake_schedule)
    return calls


@pytest.mark.parametrize(
    argnames=("scheduled", "expected_text"), argvalues=[(True, "已排程"), (False, "正在重建中")]
)
async def test_memory_regenerate_command_schedules_in_background(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch, scheduled: bool, expected_text: str
) -> None:
    cog = make_memory_cog()
    writer_sentinel = object()
    # Seeded into the cached_property's slot rather than through `setattr`, which reads the
    # old value first to restore it and would build a real `AsyncOpenAI` off credentials CI
    # does not have.
    monkeypatch.setitem(cog.__dict__, "memory_writer", writer_sentinel)
    calls = _record_schedules(monkeypatch=monkeypatch, scheduled=scheduled)
    # Evidence must exist or the command short-circuits before scheduling.
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    interaction = _interaction()
    await MemoryCogs.memory_regenerate.callback(cog, as_interaction(fake=interaction))

    # The command replies immediately and never blocks on the rebuild, so it
    # neither defers nor uses a followup.
    assert interaction.response.deferred is False
    assert interaction.followup.sent == []
    assert interaction.response.sent[-1]["ephemeral"] is True
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert expected_text in (embed.description or "")
    assert calls["scope"] == USER_SCOPE
    assert calls["writer"] is writer_sentinel
    assert calls["identity"] == f"Alice (alice) [id: {USER_ID}]"


async def test_memory_regenerate_command_rebuilds_under_the_per_user_prompt(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command rebuilds only the caller's own scope, so only the per-user prompt runs."""
    cog = make_memory_cog()
    fake_client = FakeMemoryClient()
    monkeypatch.setitem(cog.__dict__, "client", fake_client)
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)

    await MemoryCogs.memory_regenerate.callback(cog, as_interaction(fake=_interaction()))
    task = regeneration._regeneration_tasks.get(key=USER_SCOPE)
    assert task is not None
    await task

    rebuilt = {
        instructions.removesuffix(PHASE2_COMPACTION_BLOCK)
        for instructions in fake_client.responses.parse_instructions
    }
    assert rebuilt == {PHASE2_PROMPT}


async def test_memory_regenerate_without_a_proxy_key_answers_the_command(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer's client is first built here, and the SDK refuses it without a key."""
    # The SDK also accepts `OPENAI_ADMIN_KEY` from the environment, which would build the client.
    monkeypatch.delenv(name="OPENAI_ADMIN_KEY", raising=False)
    cog = make_memory_cog()
    cog.config = LLMConfig.model_construct()
    calls = _record_schedules(monkeypatch=monkeypatch)
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    interaction = _interaction()

    await MemoryCogs.memory_regenerate.callback(cog, as_interaction(fake=interaction))

    assert calls == {}
    assert interaction.response.sent[-1]["ephemeral"] is True
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert embed.footer.text == "OpenAIError"


async def test_memory_regenerate_command_reports_no_evidence(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cog = make_memory_cog()
    calls = _record_schedules(monkeypatch=monkeypatch)
    # No raw or detail evidence exists for this scope.
    interaction = _interaction()
    await MemoryCogs.memory_regenerate.callback(cog, as_interaction(fake=interaction))

    # Without evidence the background task would no-op, so nothing is scheduled
    # and the user is told there is nothing to rebuild yet.
    assert calls == {}
    assert interaction.response.deferred is False
    assert interaction.response.sent[-1]["ephemeral"] is True
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "還沒有足夠的觀察記錄" in (embed.description or "")


async def test_memory_regenerate_command_blocked_by_cooldown(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cog = make_memory_cog()
    regeneration._last_regeneration[USER_SCOPE] = time.monotonic()
    calls = _record_schedules(monkeypatch=monkeypatch)
    interaction = _interaction()
    await MemoryCogs.memory_regenerate.callback(cog, as_interaction(fake=interaction))

    # Rejected up front: nothing scheduled, no defer, just the ephemeral notice.
    assert calls == {}
    assert interaction.response.deferred is False
    assert interaction.followup.sent == []
    assert interaction.response.sent[-1]["ephemeral"] is True
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "請稍後再試" in (embed.description or "")


async def test_memory_show_reports_pending_observations_before_first_consolidation(
    memory_isolated_dir: Path,
) -> None:
    append_raw_entry(scope=USER_SCOPE, entry_text="偏好訊號:\n- 第一筆觀察")
    cog = make_memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "1 筆" in (embed.description or "")
    assert "整理" in (embed.description or "")
    assert "還沒有任何記憶" not in (embed.description or "")


async def test_memory_show_counts_pending_observations_in_the_footer(
    memory_isolated_dir: Path,
) -> None:
    """Once memory exists the pending count moves to the footer, not over the content."""
    write_fact(scope=USER_SCOPE, fact=_stored_fact(section="profile", text="愛開玩笑"))
    append_raw_entry(scope=USER_SCOPE, entry_text="偏好訊號:\n- 新觀察")
    cog = make_memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "愛開玩笑" in (embed.description or "")
    assert embed.footer is not None
    assert "1 筆" in (embed.footer.text or "")


async def test_memory_show_leads_with_the_tone_note(memory_isolated_dir: Path) -> None:
    """The tone note is a scope-wide tier, so it leads the view instead of sitting in one
    compartment; a user who only has a tone note still sees it rather than the placeholder.
    """
    write_tone(scope=USER_SCOPE, content="## 語氣偏好\n* 偏好禮貌")
    cog = make_memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert (embed.description or "").startswith("## 語氣偏好")


async def test_memory_clear_command_only_opens_the_confirmation(memory_isolated_dir: Path) -> None:
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="舊記憶"))
    cog = make_memory_cog()
    interaction = _interaction()

    await MemoryCogs.memory_clear.callback(cog, as_interaction(fake=interaction))

    assert interaction.response.sent[-1]["ephemeral"] is True
    view = interaction.response.sent[-1]["view"]
    assert isinstance(view, MemoryClearConfirmView)
    assert view.scope == USER_SCOPE
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "沒辦法復原" in (embed.description or "")
    # Bound, or an abandoned one-click wipe prompt would never go inert.
    assert view._origin is interaction
    # The command itself must never delete: that is the confirm button's job.
    assert "舊記憶" in _memory_text()


async def test_memory_server_show_displays_stored_memory(memory_isolated_dir: Path) -> None:
    """The command shows the guild's own facts, rendered under the server headings."""
    write_fact(
        scope=SERVER_SCOPE,
        fact=_server_fact(fact_id="a" * 16, section="profile", text="大家都很愛玩楓之谷"),
    )
    cog = make_memory_cog()
    interaction = FakeInteraction(guild_id=GUILD_ID)
    await MemoryCogs.memory_server_show.callback(cog, as_interaction(fake=interaction))
    assert interaction.response.sent[-1]["ephemeral"] is True
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "楓之谷" in (embed.description or "")


async def test_memory_server_show_handles_empty_memory(memory_isolated_dir: Path) -> None:
    """A guild the bot has never consolidated gets a placeholder, not an empty embed."""
    cog = make_memory_cog()
    interaction = FakeInteraction(guild_id=GUILD_ID)
    await MemoryCogs.memory_server_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "還沒有對這個伺服器的記憶" in (embed.description or "")


async def test_memory_server_show_blocks_dms(memory_isolated_dir: Path) -> None:
    """There is no server scope in a DM, so the command refuses before reading anything."""
    cog = make_memory_cog()
    interaction = FakeInteraction(in_guild=False)
    await MemoryCogs.memory_server_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "只能在伺服器" in (embed.description or "")


async def test_memory_server_show_refuses_a_server_the_bot_is_not_in(
    memory_isolated_dir: Path,
) -> None:
    """A user install reaches servers the bot was never added to; the refusal says so."""
    write_fact(scope=SERVER_SCOPE, fact=_server_fact(fact_id="a" * 16, text="大家都很愛玩楓之谷"))
    cog = make_memory_cog()
    interaction = FakeInteraction(guild_id=GUILD_ID)
    interaction.guild = None
    await MemoryCogs.memory_server_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "沒有被加進這個伺服器" in (embed.description or "")
    assert "楓之谷" not in (embed.description or "")
