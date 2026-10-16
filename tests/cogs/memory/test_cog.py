"""Tests for the `/memory` commands."""

import time

import pytest
from nextcord import Embed

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.memory.cog import MemoryCogs
from discordbot.services.memory import regeneration
from discordbot.cogs.memory.views import (
    MEMORY_PAGE_MAX_CHARS,
    MemoryPagesView,
    MemoryClearConfirmView,
    compartment_label,
    build_memory_embed,
    memory_footer_text,
)
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
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


async def test_memory_show_displays_stored_memory() -> None:
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
    assert description.startswith(
        f"# {compartment_label(compartment=GLOBAL_COMPARTMENT, bot=cog.bot)}"
    )
    # A memory that fits one embed keeps the original no-view behavior.
    assert "view" not in interaction.response.sent[-1]


async def test_memory_show_separates_a_guild_compartment_from_the_shared_one() -> None:
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
    shared = f"# {compartment_label(compartment=GLOBAL_COMPARTMENT, bot=cog.bot)}"
    guild = f"# {compartment_label(compartment=_GUILD_222, bot=cog.bot)}"
    assert description.index(shared) < description.index(guild)
    assert description.index("全域事實") < description.index(guild)


async def test_memory_show_paginates_oversized_memory() -> None:
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
    assert (embed.description or "").startswith(
        f"# {compartment_label(compartment=GLOBAL_COMPARTMENT, bot=cog.bot)}"
    )
    first_page = build_memory_embed(
        page_text="",
        page_index=0,
        page_count=len(view.pages),
        footer_text=memory_footer_text(pending_count=0),
        title="",
    )
    assert embed.footer.text == first_page.footer.text


async def test_memory_show_handles_empty_memory() -> None:
    cog = make_memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=interaction))
    assert interaction.response.sent[-1]["ephemeral"] is True
    assert "view" not in interaction.response.sent[-1]
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert embed.description
    # A placeholder, not a compartment rendered empty.
    assert not embed.description.startswith("#")


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


async def test_memory_regenerate_command_schedules_in_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cog = make_memory_cog()
    writer_sentinel = object()
    # Seeded into the cached_property's slot rather than through `setattr`, which reads the
    # old value first to restore it and would build a real `AsyncOpenAI` off credentials CI
    # does not have.
    monkeypatch.setitem(cog.__dict__, "memory_writer", writer_sentinel)
    # Evidence must exist or the command short-circuits before scheduling.
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    descriptions: list[str | None] = []
    for scheduled in (True, False):
        calls = _record_schedules(monkeypatch=monkeypatch, scheduled=scheduled)
        interaction = _interaction()
        await MemoryCogs.memory_regenerate.callback(cog, as_interaction(fake=interaction))

        # The command replies immediately and never blocks on the rebuild, so it
        # neither defers nor uses a followup.
        assert interaction.response.deferred is False
        assert interaction.followup.sent == []
        assert interaction.response.sent[-1]["ephemeral"] is True
        embed = interaction.response.sent[-1]["embed"]
        assert isinstance(embed, Embed)
        descriptions.append(embed.description)
        assert calls["scope"] == USER_SCOPE
        assert calls["writer"] is writer_sentinel
        assert calls["identity"] == f"Alice (alice) [id: {USER_ID}]"
    # A rebuild already running is not reported as a newly scheduled one.
    assert descriptions[0] != descriptions[1]


async def test_memory_regenerate_command_rebuilds_under_the_per_user_prompt(
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer's client is first built here, and the SDK refuses it without a key."""
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cog = make_memory_cog()
    # A writer in place, so only the evidence check can stop the command.
    monkeypatch.setitem(cog.__dict__, "memory_writer", object())
    calls = _record_schedules(monkeypatch=monkeypatch)
    # No raw or detail evidence exists for this scope.
    interaction = _interaction()
    await MemoryCogs.memory_regenerate.callback(cog, as_interaction(fake=interaction))

    # Without evidence the background task would no-op, so nothing is scheduled
    # and the user is told there is nothing to rebuild yet.
    assert calls == {}
    assert interaction.response.deferred is False
    assert interaction.response.sent[-1]["ephemeral"] is True
    assert isinstance(interaction.response.sent[-1]["embed"], Embed)


async def test_memory_regenerate_command_blocked_by_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cog = make_memory_cog()
    # Evidence and a writer in place, so only the cooldown can stop the command.
    monkeypatch.setitem(cog.__dict__, "memory_writer", object())
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    regeneration._last_regeneration[USER_SCOPE] = time.monotonic()
    calls = _record_schedules(monkeypatch=monkeypatch)
    interaction = _interaction()
    await MemoryCogs.memory_regenerate.callback(cog, as_interaction(fake=interaction))

    # Rejected up front: nothing scheduled, no defer, just the ephemeral notice.
    assert calls == {}
    assert interaction.response.deferred is False
    assert interaction.followup.sent == []
    assert interaction.response.sent[-1]["ephemeral"] is True
    assert isinstance(interaction.response.sent[-1]["embed"], Embed)


async def test_memory_show_reports_pending_observations_before_first_consolidation() -> None:
    for index in range(3):
        append_raw_entry(scope=USER_SCOPE, entry_text=f"偏好訊號:\n- 第 {index} 筆觀察")
    cog = make_memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "3" in (embed.description or "")
    # Not the reply a scope with nothing staged gets.
    empty = FakeInteraction(user=FakeUser(user_id=USER_ID + 1))
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=empty))
    empty_embed = empty.response.sent[-1]["embed"]
    assert isinstance(empty_embed, Embed)
    assert embed.description != empty_embed.description


async def test_memory_show_counts_pending_observations_in_the_footer() -> None:
    """Once memory exists the pending count moves to the footer, not over the content."""
    write_fact(scope=USER_SCOPE, fact=_stored_fact(section="profile", text="愛開玩笑"))
    append_raw_entry(scope=USER_SCOPE, entry_text="偏好訊號:\n- 新觀察")
    cog = make_memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "愛開玩笑" in (embed.description or "")
    assert embed.footer.text == memory_footer_text(pending_count=1)


async def test_memory_show_leads_with_the_tone_note() -> None:
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


async def test_memory_clear_command_only_opens_the_confirmation() -> None:
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="舊記憶"))
    cog = make_memory_cog()
    interaction = _interaction()

    await MemoryCogs.memory_clear.callback(cog, as_interaction(fake=interaction))

    assert interaction.response.sent[-1]["ephemeral"] is True
    view = interaction.response.sent[-1]["view"]
    assert isinstance(view, MemoryClearConfirmView)
    assert view.scope == USER_SCOPE
    # Bound, or an abandoned one-click wipe prompt would never go inert.
    await view.on_timeout()
    assert interaction.edits[-1]["view"] is view
    # The command itself must never delete: that is the confirm button's job.
    assert "舊記憶" in _memory_text()


async def test_memory_server_show_displays_stored_memory() -> None:
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


async def test_memory_server_show_handles_empty_memory() -> None:
    """A guild the bot has never consolidated gets a placeholder, not an empty embed."""
    cog = make_memory_cog()
    interaction = FakeInteraction(guild_id=GUILD_ID)
    await MemoryCogs.memory_server_show.callback(cog, as_interaction(fake=interaction))
    assert interaction.response.sent[-1]["ephemeral"] is True
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert embed.description


async def test_memory_server_show_blocks_dms() -> None:
    """There is no server scope in a DM, so the command refuses before reading anything."""
    cog = make_memory_cog()
    interaction = FakeInteraction(in_guild=False)
    await MemoryCogs.memory_server_show.callback(cog, as_interaction(fake=interaction))
    [sent] = interaction.response.sent
    assert sent["ephemeral"] is True
    assert isinstance(sent["embed"], Embed)


async def test_memory_server_show_refuses_a_server_the_bot_is_not_in() -> None:
    """A user install reaches servers the bot was never added to; the refusal says so."""
    write_fact(scope=SERVER_SCOPE, fact=_server_fact(fact_id="a" * 16, text="大家都很愛玩楓之谷"))
    cog = make_memory_cog()
    interaction = FakeInteraction(guild_id=GUILD_ID)
    interaction.guild = None
    await MemoryCogs.memory_server_show.callback(cog, as_interaction(fake=interaction))
    assert interaction.response.sent[-1]["ephemeral"] is True
    embed = interaction.response.sent[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "楓之谷" not in (embed.description or "")
