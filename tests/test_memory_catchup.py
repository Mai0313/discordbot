"""Tests for `/memory server catchup`: reading a channel into its server's memory."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
import asyncio
from pathlib import Path
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from nextcord import Embed, TextChannel
from pydantic import BaseModel
from nextcord.utils import utcnow

from discordbot.typings.colors import DISCORD_RED, DISCORD_GREEN, DISCORD_YELLOW
from discordbot.typings.models import ModelSettings
from discordbot.cogs.memory.cog import MemoryCogs, _catchup_tasks
from discordbot.services.memory.store import server_scope, count_raw_entries
from discordbot.services.memory.writer import (
    MemoryWriterAI,
    RawMemoryDraft,
    MemoryObservation,
    ServerCatchupNotes,
    server_subject,
    transcript_from_messages,
)
from discordbot.services.memory.catchup import catchup_on_cooldown, review_catchup_notes

from tests.helpers.memory import FakeMemoryClient, make_memory_cog
from tests.helpers.casting import as_interaction
from tests.helpers.discord_mocks import DiscordPayload, FakeInteraction

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from discordbot.typings.memory import MemoryCategory, MemoryEvidenceKind

GUILD_ID = 777
BOT_ID = 999
SCOPE = server_scope(server_id=GUILD_ID)
SUBJECT = server_subject(server_id=GUILD_ID)
MODEL = ModelSettings(name="test-memories-model", effort="minimal")


def _observation(summary: str) -> MemoryObservation:
    """One community observation the review keeps."""
    return MemoryObservation(
        category=cast("MemoryCategory", "stable_fact"),
        subject_is_target_user=True,
        evidence_kind=cast("MemoryEvidenceKind", "stable_fact"),
        confidence="high",
        durability="stable",
        promotion_eligible=True,
        normalized_key="culture.friday_valorant",
        sharing="global",
        summary_zh=summary,
        evidence_quote="週五老樣子",
        ttl_days=None,
    )


def _writer(
    notes: tuple[str, ...] | None, draft: RawMemoryDraft | None
) -> tuple[MemoryWriterAI, FakeMemoryClient]:
    """A writer whose proposal answers `notes` and whose review answers `draft`."""
    client = FakeMemoryClient()

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Answers each call by the schema it asked for."""
        del body
        if text_format is ServerCatchupNotes:
            return None if notes is None else ServerCatchupNotes(notes=notes)
        return draft

    client.responses.answer = answer
    return MemoryWriterAI(client=cast("AsyncOpenAI", client), model=MODEL), client


def _jump_url(message_id: int) -> str:
    """The link Discord gives a message in the test channel."""
    return f"https://discord.com/channels/{GUILD_ID}/20/{message_id}"


def _member_message(content: str, message_id: int = 1) -> SimpleNamespace:
    """A member's message as the history walk sees it."""
    author = SimpleNamespace(id=1, bot=False, display_name="小李", name="lee")
    return SimpleNamespace(
        id=message_id,
        jump_url=_jump_url(message_id=message_id),
        author=author,
        content=content,
        embeds=[],
    )


def _bot_result(color: int, message_id: int, read_until: int) -> SimpleNamespace:
    """One of the bot's catchup results, linking to the newest message its run read."""
    author = SimpleNamespace(id=BOT_ID, bot=True, display_name="破貓", name="pocat")
    embed = Embed(title="📚 跟上進度", color=color, url=_jump_url(message_id=read_until))
    return SimpleNamespace(
        id=message_id,
        jump_url=_jump_url(message_id=message_id),
        author=author,
        content="",
        embeds=[embed],
    )


def _channel(messages: list[SimpleNamespace], public: bool = True) -> MagicMock:
    """A guild text channel whose history answers `messages`, newest first."""
    channel = MagicMock(spec=TextChannel)
    channel.id = 20
    channel.permissions_for.return_value = SimpleNamespace(view_channel=public)

    async def history(limit: int) -> object:
        """Yields the channel's messages, newest first, up to `limit`."""
        for message in messages[:limit]:
            yield message

    channel.history = history
    return channel


class _CatchupInteraction(FakeInteraction):
    """A slash command in a guild text channel, carrying what catchup reads off it."""

    def __init__(self, channel: MagicMock, readable: bool = True) -> None:
        """Places the command in `channel`, with or without read-history permission."""
        super().__init__(guild_id=GUILD_ID)
        self.guild = SimpleNamespace(id=GUILD_ID, name="My Server", default_role=object())
        self.channel = channel
        self.app_permissions = SimpleNamespace(view_channel=True, read_message_history=readable)
        self.client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID))
        self.expires_at = utcnow() + timedelta(minutes=15)


async def _catch_up(cog: MemoryCogs, interaction: _CatchupInteraction) -> None:
    """Runs the command and waits out the run it started, if any."""
    await MemoryCogs.memory_server_catchup.callback(cog, as_interaction(fake=interaction))
    running = _catchup_tasks.get(key=GUILD_ID)
    if running is not None:
        await running


def _cog(writer: MemoryWriterAI) -> MemoryCogs:
    """The memory cog with `writer` in place of the proxy-backed one."""
    cog = make_memory_cog()
    cog.memory_writer = writer
    return cog


def _last_embed(payloads: list[DiscordPayload]) -> Embed:
    """The embed of the last recorded send."""
    embed = payloads[-1]["embed"]
    assert isinstance(embed, Embed)
    return embed


async def test_catchup_posts_its_start_then_replaces_it_with_what_it_recorded(
    memory_isolated_dir: Path,
) -> None:
    """A run says what it read, stages what the review kept, and posts that in public."""
    writer, _ = _writer(
        notes=("大家每週五晚上一起打 Valorant",),
        draft=RawMemoryDraft(
            has_signal=True, observations=(_observation("每週五晚上打 Valorant"),)
        ),
    )
    interaction = _CatchupInteraction(
        channel=_channel(messages=[_member_message("今晚老樣子"), _member_message("週五打瓦")])
    )
    await _catch_up(cog=_cog(writer=writer), interaction=interaction)

    start = _last_embed(payloads=interaction.response.sent)
    assert "ephemeral" not in interaction.response.sent[-1]
    assert "讀了這個頻道最近的 2 則對話" in (start.description or "")
    result = _last_embed(payloads=interaction.followup.sent)
    assert result.colour is not None
    assert result.colour.value == DISCORD_GREEN
    assert "每週五晚上打 Valorant" in (result.description or "")
    assert interaction.original_deleted
    assert count_raw_entries(scope=SCOPE) == 1
    assert catchup_on_cooldown(scope=SCOPE)


async def test_catchup_reads_on_from_where_the_last_finished_run_stopped(
    memory_isolated_dir: Path,
) -> None:
    """Reading stops at what a green or yellow result read, not at the result.

    What members said while that run worked sits below its result and is read now; a red
    result is read past, since its run recorded nothing.
    """
    writer, client = _writer(notes=(), draft=None)
    history = [
        _member_message("新的一句", message_id=60),
        _bot_result(color=DISCORD_RED, message_id=55, read_until=52),
        _member_message("失敗那次讀過的", message_id=52),
        _bot_result(color=DISCORD_YELLOW, message_id=45, read_until=30),
        _member_message("跑的時候說的", message_id=40),
        _member_message("上次讀過的", message_id=30),
        _member_message("更早的一句", message_id=20),
    ]
    interaction = _CatchupInteraction(channel=_channel(messages=history))
    await _catch_up(cog=_cog(writer=writer), interaction=interaction)

    start = _last_embed(payloads=interaction.response.sent)
    assert "讀了上次跟上進度之後的 3 則新對話" in (start.description or "")
    body = client.responses.parse_bodies[0]
    assert "跑的時候說的" in body
    assert "失敗那次讀過的" in body
    assert "上次讀過的" not in body
    result = _last_embed(payloads=interaction.followup.sent)
    assert result.colour is not None
    assert result.colour.value == DISCORD_YELLOW
    assert result.url == _jump_url(message_id=60)


async def test_catchup_reports_a_failed_review_in_red(memory_isolated_dir: Path) -> None:
    """A failed call is told apart from an empty result, so the next run reads it again."""
    writer, _ = _writer(notes=("大家每週五晚上一起打 Valorant",), draft=None)
    interaction = _CatchupInteraction(channel=_channel(messages=[_member_message("週五打瓦")]))
    await _catch_up(cog=_cog(writer=writer), interaction=interaction)

    result = _last_embed(payloads=interaction.followup.sent)
    assert result.colour is not None
    assert result.colour.value == DISCORD_RED
    assert interaction.original_deleted
    assert count_raw_entries(scope=SCOPE) == 0


async def test_catchup_refuses_a_channel_everyone_cannot_see(memory_isolated_dir: Path) -> None:
    """Server memory is readable by every member, so a private channel is never read."""
    writer, client = _writer(notes=(), draft=None)
    interaction = _CatchupInteraction(
        channel=_channel(messages=[_member_message("秘密")], public=False)
    )
    await _catch_up(cog=_cog(writer=writer), interaction=interaction)

    assert interaction.response.sent[-1]["ephemeral"] is True
    assert "私人頻道" in (_last_embed(payloads=interaction.response.sent).description or "")
    assert client.responses.parse_bodies == []


async def test_catchup_without_read_history_refuses_and_gives_the_cooldown_back(
    memory_isolated_dir: Path,
) -> None:
    """Discord answers an empty page without the permission, so it is checked up front."""
    writer, _ = _writer(notes=(), draft=None)
    interaction = _CatchupInteraction(
        channel=_channel(messages=[_member_message("週五打瓦")]), readable=False
    )
    await _catch_up(cog=_cog(writer=writer), interaction=interaction)

    assert interaction.response.sent[-1]["ephemeral"] is True
    assert "不給權限" in (_last_embed(payloads=interaction.response.sent).description or "")
    assert not catchup_on_cooldown(scope=SCOPE)


async def test_catchup_with_nothing_said_refuses_and_gives_the_cooldown_back(
    memory_isolated_dir: Path,
) -> None:
    """A window of bot messages alone starts no run and leaves no cooldown behind."""
    writer, client = _writer(notes=(), draft=None)
    interaction = _CatchupInteraction(
        channel=_channel(messages=[_bot_result(color=DISCORD_RED, message_id=5, read_until=3)])
    )
    await _catch_up(cog=_cog(writer=writer), interaction=interaction)

    assert interaction.response.sent[-1]["ephemeral"] is True
    assert "記個鬼" in (_last_embed(payloads=interaction.response.sent).description or "")
    assert not catchup_on_cooldown(scope=SCOPE)
    assert client.responses.parse_bodies == []


async def test_catchup_on_cooldown_refuses(memory_isolated_dir: Path) -> None:
    """A second run within the cooldown is refused before the channel is read."""
    writer, _ = _writer(notes=(), draft=None)
    cog = _cog(writer=writer)
    await _catch_up(
        cog=cog,
        interaction=_CatchupInteraction(channel=_channel(messages=[_member_message("週五打瓦")])),
    )
    again = _CatchupInteraction(channel=_channel(messages=[_member_message("再一次")]))
    await _catch_up(cog=cog, interaction=again)

    assert again.response.sent[-1]["ephemeral"] is True
    assert "記過了啦" in (_last_embed(payloads=again.response.sent).description or "")


async def test_catchup_refuses_while_one_is_still_running(memory_isolated_dir: Path) -> None:
    """A run that outlives the cooldown still blocks a second one in the same server."""
    writer, client = _writer(notes=(), draft=None)
    release = asyncio.Event()
    running = asyncio.create_task(release.wait())
    _catchup_tasks.set(key=GUILD_ID, value=cast("asyncio.Task[None]", running))
    interaction = _CatchupInteraction(channel=_channel(messages=[_member_message("週五打瓦")]))
    await MemoryCogs.memory_server_catchup.callback(
        _cog(writer=writer), as_interaction(fake=interaction)
    )
    release.set()
    await running

    assert "已經有人在整理" in (_last_embed(payloads=interaction.response.sent).description or "")
    assert client.responses.parse_bodies == []


async def test_catchup_gives_the_cooldown_back_when_reading_breaks(
    memory_isolated_dir: Path,
) -> None:
    """Any failure before a run starts leaves the server free to try again."""
    writer, _ = _writer(notes=(), draft=None)
    channel = _channel(messages=[])

    async def broken_history(limit: int) -> object:
        """Fails the way a Discord 5xx does once nextcord's retries run out."""
        del limit
        raise RuntimeError("history unavailable")
        yield  # pragma: no cover -- makes this an async generator

    channel.history = broken_history
    interaction = _CatchupInteraction(channel=channel)
    with pytest.raises(RuntimeError):
        await MemoryCogs.memory_server_catchup.callback(
            _cog(writer=writer), as_interaction(fake=interaction)
        )

    assert not catchup_on_cooldown(scope=SCOPE)


async def test_catchup_refuses_a_server_the_bot_is_not_in(memory_isolated_dir: Path) -> None:
    """A user install reaches servers the bot was never added to, which have no memory here."""
    interaction = FakeInteraction(guild_id=GUILD_ID)
    interaction.guild = None
    await MemoryCogs.memory_server_catchup.callback(
        make_memory_cog(), as_interaction(fake=interaction)
    )

    assert interaction.response.sent[-1]["ephemeral"] is True
    assert "把老子加進來" in (_last_embed(payloads=interaction.response.sent).description or "")


async def test_catchup_refuses_in_a_dm(memory_isolated_dir: Path) -> None:
    """There is no server memory to write in a DM."""
    interaction = FakeInteraction(in_guild=False)
    await MemoryCogs.memory_server_catchup.callback(
        make_memory_cog(), as_interaction(fake=interaction)
    )

    assert interaction.response.sent[-1]["ephemeral"] is True
    assert "只能在伺服器" in (_last_embed(payloads=interaction.response.sent).description or "")


async def test_review_catchup_notes_skips_the_review_when_nothing_was_proposed(
    memory_isolated_dir: Path,
) -> None:
    """An empty proposal is the common answer and costs no second call."""
    writer, client = _writer(notes=(), draft=None)
    assert (
        await review_catchup_notes(scope=SCOPE, subject=SUBJECT, transcript="x", writer=writer)
        == ()
    )
    assert len(client.responses.parse_bodies) == 1


async def test_propose_server_notes_keeps_at_most_five() -> None:
    """The cap matches how many server notes one reply may write."""
    writer, _ = _writer(notes=tuple(f"第 {n} 件事" for n in range(8)), draft=None)
    notes = await writer.propose_server_notes(subject=SUBJECT, transcript="x")
    assert notes == tuple(f"第 {n} 件事" for n in range(5))


def test_transcript_without_a_reply_has_no_assistant_block() -> None:
    """A catchup has no reply, and an empty block would tell the reviewer the bot answered."""
    transcript = transcript_from_messages(
        message_list=[{"role": "user", "content": "小李 (lee) [id: 1]: 週五打瓦"}], full_reply=""
    )
    assert "assistant reply" not in transcript
