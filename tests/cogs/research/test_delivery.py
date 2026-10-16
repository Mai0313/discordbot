"""Tests for delivering a research report: splitting, files, footer and mention policy."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from pathlib import Path

import pytest
from nextcord import File, Thread, AllowedMentions

from discordbot.cogs.research import agent
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.llm_transcript import render_usage_footer
from discordbot.cogs.research.delivery import (
    split_report,
    deliver_report,
    owner_allowed_mentions,
    split_report_by_sections,
)

from tests.helpers.casting import as_bot, as_message, make_forbidden
from tests.helpers.research import _THREAD_ID, _RunStatus, _RunThread
from tests.helpers.link_sources import hosting_planner, hosting_off_planner
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from discordbot.cogs.gen_reply.attachment.base import AttachmentRenderer

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


def test_split_report_by_sections_splits_on_thematic_breaks() -> None:
    chunks = split_report_by_sections(text="## A\n\nAlpha body\n\n---\n\n## B\n\nBeta body")
    assert chunks == ["## A\n\nAlpha body", "## B\n\nBeta body"]


def test_split_report_by_sections_subsplits_oversized_section() -> None:
    chunks = split_report_by_sections(text="intro\n\n---\n\n" + "X" * 2500, limit=2000)
    assert chunks[0] == "intro"
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert "".join(chunks[1:]) == "X" * 2500
    assert len(chunks) == 3


def test_split_report_by_sections_falls_back_to_paragraph_packing() -> None:
    text = f"{'A' * 1200}\n\n{'B' * 1200}"
    assert split_report_by_sections(text=text) == split_report(text=text)


def test_split_report_by_sections_ignores_break_inside_code_fence() -> None:
    chunks = split_report_by_sections(text="before\n\n```\n---\n```\n\nafter")
    assert len(chunks) == 1
    assert "---" in chunks[0]


def test_split_report_by_sections_keeps_table_delimiter_row() -> None:
    chunks = split_report_by_sections(text="| Col | Val |\n| --- | --- |\n| a | 1 |")
    assert len(chunks) == 1


def test_split_report_by_sections_keeps_setext_heading() -> None:
    chunks = split_report_by_sections(text="Heading\n---\n\nbody")
    assert len(chunks) == 1


def test_split_report_by_sections_drops_empty_sections() -> None:
    chunks = split_report_by_sections(text="---\n\nonly body\n\n---")
    assert chunks == ["only body"]


def test_owner_allowed_mentions_blocks_everyone_and_roles() -> None:
    mentions = owner_allowed_mentions(owner_id=42)
    assert mentions.everyone is False
    assert mentions.roles is False
    users = mentions.users
    assert isinstance(users, list)
    assert [obj.id for obj in users] == [42]


# ----- delivery completion footer -----------------------------------------------------------


def _completed_result(report_text: str, image_bytes: bytes | None = None) -> agent.ResearchResult:
    return agent.ResearchResult(
        status="completed", report_text=report_text, image_bytes=image_bytes
    )


async def test_delivery_keeps_footer_message_under_the_limit() -> None:
    thread = _RunThread()
    status = _RunStatus(thread=thread)
    footer = "-# antigravity-preview-09-2026 · ⬆ 0 ⬇ 0 · $0.00000000"
    # A report chunk that sits just under the 2000-char message cap; appending the footer inline
    # would overflow, so it must ride its own trailing message.
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text="X" * 1990),
        footer=footer,
        media_delivery=hosting_off_planner(),
    )
    assert all(len(str(write["content"])) <= 2000 for write in thread.writes)
    # The footer + owner ping + research.md ride the trailing send, not the near-limit chunk.
    assert thread.sends == 1
    footer_send = thread.writes[-1]
    assert "<@1>" in str(footer_send["content"])
    assert footer in str(footer_send["content"])
    assert footer_send["files"]
    # Every report message carries the owner-only mention policy so agent text can't mass-ping.
    for write in thread.writes:
        mentions = cast("AllowedMentions", write["allowed_mentions"])
        assert mentions.everyone is False
        assert mentions.roles is False
        assert isinstance(mentions.users, list)
        assert [user.id for user in mentions.users] == [1]


async def test_delivery_inlines_footer_for_short_reports() -> None:
    thread = _RunThread()
    status = _RunStatus(thread=thread)
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text="# Report\nbody"),
        footer="-# footer",
        media_delivery=hosting_off_planner(),
    )
    # One message: the opening status edited into report + footer + the research.md attachment.
    assert thread.sends == 0
    assert len(thread.writes) == 1
    assert "<@1>" in str(thread.writes[0]["content"])
    assert thread.writes[0]["files"]


async def test_delivery_hosts_oversized_report_file(tmp_path: Path) -> None:
    """A report file too big to attach is hosted and its URL linked instead of silently dropped."""
    thread = _RunThread()
    status = _RunStatus(thread=thread)
    thread.guild = SimpleNamespace(filesize_limit=4)  # tiny ceiling so research.md is oversize
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text="# Report\nbody"),
        footer="-# footer",
        media_delivery=hosting_planner(serve_dir=tmp_path),
    )
    # The report .md was hosted (no native attachment); its URL rides the message content.
    edit = thread.writes[0]
    assert not edit.get("files")
    content = str(edit["content"])
    assert any(line.startswith("https://media.test/") for line in content.splitlines())


async def test_delivery_attaches_both_files_when_each_fits_but_combined_over() -> None:
    """Host-off contract: md + png that each fit but jointly exceed the limit BOTH attach natively.

    Routing both through one `plan()` call would fire the planner's combined-peel and drop the
    larger (the report), so delivery decides each attachment on its own.
    """
    thread = _RunThread()
    status = _RunStatus(thread=thread)
    thread.guild = SimpleNamespace(filesize_limit=100)  # each file fits, md + png together do not
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text="R" * 60, image_bytes=b"x" * 60),
        footer="-# footer",
        media_delivery=hosting_off_planner(),
    )
    edit = thread.writes[0]
    files = edit["files"]
    assert isinstance(files, list)
    assert len(files) == 2  # research.md AND research.png both attached, neither dropped
    assert "https://" not in str(edit["content"])  # nothing was hosted


async def test_delivery_names_a_report_file_it_leaves_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """With hosting off, a file too big to attach is left out, and the log says which one."""
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    thread = _RunThread()
    status = _RunStatus(thread=thread)
    thread.guild = SimpleNamespace(filesize_limit=4)  # tiny ceiling so research.md is oversize
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text="# Report\nbody"),
        footer="-# footer",
        media_delivery=hosting_off_planner(),
    )
    assert not thread.writes[0].get("files")
    assert [fields for _, fields in warns] == [
        {"thread_id": _THREAD_ID, "filename": "research.md"}
    ]


@pytest.mark.parametrize(
    ("report_text", "hosted"),
    [("# Report\nbody", False), ("X" * 1990, False), ("# Report\nbody", True)],
    ids=["inline", "own_message", "hosted_file"],
)
async def test_a_delivered_reports_usage_footer_never_reaches_the_bots_history(
    tmp_path: Path, report_text: str, hosted: bool
) -> None:
    """Every delivered shape renders back as the bot's own history with the footer gone."""
    thread = _RunThread()
    status = _RunStatus(thread=thread)
    planner = hosting_off_planner()
    if hosted:
        thread.guild = SimpleNamespace(filesize_limit=4)  # tiny ceiling so research.md is oversize
        planner = hosting_planner(serve_dir=tmp_path)
    footer, _ = render_usage_footer(
        model_name="m", label="antigravity-preview-09-2026", input_tokens=1234, output_tokens=567
    )
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text=report_text),
        footer=footer,
        media_delivery=planner,
    )
    posted = [str(write["content"]) for write in thread.writes]
    assert footer in posted[-1]
    builder = MessageInputBuilder(
        bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))),
        runtime_models=RuntimeModelCatalog(),
        # Rendering a message's text never reaches its attachments.
        attachment_handler=cast("AttachmentRenderer", SimpleNamespace()),
    )

    history = [
        await builder.get_cleaned_content(
            message=as_message(
                fake=SimpleNamespace(
                    content=content,
                    author=SimpleNamespace(id=999),
                    embeds=[],
                    snapshots=[],
                    is_system=lambda: False,
                )
            )
        )
        for content in posted
    ]

    assert not any("⬆" in text for text in history)
    # Only the footer goes: the report, the owner ping and any hosted link all stay.
    assert history == [content.replace(f"\n\n{footer}", "").strip() for content in posted]


async def test_a_refused_report_is_a_warn_even_on_its_last_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last chunk logs `error` for a real failure, but a refusal is the server's setting."""
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    errors = capture_logs(monkeypatch=monkeypatch, level="error")
    thread = _RunThread(error=make_forbidden(message="Missing Permissions"), status_posts=False)

    await deliver_report(
        thread=cast("Thread", thread),
        status=None,
        owner_id=1,
        result=_completed_result(report_text="report"),
        footer="-# footer",
        media_delivery=hosting_off_planner(),
    )

    assert errors == []
    assert [fields for _, fields in warns] == [
        {"thread_id": _THREAD_ID, "chunk_index": 0, "is_last": True}
    ]


class _RefusingStatus:
    """An opening status message whose edit Discord refuses after reading the upload body."""

    async def edit(self, files: list[File] | None = None, **kwargs: object) -> None:
        """Consumes every file the way a real multipart edit does, then refuses."""
        del kwargs
        for file in files or []:
            file.fp.read()
        raise make_forbidden(message="Missing Permissions")


class _ReadingThread:
    """A thread that records what each attached file actually carried when it was sent."""

    id = 1

    def __init__(self) -> None:
        """Initializes the guild upload limit and the bodies read off each send's files."""
        self.guild = SimpleNamespace(filesize_limit=10 * 1024 * 1024)
        self.bodies: list[bytes] = []

    async def send(self, files: list[File] | None = None, **kwargs: object) -> None:
        """Reads every attached file, as the upload would."""
        del kwargs
        self.bodies.extend(file.fp.read() for file in files or [])


async def test_a_refused_status_edit_still_hands_the_fallback_a_full_report_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refused edit already read the file, so the fallback send must get it rewound."""
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    thread = _ReadingThread()

    await deliver_report(
        thread=cast("Thread", thread),
        status=as_message(fake=_RefusingStatus()),
        owner_id=1,
        result=_completed_result(report_text="the whole report"),
        footer="-# footer",
        media_delivery=hosting_off_planner(),
    )

    assert thread.bodies == [b"the whole report"]
    assert [fields for _, fields in warns] == [{"thread_id": 1, "chunk_index": 0}]
