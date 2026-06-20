"""Posts a finished research report body into its Discord thread.

The report is long cited markdown. It is delivered two ways at once so nothing is
lost: chunked inline messages (split on paragraph boundaries so citations and
headings survive) for in-thread readability, plus the full report as a `research.md`
File attachment (the durable artifact). A generated chart, if any, rides a follow-up
message. Every send is best-effort. The completion line (usage footer, escalation
buttons, owner ping) is owned by the cog, which edits the opening status message.
"""

import io
from typing import TYPE_CHECKING

import logfire
import nextcord

if TYPE_CHECKING:
    from nextcord import Thread

    from discordbot.cogs._research.agent import ResearchResult

DISCORD_MESSAGE_LIMIT = 2000


def _upload_limit(*, thread: "Thread") -> int:
    """The thread's real upload ceiling (its guild's boost-tier `filesize_limit`).

    A Discord Thread always lives in a guild, so `thread.guild` is never None and is read
    directly (no DM fallback: research never runs in a DM).
    """
    return thread.guild.filesize_limit


def split_report(*, text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Splits report markdown into <=limit chunks, preferring paragraph then line breaks.

    A hard cut at `limit` is the last resort, used only when a single paragraph or line
    is longer than the limit, so normal reports keep their markdown structure intact.
    """
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


async def deliver_report(  # noqa: PLR0913 -- the report body plus its completion-message inputs
    *,
    thread: "Thread",
    status: nextcord.Message | None,
    owner_mention: str,
    result: "ResearchResult",
    footer: str,
    view: nextcord.ui.View | None,
) -> None:
    """Delivers the report into the thread.

    The opening status message ("Researching...") is edited into the first chunk so no message is
    wasted; the remaining chunks follow as new messages; the LAST chunk carries the usage footer,
    the escalation view, the owner ping, and the full report as a `research.md` attachment (plus
    any generated image). Every write is best-effort.
    """
    report = result.report_text.strip() or "(the research returned no report text)"
    chunks = split_report(text=report) or ["(empty report)"]
    limit = _upload_limit(thread=thread)
    last = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        target = status if index == 0 else None
        if index == last:
            await _place(
                status=target,
                thread=thread,
                content=f"{chunk}\n\n{owner_mention}\n{footer}",
                files=_final_files(report=report, image_bytes=result.image_bytes, limit=limit),
                view=view,
            )
        else:
            await _place(status=target, thread=thread, content=chunk, files=[], view=None)


def _final_files(*, report: str, image_bytes: bytes | None, limit: int) -> list[nextcord.File]:
    """The `research.md` (+ optional image) attachments for the last message, honoring the limit."""
    files: list[nextcord.File] = []
    encoded = report.encode("utf-8")
    if len(encoded) <= limit:
        files.append(nextcord.File(fp=io.BytesIO(encoded), filename="research.md"))
    if image_bytes is not None and len(image_bytes) <= limit:
        files.append(nextcord.File(fp=io.BytesIO(image_bytes), filename="research.png"))
    return files


async def _place(
    *,
    status: nextcord.Message | None,
    thread: "Thread",
    content: str,
    files: list[nextcord.File],
    view: nextcord.ui.View | None,
) -> None:
    """Edits the opening status message (when given) or sends a new message, with optional files/view."""
    if status is not None:
        try:
            if files and view is not None:
                await status.edit(content=content, files=files, view=view)
            elif files:
                await status.edit(content=content, files=files)
            elif view is not None:
                await status.edit(content=content, view=view)
            else:
                await status.edit(content=content)
            return
        except Exception:
            logfire.warn("failed to edit research status into report", thread_id=thread.id)
    try:
        if files and view is not None:
            await thread.send(content=content, files=files, view=view)
        elif files:
            await thread.send(content=content, files=files)
        elif view is not None:
            await thread.send(content=content, view=view)
        else:
            await thread.send(content=content)
    except Exception:
        logfire.warn("failed to post research report message", thread_id=thread.id)
