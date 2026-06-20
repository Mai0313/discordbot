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
    allowed_mentions: nextcord.AllowedMentions,
) -> None:
    """Delivers the report into the thread.

    The opening status message ("Researching...") is edited into the first chunk so no message is
    wasted; the remaining chunks follow as new messages; the LAST chunk carries the usage footer,
    the escalation view, the owner ping, and the full report as a `research.md` attachment (plus
    any generated image). Every write is best-effort.

    `allowed_mentions` restricts the report body (agent-generated, so it may quote `@everyone` /
    roles / other users) to ping only the owner; the caller passes an owner-only policy.
    """
    report = result.report_text.strip() or "(the research returned no report text)"
    chunks = split_report(text=report) or ["(empty report)"]
    limit = _upload_limit(thread=thread)
    files = _final_files(report=report, image_bytes=result.image_bytes, limit=limit)
    # The completion suffix (owner ping + usage footer) rides the last chunk only when it still fits
    # under Discord's message-length cap; otherwise it becomes its own trailing message so a near-limit
    # final chunk never pushes the send over the limit and drops the chunk / attachment / buttons.
    suffix = f"\n\n{owner_mention}\n{footer}"
    if len(chunks[-1]) + len(suffix) <= DISCORD_MESSAGE_LIMIT:
        chunks[-1] = f"{chunks[-1]}{suffix}"
    else:
        chunks.append(suffix.lstrip("\n"))
    last = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        is_last = index == last
        await _place(
            status=status if index == 0 else None,
            thread=thread,
            content=chunk,
            files=files if is_last else [],
            view=view if is_last else None,
            allowed_mentions=allowed_mentions,
        )


def _final_files(*, report: str, image_bytes: bytes | None, limit: int) -> list[nextcord.File]:
    """The `research.md` (+ optional image) attachments for the last message, honoring the limit."""
    files: list[nextcord.File] = []
    encoded = report.encode("utf-8")
    if len(encoded) <= limit:
        files.append(nextcord.File(fp=io.BytesIO(encoded), filename="research.md"))
    if image_bytes is not None and len(image_bytes) <= limit:
        files.append(nextcord.File(fp=io.BytesIO(image_bytes), filename="research.png"))
    return files


async def _place(  # noqa: PLR0913 -- target message plus its optional files / view / mention policy
    *,
    status: nextcord.Message | None,
    thread: "Thread",
    content: str,
    files: list[nextcord.File],
    view: nextcord.ui.View | None,
    allowed_mentions: nextcord.AllowedMentions,
) -> None:
    """Edits the opening status message (when given) or sends a new message, with optional files/view."""
    if status is not None:
        try:
            if files and view is not None:
                await status.edit(
                    content=content, files=files, view=view, allowed_mentions=allowed_mentions
                )
            elif files:
                await status.edit(content=content, files=files, allowed_mentions=allowed_mentions)
            elif view is not None:
                await status.edit(content=content, view=view, allowed_mentions=allowed_mentions)
            else:
                await status.edit(content=content, allowed_mentions=allowed_mentions)
            return
        except Exception:
            logfire.warn("failed to edit research status into report", thread_id=thread.id)
    try:
        if files and view is not None:
            await thread.send(
                content=content, files=files, view=view, allowed_mentions=allowed_mentions
            )
        elif files:
            await thread.send(content=content, files=files, allowed_mentions=allowed_mentions)
        elif view is not None:
            await thread.send(content=content, view=view, allowed_mentions=allowed_mentions)
        else:
            await thread.send(content=content, allowed_mentions=allowed_mentions)
    except Exception:
        logfire.warn("failed to post research report message", thread_id=thread.id)
