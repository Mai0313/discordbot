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


async def deliver_report(*, thread: "Thread", result: "ResearchResult") -> None:
    """Posts the report chunks + `research.md` + optional image (the report body)."""
    report = result.report_text.strip() or "(the research returned no report text)"
    chunks = split_report(text=report) or ["(empty report)"]
    limit = _upload_limit(thread=thread)
    encoded = report.encode("utf-8")
    md_file = (
        nextcord.File(fp=io.BytesIO(encoded), filename="research.md")
        if len(encoded) <= limit
        else None
    )
    for index, chunk in enumerate(chunks):
        try:
            if index == 0 and md_file is not None:
                await thread.send(content=chunk, file=md_file)
            else:
                await thread.send(content=chunk)
        except Exception:
            logfire.warn(
                "failed to post research report chunk", thread_id=thread.id, chunk_index=index
            )
    if result.image_bytes is not None and len(result.image_bytes) <= limit:
        try:
            await thread.send(
                file=nextcord.File(fp=io.BytesIO(result.image_bytes), filename="research.png")
            )
        except Exception:
            logfire.warn("failed to post research image", thread_id=thread.id)
