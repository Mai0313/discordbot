"""Posts a finished research report body into its Discord thread.

The report is long cited markdown. It is delivered two ways at once so nothing is
lost: chunked inline messages (split on paragraph boundaries so citations and
headings survive) for in-thread readability, plus the full report as a `research.md`
File attachment (the durable artifact). A generated chart, if any, rides a follow-up
message. Every send is best-effort. The completion line (usage footer, escalation
buttons, owner ping) is owned by the cog, which edits the opening status message.
"""

from typing import TYPE_CHECKING

import logfire
from nextcord import File, Message, AllowedMentions
from nextcord.ui import View

from discordbot.utils.media_delivery import MediaItem, MediaDeliveryPlanner, upload_limit_for

if TYPE_CHECKING:
    from nextcord import Thread

    from discordbot.cogs._research.agent import ResearchResult

DISCORD_MESSAGE_LIMIT = 2000


def _upload_limit(*, thread: "Thread") -> int:
    """The thread's real upload ceiling (its guild's boost-tier `filesize_limit`).

    A Discord Thread always lives in a guild, so `thread.guild` is never None (no DM fallback:
    research never runs in a DM); the shared helper returns its `filesize_limit` directly.
    """
    return upload_limit_for(guild=thread.guild)


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


def split_report_by_sections(*, text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Splits a report into one chunk list per `---` section, packing each section under `limit`.

    The report body is model-generated cited markdown whose major sections are divided by
    blank-surrounded thematic breaks (a line of `---`). Each section becomes its own Discord
    message so the delivered layout mirrors the report's structure; the separator line itself is
    dropped. A section still longer than `limit` is sub-packed by `split_report` (paragraph then
    line boundaries, a hard cut only as a last resort), and empty sections yield nothing. A report
    with no thematic break is a single section, so its output is byte-for-byte `split_report`'s
    paragraph packing (today's behavior). A `---`-only line inside a fenced code block, a setext
    `## heading` underline (no blank line above), and a table delimiter row (`| --- |`) are all
    left intact -- only a line of pure dashes with a blank line on both sides splits a section.
    """
    lines = text.split("\n")
    sections: list[str] = []
    current: list[str] = []
    in_fence = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        prev_blank = index == 0 or lines[index - 1].strip() == ""
        next_blank = index == len(lines) - 1 or lines[index + 1].strip() == ""
        stripped = line.strip()
        is_break = (
            not in_fence
            and prev_blank
            and next_blank
            and len(stripped) >= 3
            and set(stripped) == {"-"}
        )
        if is_break:
            sections.append("\n".join(current))
            current = []
        else:
            current.append(line)
    sections.append("\n".join(current))

    chunks: list[str] = []
    for section in sections:
        chunks.extend(split_report(text=section, limit=limit))
    return chunks


async def deliver_report(  # noqa: PLR0913 -- the report body plus its completion-message inputs
    *,
    thread: "Thread",
    status: Message | None,
    owner_mention: str,
    result: "ResearchResult",
    footer: str,
    view: View | None,
    allowed_mentions: AllowedMentions,
    media_delivery: MediaDeliveryPlanner,
) -> None:
    """Delivers the report into the thread.

    The opening status message ("Researching...") is edited into the first chunk so no message is
    wasted; the remaining chunks follow as new messages; the LAST chunk carries the usage footer,
    the escalation view, the owner ping, and the full report as a `research.md` attachment (plus
    any generated image). A report file too big to upload is hosted on the external static server
    and linked on the last chunk instead of being dropped; if hosting is unavailable it degrades
    to today's silent drop. Every write is best-effort.

    `allowed_mentions` restricts the report body (agent-generated, so it may quote `@everyone` /
    roles / other users) to ping only the owner; the caller passes an owner-only policy.
    """
    report = result.report_text.strip() or "(the research returned no report text)"
    chunks = split_report_by_sections(text=report) or ["(empty report)"]
    # Each attachment (the report `.md`, plus the chart `.png` if any) is decided independently:
    # they are unrelated files, so one is never peeled to a URL just because their *combined* size
    # crosses the limit (the planner's combined-body guard is for a single multi-file edit; here the
    # `.md` is the durable artifact and must attach whenever it individually fits). A file too big on
    # its own is hosted; with hosting off it is silently dropped, exactly as before this fold-in.
    limit = _upload_limit(thread=thread)
    items: list[MediaItem] = [MediaItem(source=report.encode("utf-8"), filename="research.md")]
    if result.image_bytes is not None:
        items.append(MediaItem(source=result.image_bytes, filename="research.png"))
    files: list[File] = []
    hosted_urls: list[str] = []
    for item in items:
        item_plan = await media_delivery.plan(items=[item], upload_limit=limit)
        files.extend(native.to_file() for native in item_plan.native)
        hosted_urls.extend(item_plan.hosted_urls)
    # The completion suffix (owner ping + usage footer, plus a hosted-URL line for any report file
    # too big to attach) rides the last chunk only when it still fits under Discord's message-length
    # cap; otherwise it becomes its own trailing message so a near-limit final chunk never pushes the
    # send over the limit and drops the chunk / attachment / buttons.
    hosted_lines = ("\n" + "\n".join(hosted_urls)) if hosted_urls else ""
    suffix = f"\n\n{owner_mention}\n{footer}{hosted_lines}"
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
            chunk_index=index,
            is_last=is_last,
        )


async def _place(  # noqa: PLR0913 -- target message plus its optional files / view / mention policy
    *,
    status: Message | None,
    thread: "Thread",
    content: str,
    files: list[File],
    view: View | None,
    allowed_mentions: AllowedMentions,
    chunk_index: int,
    is_last: bool,
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
        except Exception as exc:
            # Broad: any Discord failure here is recoverable by the fallback send below.
            logfire.warn(
                "failed to edit research status into report",
                thread_id=thread.id,
                chunk_index=chunk_index,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
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
    except Exception as exc:
        # Broad on purpose: last resort of a best-effort delivery, so it must not abort the
        # caller's phase bookkeeping. Only the last chunk carries the file, ping, footer and
        # escalation view, so losing it breaks the deliverable; an earlier one is partial.
        log = logfire.error if is_last else logfire.warn
        log(
            "failed to post research report message",
            thread_id=thread.id,
            chunk_index=chunk_index,
            is_last=is_last,
            has_files=bool(files),
            has_view=view is not None,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
