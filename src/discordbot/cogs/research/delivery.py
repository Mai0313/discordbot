"""Posts a finished research report body into its Discord thread.

The report is long cited markdown. It is delivered two ways at once so nothing is
lost: chunked inline messages (split on `---` sections, then paragraph boundaries, so
citations and headings survive) for in-thread readability, plus the full report as a
`research.md` File attachment (the durable artifact). A generated chart, if any, rides
the same message as that attachment. Every send is best-effort. The cog supplies the
completion line's parts (usage footer, owner ping); this module places them on the
last chunk, having spent the opening status message on the first.
"""

from typing import TYPE_CHECKING

import logfire
from nextcord import File, Object, Message, Forbidden, AllowedMentions

from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT
from discordbot.utils.media_delivery import MediaItem, MediaDeliveryPlanner, upload_limit_for

if TYPE_CHECKING:
    from nextcord import Thread

    from discordbot.cogs.research.agent import ResearchResult


def owner_allowed_mentions(owner_id: int) -> AllowedMentions:
    """Returns an `AllowedMentions` policy that permits pinging only `owner_id`.

    Args:
        owner_id (int): Discord user ID of the research requester.
    """
    return AllowedMentions(everyone=False, roles=False, users=[Object(id=owner_id)])


def split_report(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Returns `text` split into chunks of at most `limit` characters.

    Prefers paragraph breaks, then line breaks, falling back to a hard cut at `limit` only when a
    single line exceeds `limit`.

    Args:
        text (str): Report markdown to split.
        limit (int): Maximum characters per chunk.
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


def split_report_by_sections(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Returns `text` split on blank-surrounded `---` thematic breaks and packed under `limit`.

    Drops the thematic-break lines themselves while leaving fenced code blocks, setext headings,
    and table delimiter rows intact. Each section is sub-packed via `split_report`.

    Args:
        text (str): Report markdown to split.
        limit (int): Maximum characters per chunk.
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
    thread: "Thread",
    status: Message | None,
    owner_id: int,
    result: "ResearchResult",
    footer: str,
    media_delivery: MediaDeliveryPlanner,
) -> None:
    """Delivers the completed research report into `thread`.

    Edits `status` into the first chunk and sends remaining chunks as new messages, attaching
    `research.md` (and `research.png` when present), the owner ping, and `footer` on the final
    message. Oversized files are hosted via `media_delivery` or omitted with a warning.

    Args:
        thread (Thread): Destination research thread.
        status (Message | None): Opening status message to edit into the first chunk, if any.
        owner_id (int): Discord user ID of the requester to ping on completion.
        result (ResearchResult): Completed research result containing markdown and optional chart.
        footer (str): Usage footer appended to the final message.
        media_delivery (MediaDeliveryPlanner): Planner for attaching or hosting report files.
    """
    report = result.report_text.strip() or "(the research returned no report text)"
    chunks = split_report_by_sections(text=report) or ["(empty report)"]
    # Each attachment (the report `.md`, plus the chart `.png` if any) is decided independently:
    # they are unrelated files, so one is never peeled to a URL just because their *combined* size
    # crosses the limit (the planner's combined-body guard is for a single multi-file edit; here the
    # `.md` is the durable artifact and must attach whenever it individually fits). A file too big on
    # its own is hosted, or left out with a warning when hosting is off or fails.
    # A Discord Thread always lives in a guild (research never runs in a DM), so the shared
    # helper reads its boost-tier `filesize_limit` with no None-guild fallback in play.
    limit = upload_limit_for(guild=thread.guild)
    items: list[MediaItem] = [MediaItem(source=report.encode("utf-8"), filename="research.md")]
    if result.image_bytes is not None:
        items.append(MediaItem(source=result.image_bytes, filename="research.png"))
    files: list[File] = []
    hosted_urls: list[str] = []
    for item in items:
        item_plan = await media_delivery.plan(items=[item], upload_limit=limit)
        files.extend(native.to_file() for native in item_plan.native)
        hosted_urls.extend(item_plan.hosted_urls)
        if item_plan.dropped_items:
            logfire.warn(
                "research report file too big to attach and not hosted; left out",
                thread_id=thread.id,
                filename=item.filename,
            )
    # The completion suffix (owner ping + usage footer, plus a hosted-URL line for any report file
    # too big to attach) rides the last chunk only when it still fits under Discord's message-length
    # cap; otherwise it becomes its own trailing message so a near-limit final chunk never pushes the
    # send over the limit and drops the chunk / attachment. The footer ends the suffix after a
    # blank line, the only place `utils/llm_transcript.py::USAGE_FOOTER_RE` strips it from the
    # bot's own history, so every other line goes before it.
    hosted_lines = ("\n" + "\n".join(hosted_urls)) if hosted_urls else ""
    suffix = f"\n\n<@{owner_id}>{hosted_lines}\n\n{footer}"
    if len(chunks[-1]) + len(suffix) <= DISCORD_MESSAGE_LIMIT:
        chunks[-1] = f"{chunks[-1]}{suffix}"
    else:
        chunks.append(suffix.lstrip("\n"))
    allowed_mentions = owner_allowed_mentions(owner_id=owner_id)
    last = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        is_last = index == last
        await _place(
            status=status if index == 0 else None,
            thread=thread,
            content=chunk,
            files=files if is_last else [],
            allowed_mentions=allowed_mentions,
            chunk_index=index,
            is_last=is_last,
        )


async def _place(  # noqa: PLR0913 -- target message plus its optional files / mention policy
    status: Message | None,
    thread: "Thread",
    content: str,
    files: list[File],
    allowed_mentions: AllowedMentions,
    chunk_index: int,
    is_last: bool,
) -> None:
    """Edits the opening status message (when given) or sends a new message, with optional files."""
    if status is not None:
        try:
            if files:
                await status.edit(content=content, files=files, allowed_mentions=allowed_mentions)
            else:
                await status.edit(content=content, allowed_mentions=allowed_mentions)
            return
        except Forbidden:
            # The bot's permissions in the parent channel, which the thread inherits, changed
            # under the run; the ids are the whole finding.
            logfire.warn(
                "research thread refused the report edit",
                thread_id=thread.id,
                chunk_index=chunk_index,
            )
        except Exception as exc:
            # Broad: any Discord failure here is recoverable by the fallback send below.
            logfire.warn(
                "failed to edit research status into report",
                thread_id=thread.id,
                chunk_index=chunk_index,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        # The failed edit already read each `fp` into its multipart body and `Message.edit`
        # neither closes nor rewinds it, so the fallback below would post empty attachments.
        # Rewinding is what nextcord's own retry loop does with a File it re-sends.
        for file in files:
            file.reset(seek=True)
    try:
        if files:
            await thread.send(content=content, files=files, allowed_mentions=allowed_mentions)
        else:
            await thread.send(content=content, allowed_mentions=allowed_mentions)
    except Forbidden:
        logfire.warn(
            "research thread refused a report message",
            thread_id=thread.id,
            chunk_index=chunk_index,
            is_last=is_last,
        )
    except Exception as exc:
        # Broad on purpose: last resort of a best-effort delivery, so it must not abort the
        # caller's phase bookkeeping. Only the last chunk carries the file, ping and footer, so
        # losing it breaks the deliverable; an earlier one is partial.
        log = logfire.error if is_last else logfire.warn
        log(
            "failed to post research report message",
            thread_id=thread.id,
            chunk_index=chunk_index,
            is_last=is_last,
            has_files=bool(files),
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
