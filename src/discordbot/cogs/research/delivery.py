"""Delivers a finished research report into its Discord thread.

Posts the report both as section-chunked inline messages and as a `research.md` attachment (plus
`research.png` when present) on the final chunk alongside the owner ping and usage footer. Edits
the opening status message into the first chunk; every send is best-effort.
"""

from typing import TYPE_CHECKING

import logfire
from nextcord import File, Object, Message, AllowedMentions

from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT
from discordbot.utils.discord_errors import DiscordLogLevel, report_discord_failure
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaDeliveryPlanner,
    upload_limit_for,
)

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
    # Plan research.md before research.png against a shared remaining budget so .md keeps native
    # priority while combined attachments stay within upload_limit - MEDIA_ENVELOPE_MARGIN.
    remaining_limit = upload_limit_for(guild=thread.guild)
    items: list[MediaItem] = [MediaItem(source=report.encode("utf-8"), filename="research.md")]
    if result.image_bytes is not None:
        items.append(MediaItem(source=result.image_bytes, filename="research.png"))
    files: list[File] = []
    hosted_urls: list[str] = []
    for item in items:
        item_plan = await media_delivery.plan(
            items=[item], upload_limit=remaining_limit, envelope_margin=MEDIA_ENVELOPE_MARGIN
        )
        remaining_limit = max(remaining_limit - sum(native.size for native in item_plan.native), 0)
        files.extend(native.to_file() for native in item_plan.native)
        hosted_urls.extend(item_plan.hosted_urls)
        if item_plan.dropped_items:
            logfire.warn(
                "research report file too big to attach and not hosted; left out",
                thread_id=thread.id,
                filename=item.filename,
            )
    # Place ping and hosted URLs before the blank-line-delimited footer so USAGE_FOOTER_RE strips
    # the footer; overflow into a trailing message if needed.
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
        except Exception as exc:
            # Broad: any Discord failure here is recoverable by the fallback send below.
            report_discord_failure(
                error=exc,
                not_found_message="failed to edit research status into report",
                not_found_level="warn",
                not_found_exc_info=True,
                forbidden_message="research thread refused the report edit",
                other_message="failed to edit research status into report",
                thread_id=thread.id,
                chunk_index=chunk_index,
            )
        # Rewind each File after a failed edit: Message.edit reads fp without rewinding.
        for file in files:
            file.reset(seek=True)
    try:
        if files:
            await thread.send(content=content, files=files, allowed_mentions=allowed_mentions)
        else:
            await thread.send(content=content, allowed_mentions=allowed_mentions)
    except Exception as exc:
        # Broad: best-effort final send must not abort phase bookkeeping; losing the last chunk
        # drops the attachments and ping.
        other_level: DiscordLogLevel = "error" if is_last else "warn"
        report_discord_failure(
            error=exc,
            not_found_message="failed to post research report message",
            not_found_level=other_level,
            not_found_exc_info=True,
            forbidden_message="research thread refused a report message",
            other_message="failed to post research report message",
            other_level=other_level,
            thread_id=thread.id,
            chunk_index=chunk_index,
            is_last=is_last,
            has_files=bool(files),
        )
