"""Posts a finished research report into its Discord thread.

A report is long cited markdown that no single Discord message can hold, so it is delivered two
ways at once: as chunked inline messages for in-thread readability, and as the whole report in one
`research.md` attachment, which is the durable artifact. Chunking cuts on the report's own `---`
thematic breaks first (`split_report_by_sections`), so the posted layout mirrors the written one,
then packs anything still over the message cap on paragraph and then line boundaries
(`split_report`) so headings and citations are not severed mid-line. Any chart the agent drew rides
the same final message as `research.png`; an attachment past the guild's upload ceiling is hosted
as a URL instead (`utils/media_delivery.py`), or silently dropped when hosting is off.

On the success path this module, not the cog, closes out the run's opening status message: it edits
"Researching..." into the first chunk instead of leaving it stranded above the report, and appends
the owner ping plus the usage footer the cog composed to the last chunk. The cog keeps the failure
path (`cog.py::_finalize_status`) and composes the footer text. Every write here is best-effort:
the agent has already spent minutes on the run, so a Discord failure is logged and the remaining
chunks still go out rather than aborting the delivery.
"""

from typing import TYPE_CHECKING

import logfire
from nextcord import File, Message, AllowedMentions

from discordbot.utils.media_delivery import MediaItem, MediaDeliveryPlanner, upload_limit_for

if TYPE_CHECKING:
    from nextcord import Thread

    from discordbot.cogs.research.agent import ResearchResult

DISCORD_MESSAGE_LIMIT = 2000


def _upload_limit(*, thread: "Thread") -> int:
    """The thread's real upload ceiling (its guild's boost-tier `filesize_limit`).

    A Discord Thread always lives in a guild, so `thread.guild` is never None (no DM fallback:
    research never runs in a DM); the shared helper returns its `filesize_limit` directly.

    Args:
        thread (Thread): The research thread the report's attachments are bound for.

    Returns:
        The maximum attachment size in bytes for that thread.
    """
    return upload_limit_for(guild=thread.guild)


def split_report(*, text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Splits report markdown into <=limit chunks, preferring paragraph then line breaks.

    A break landing in the first half of the window is passed over for the next fallback, so a
    chunk is never left half-empty by an early boundary. A hard cut at `limit` is the last resort,
    used only when a single paragraph or line is longer than the limit, so normal reports keep
    their markdown structure intact. Surrounding whitespace is stripped, so blank text yields no
    chunks at all and callers have to supply their own placeholder.

    Args:
        text (str): The report markdown to pack.
        limit (int): Maximum characters per chunk.

    Returns:
        The chunks in report order, each at most `limit` characters.
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
    paragraph packing. A `---`-only line inside a fenced code block, a setext `## heading`
    underline (no blank line above), and a table delimiter row (`| --- |`) are all left intact --
    only a line of pure dashes with a blank line on both sides splits a section.

    Args:
        text (str): The report markdown to split.
        limit (int): Maximum characters per chunk, passed through to `split_report`.

    Returns:
        The chunks of every section concatenated in report order, each at most `limit` characters.
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
    allowed_mentions: AllowedMentions,
    media_delivery: MediaDeliveryPlanner,
) -> None:
    """Delivers the report into the thread.

    The opening status message ("Researching...") is edited into the first chunk so no message is
    wasted; the remaining chunks follow as new messages; the LAST chunk carries the usage footer,
    the owner ping, and the full report as a `research.md` attachment (plus any generated image),
    unless that suffix would push it past Discord's message-length cap, in which case the suffix
    becomes a trailing message of its own. A file too big to upload is hosted on the external
    static server and linked on the last chunk instead of being dropped; with hosting unavailable
    it is silently dropped. Every write is best-effort, so a failure is logged and the remaining
    chunks still go out.

    Args:
        thread (Thread): The research thread every chunk is posted into.
        status (Message | None): The run's opening status message, edited into the first chunk;
            None when that send failed, and then the first chunk is sent as a new message too.
        owner_mention (str): The requester's `<@id>` mention, appended after the report body.
        result (ResearchResult): The settled run, read for its report markdown and chart bytes.
        footer (str): The usage footer the cog rendered, appended after the owner ping.
        allowed_mentions (AllowedMentions): Mention policy for every message here. The body is
            agent-generated, so it may quote `@everyone` / roles / other users; the caller passes
            an owner-only policy so only the requester is pinged.
        media_delivery (MediaDeliveryPlanner): Decides per attachment whether it rides natively,
            becomes a hosted URL, or is dropped.
    """
    report = result.report_text.strip() or "(the research returned no report text)"
    chunks = split_report_by_sections(text=report) or ["(empty report)"]
    # Each attachment (the report `.md`, plus the chart `.png` if any) is decided independently:
    # they are unrelated files, so one is never peeled to a URL just because their *combined* size
    # crosses the limit (the planner's combined-body guard is for a single multi-file edit; here
    # the `.md` is the durable artifact and must attach whenever it individually fits). A file too
    # big on its own is hosted; with hosting off it is silently dropped, exactly its pre-hosting
    # behavior.
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
    # too big to attach) rides the last chunk only when it still fits under Discord's
    # message-length cap; otherwise it becomes its own trailing message so a near-limit final chunk
    # never pushes the send over the limit and drops the chunk / attachment.
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
            allowed_mentions=allowed_mentions,
            chunk_index=index,
            is_last=is_last,
        )


async def _place(  # noqa: PLR0913 -- target message plus its optional files / mention policy
    *,
    status: Message | None,
    thread: "Thread",
    content: str,
    files: list[File],
    allowed_mentions: AllowedMentions,
    chunk_index: int,
    is_last: bool,
) -> None:
    """Edits the opening status message (when given) or sends a new message, with optional files.

    Never raises: a failed edit falls back to a fresh send, and a failed send is only logged, since
    the caller is a fire-and-forget delivery with nobody to raise to. The last chunk is the one
    carrying the attachment, ping and footer, so losing it is an error while losing an earlier one
    only makes the delivery partial, which is why `is_last` picks the log level.

    Args:
        status (Message | None): The message to edit into `content`; None sends a new one instead.
        thread (Thread): The thread to send into, and the id every log line carries.
        content (str): The chunk text to place.
        files (list[File]): Attachments for this message; empty on every chunk but the last.
        allowed_mentions (AllowedMentions): Mention policy for the write.
        chunk_index (int): The chunk's position, logged so a partial delivery is locatable.
        is_last (bool): Whether this chunk carries the deliverable.
    """
    if status is not None:
        try:
            if files:
                await status.edit(content=content, files=files, allowed_mentions=allowed_mentions)
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
        if files:
            await thread.send(content=content, files=files, allowed_mentions=allowed_mentions)
        else:
            await thread.send(content=content, allowed_mentions=allowed_mentions)
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
