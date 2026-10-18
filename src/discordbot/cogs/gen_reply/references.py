"""One-hop reply resolution and URL extraction over model-visible message text.

Every scanner here reads the text a message renders to the model rather than raw nextcord
fields, so URL detection never triggers on a link the answer model cannot see.
"""

import re

from nextcord import Message

from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.services.platforms.youtube import YOUTUBE_URL_RE
from discordbot.cogs.gen_reply.link_sources import LinkUrlFilter, LinkContextSource


def replied_to_message(message: Message) -> Message | None:
    """Returns the message `message` directly replies to, or None when it is not a reply.

    Resolves one hop from `message.reference.resolved`. Discord does not nest `referenced_message`
    payloads, and older ancestors in the same channel are already covered by channel history.

    Args:
        message (Message): Triggering Discord message.
    """
    if message.reference is None:
        return None
    resolved = message.reference.resolved
    return resolved if isinstance(resolved, Message) else None


def message_link_texts(message: Message, strip_usage_footer: bool) -> list[str]:
    """Returns the text spans `message` renders to the model, for URL scanning.

    Matches `MessageInputBuilder`: `content` takes precedence, embeds are included only when
    `content` is empty, and forwarded messages contribute `snapshot_text` from `message.snapshots`.

    Args:
        message (Message): Discord message to extract rendered text spans from.
        strip_usage_footer (bool): Whether to strip the bot's usage footer from each span.
    """
    content = message.content or ""
    content_present = bool(content.strip())
    # Strip before `.strip()` and again on snapshot spans: `USAGE_FOOTER_RE` anchors on the leading
    # blank line, which `.strip()` would remove on a footer-only message. `content_present` is read
    # before stripping so a footer-only message still counts as having content and skips embeds.
    if strip_usage_footer:
        content = USAGE_FOOTER_RE.sub("", content)
    content = content.strip()
    texts = [content]
    if not content_present:
        texts.append(MessageInputBuilder.extract_embed_text(embeds=list(message.embeds)))
    for snapshot in message.snapshots:
        texts.append(MessageInputBuilder.snapshot_text(snapshot=snapshot))
    if strip_usage_footer:
        return [USAGE_FOOTER_RE.sub("", text).strip() for text in texts]
    return texts


def authored_link_texts(message: Message) -> list[str]:
    """Returns the human-authored text spans of `message` with usage footers stripped.

    Excludes embeds on both `message` and `message.snapshots` so scanning a replied-to message
    reads only links in `content` and never a link from an expansion card's embed chain.

    Args:
        message (Message): Replied-to Discord message to scan.
    """
    spans = [message.content or "", *(snapshot.content for snapshot in message.snapshots)]
    return [USAGE_FOOTER_RE.sub("", span).strip() for span in spans]


def _first_url_match(
    pattern: re.Pattern[str], texts: list[str], url_filter: LinkUrlFilter | None
) -> re.Match[str] | None:
    """First match of a URL pattern across one message's already-rendered text spans.

    A match `url_filter` refuses is skipped rather than ending the scan, so a link the source
    cannot read never hides one after it that it can.
    """
    for text in texts:
        for match in pattern.finditer(string=text):
            if url_filter is None or url_filter(url=match.group(0)):
                return match
    return None


def link_url_for_source(source: LinkContextSource, message: Message) -> str | None:
    """Returns the first readable URL for `source` in `message` or its replied-to message.

    Scans `message` via `message_link_texts` first; when `source.search_replied_to_message` is
    enabled, falls back to scanning `replied_to_message(message)` via `authored_link_texts`.
    Matches rejected by `source.url_filter` are skipped.

    Args:
        source (LinkContextSource): Link-context source definition with pattern and filter rules.
        message (Message): Triggering Discord message.
    """
    match = _first_url_match(
        pattern=source.url_pattern,
        texts=message_link_texts(message=message, strip_usage_footer=False),
        url_filter=source.url_filter,
    )
    if match is None and source.search_replied_to_message:
        replied_to = replied_to_message(message=message)
        if replied_to is not None:
            match = _first_url_match(
                pattern=source.url_pattern,
                texts=authored_link_texts(message=replied_to),
                url_filter=source.url_filter,
            )
    return match.group(0) if match else None


def _youtube_url_in_message(message: Message, strip_usage_footer: bool) -> str | None:
    """Returns the first YouTube URL in a message's text, embeds, or forwarded snapshots, if any."""
    match = _first_url_match(
        pattern=YOUTUBE_URL_RE,
        texts=message_link_texts(message=message, strip_usage_footer=strip_usage_footer),
        url_filter=None,
    )
    return match.group(0) if match else None


def find_youtube_url(message: Message) -> str | None:
    """Returns the first YouTube URL in `message` or the message it replies to, or None.

    Scans embeds on the replied-to message as well as `content`, since a YouTube embed card
    points to the video itself rather than an expansion chain.

    Args:
        message (Message): Triggering Discord message.
    """
    found = _youtube_url_in_message(message=message, strip_usage_footer=False)
    if found is not None:
        return found
    replied_to = replied_to_message(message=message)
    if replied_to is not None:
        return _youtube_url_in_message(message=replied_to, strip_usage_footer=True)
    return None
