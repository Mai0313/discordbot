"""What one triggering message points at: the message it replies to, and the links it carries.

Every reader here works off the text a message actually RENDERS to the model, never off raw
nextcord fields, so a URL scanner can never fire on a link the answer model was not shown.
`replied_to_message` is the single place the one-hop reference is resolved, which is what keeps
the route, the media handlers and the streamer agreeing on what "the message being replied to"
means.
"""

import re

from nextcord import Message

from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.services.platforms.youtube import YOUTUBE_URL_RE
from discordbot.cogs.gen_reply.link_sources import LinkUrlFilter, LinkContextSource


def replied_to_message(message: Message) -> Message | None:
    """The message this one replies to, or None when it is not a reply.

    One hop is everything Discord hands over. nextcord fills `MessageReference.resolved` in
    exactly one place, from the `referenced_message` key of the payload it is building, and
    never from the message cache; Discord does not nest that key, so a referenced message's own
    `.reference.resolved` is always `None`. Reaching a grandparent needs an explicit
    `fetch_message` per ancestor, which #593 decided against: an ancestor is another message in
    this same channel, so the history every reply already carries holds it, and a second
    Reference Message block would dilute the one below that says it is the primary context.
    """
    if message.reference is None:
        return None
    resolved = message.reference.resolved
    return resolved if isinstance(resolved, Message) else None


def message_link_texts(message: Message, strip_usage_footer: bool) -> list[str]:
    """The text spans a message actually renders to the model, for URL detection.

    Mirrors `get_cleaned_content` / `snapshot_text`: content takes precedence and an embed is
    rendered (and thus scanned) only when its content is empty. So a URL scanner never fires on a
    link the answer model was not shown, e.g. a captioned forwarded link card whose URL lives only
    in the embed. A forward puts its payload in `message.snapshots`, scanned via `snapshot_text`.

    `strip_usage_footer` removes the bot-authored footer from every span when a caller scans the
    message being replied to. The triggering message keeps its complete author-controlled text.
    """
    content = message.content or ""
    content_present = bool(content.strip())
    # Stripped here and again at the end, and the order is load-bearing rather than redundant:
    # `USAGE_FOOTER_RE` is anchored on the blank line before the footer, so a `.strip()` first
    # destroys that anchor and the whole footer survives — memory labels, and any URL inside
    # them, included. Whether content is PRESENT is read off the raw text above for the same
    # reason: a message that is only a footer has content, it just renders empty.
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
    """The text spans a message's author actually wrote, for scanning a message replied to.

    Narrower than `message_link_texts` by exactly one thing: an embed card never counts,
    neither the message's own nor a forwarded snapshot's. One hop out an embed is a card the
    author did not write, and the bot's own Threads expansion is the common one:
    `parse_threads._build_embeds` emits one permalink per post in the reply chain, ROOT first,
    so a scan keyed on it would read the thread's top post rather than the one the human
    linked — and it disappears entirely when an oversize video pushes hosted URLs into
    `content`. A link a person typed always lives in `content` (or in the content of what they
    forwarded), so nothing human-written is lost. The bot's own replies pass through here too,
    so every span gets the `get_cleaned_content` / `snapshot_text` usage-footer strip: the
    footer carries the memory labels, which are display names their owners choose.
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
    """The URL one link source should read: the current message's, else the replied-to one's.

    The current message always wins. A source that opts into `search_replied_to_message` then
    falls back to the message being replied to, the same one hop `find_youtube_url` takes, so
    "@bot 這篇底下在吵什麼" sent as a reply to someone else's link still reads the post; one
    that does not opt in never looks past the triggering message. That parent is scanned with
    `authored_link_texts`, which is what keeps the bot's own expansion from triggering a read
    of the wrong post.

    A source's `url_filter` rejects a matched link it cannot read (e.g. a Douyin profile or
    live room, whose regex matches the host, not the path), which would only spend a
    rate-limited request to say so. A rejected link is skipped as if it were not there, so it
    neither hides a readable link after it nor stops the fallback to the replied-to message.
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
    """Finds a YouTube URL in the current message or the message it replies to.

    A reply to a message that merely links a video would otherwise be missed, so the parent is
    searched too and "summarize this" on a replied-to video still watches it. The current
    message wins. Unlike `link_url_for_source`, this keeps scanning the parent's embeds: a
    YouTube link card is the link itself, not a rendering of some other post the way a Threads
    expansion is.
    """
    found = _youtube_url_in_message(message=message, strip_usage_footer=False)
    if found is not None:
        return found
    replied_to = replied_to_message(message=message)
    if replied_to is not None:
        return _youtube_url_in_message(message=replied_to, strip_usage_footer=True)
    return None
