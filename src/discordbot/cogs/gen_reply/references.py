"""What one triggering message points at: the message it replies to, and the links it carries.

Every reader here works off the text a message actually RENDERS to the model, never off raw
nextcord fields, so a URL scanner can never fire on a link the answer model was not shown.
`replied_to_message` is the single place the one-hop reference is resolved, which is what keeps
the route, the media handlers and the streamer agreeing on what "the message being replied to"
means.
"""

import re

from nextcord import Message

from discordbot.utils.youtube import YOUTUBE_URL_RE
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.cogs.gen_reply.link_sources import LinkContextSource


def replied_to_message(*, message: Message) -> Message | None:
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


def message_link_texts(*, message: Message, strip_usage_footer: bool) -> list[str]:
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


def authored_link_texts(*, message: Message) -> list[str]:
    """The text spans a message's author actually wrote, for scanning a message replied to.

    Narrower than `message_link_texts` by exactly one thing: an embed card never counts,
    neither the message's own nor a forwarded snapshot's. One hop out an embed is a card the
    author did not write, and the bot's own Threads expansion is the common one:
    `parse_threads._build_embed_plan` emits one permalink per post in the reply chain, ROOT first,
    so a scan keyed on it would read the thread's top post rather than the one the human
    linked — and it disappears entirely when an oversize video pushes hosted URLs into
    `content`. A link a person typed always lives in `content` (or in the content of what they
    forwarded), so nothing human-written is lost. The bot's own replies pass through here too,
    so every span gets the `get_cleaned_content` / `snapshot_text` usage-footer strip: the
    footer carries the memory labels, which are display names their owners choose.
    """
    spans = [message.content or "", *(snapshot.content for snapshot in message.snapshots)]
    return [USAGE_FOOTER_RE.sub("", span).strip() for span in spans]


def _first_url_match(pattern: re.Pattern[str], texts: list[str]) -> re.Match[str] | None:
    """First match of a URL pattern across one message's already-rendered text spans."""
    for text in texts:
        match = pattern.search(string=text)
        if match:
            return match
    return None


def link_url_for_source(*, source: LinkContextSource, message: Message) -> str | None:
    """The URL one link source should read: the current message's, else the replied-to one's.

    The current message always wins. A source that opts into `search_replied_to_message` then
    falls back to the message being replied to, the same one hop `find_youtube_url` takes, so
    "@bot 這篇底下在吵什麼" sent as a reply to someone else's link still reads the post; one
    that does not opt in never looks past the triggering message. That parent is scanned with
    `authored_link_texts`, which is what keeps the bot's own expansion from triggering a read
    of the wrong post.

    A source's `url_filter` rejects a matched link it cannot read (e.g. a Douyin profile or
    live room, whose regex matches the host, not the path), which would only spend a
    rate-limited request to say so. It applies to the chosen match alone: a rejected link
    drops the source rather than sending the scan hunting for a second URL.
    """
    match = _first_url_match(
        pattern=source.url_pattern,
        texts=message_link_texts(message=message, strip_usage_footer=False),
    )
    if match is None and source.search_replied_to_message:
        replied_to = replied_to_message(message=message)
        if replied_to is not None:
            match = _first_url_match(
                pattern=source.url_pattern, texts=authored_link_texts(message=replied_to)
            )
    if match is None:
        return None
    url = match.group(0)
    if source.url_filter is not None and not source.url_filter(url=url):
        return None
    return url


def _youtube_url_in_message(*, message: Message, strip_usage_footer: bool) -> str | None:
    """Returns the first YouTube URL in a message's text, embeds, or forwarded snapshots, if any."""
    match = _first_url_match(
        pattern=YOUTUBE_URL_RE,
        texts=message_link_texts(message=message, strip_usage_footer=strip_usage_footer),
    )
    return match.group(0) if match else None


def find_youtube_url(*, message: Message) -> str | None:
    """Finds a YouTube URL in the current message or the message it replies to.

    A reply to a message that merely links a video would otherwise be missed, so the parent is
    searched too and "summarize this" on a replied-to video still watches it. The current
    message wins. Threads reaches the same one hop (`link_url_for_source`,
    `search_replied_to_message`); Douyin and Bilibili deliberately do not, since their value is
    the clip rather than a discussion and both are rate-limit sensitive. This one keeps scanning
    embeds out there — a YouTube link card is the link itself, not a rendering of some other
    post the way a Threads expansion is.
    """
    found = _youtube_url_in_message(message=message, strip_usage_footer=False)
    if found is not None:
        return found
    replied_to = replied_to_message(message=message)
    if replied_to is not None:
        return _youtube_url_in_message(message=replied_to, strip_usage_footer=True)
    return None


def source_channel_is_public(*, message: Message) -> bool:
    """Whether @everyone can view the message's channel, so its content is not private.

    `message.channel` is a heterogeneous messageable union, so visibility is read
    defensively (mirrors `utils.discord_embeds`): a private thread is never public; a
    thread otherwise inherits its parent channel's `@everyone` visibility; a regular
    guild channel uses its own. A non-guild message, or any channel whose permissions
    cannot be resolved, counts as non-public — so content from channels members cannot
    see never enters the server-wide memory any member can read via `/memory server show`.
    """
    guild = message.guild
    if guild is None:
        return False
    channel = message.channel
    is_private = getattr(channel, "is_private", None)
    if callable(is_private) and is_private():
        return False
    source = getattr(channel, "parent", None) or channel
    permissions_for = getattr(source, "permissions_for", None)
    if not callable(permissions_for):
        return False
    return bool(getattr(permissions_for(guild.default_role), "view_channel", False))
