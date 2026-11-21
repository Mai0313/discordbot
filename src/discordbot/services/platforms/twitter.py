"""Twitter (x.com) URL parsing and reading via the public syndication API and FxTwitter.

`x.com` serves an empty SPA shell logged out, so posts are read from
`cdn.syndication.twimg.com/tweet-result`. Success is indicated by `__typename == "Tweet"`
(tombstones return HTTP 200).

FxTwitter's `/2/conversation/<id>` is layered on top for full ancestor chains, replies, and
untruncated note-tweet bodies (syndication truncates long posts around 275 characters and marks
`note_tweet` on the target only). Any FxTwitter failure degrades to the syndication read.
"""

import re
from typing import Any
from datetime import UTC, datetime
from functools import cached_property
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

import logfire
from pydantic import Field, BaseModel, ValidationError, computed_field
import requests

from discordbot.utils.urls import URL_START_ANCHOR
from discordbot.typings.timeouts import (
    TWITTER_PAGE_TIMEOUT_SECONDS,
    FXTWITTER_CONVERSATION_TIMEOUT_SECONDS,
)
from discordbot.utils.link_errors import link_fetch_error
from discordbot.services.platforms.base import (
    PlatformOutput,
    PlatformDownloader,
    PlatformConversation,
    thread_branches,
)

# Path-anchored to `/status/<digits>` (no post filter needed); handle is untrusted and rebuilt
# from the payload, and the ASCII query tail stops before CJK punctuation.
TWITTER_URL_RE = re.compile(
    pattern=rf"{URL_START_ANCHOR}https?://(?:www\.|mobile\.|m\.)?(?:x|twitter)\.com/"
    r"(?:i/web/status|[A-Za-z0-9_]{1,20}/status(?:es)?)/\d+"
    r"(?:/(?:photo|video)/\d+)?/?"
    r"(?:\?[A-Za-z0-9=&%_.-]*[A-Za-z0-9_-])?"
)

_STATUS_PATH_RE = re.compile(pattern=r"/status(?:es)?/(?P<id>\d+)")

# Signed 64-bit snowflake max; larger ids return HTTP 400 (unclassified) instead of 404.
_MAX_STATUS_ID = 2**63 - 1

_SYNDICATION_URL = "https://cdn.syndication.twimg.com/tweet-result"
_FXTWITTER_CONVERSATION_URL = "https://api.fxtwitter.com/2/conversation/{status_id}"
_CANONICAL_TWITTER_ORIGIN = "https://x.com"

# Syndication returns 400 without User-Agent and `{}` without a non-empty `token`.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
_REQUEST_TOKEN = "a"  # noqa: S105 — a parameter the endpoint requires and never reads, not a secret

# Bare `pbs.twimg.com` URLs serve medium size; `?name=orig` fetches full resolution.
_ORIGINAL_SIZE_QUERY = "?name=orig"

# Twitter's per-post media limit.
_MAX_MEDIA_ITEMS = 4


def _time_of(value: str) -> datetime | None:
    """Returns the ISO-8601 Zulu timestamp as a UTC `datetime`, or `None` when absent or malformed."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


class TwitterURL(BaseModel):
    """Parses and normalises a Twitter post URL."""

    raw_url: str = Field(..., description="Original Twitter URL provided by the caller")

    @computed_field
    @cached_property
    def status_id(self) -> str:
        """Returns the numeric status id from the path, or an empty string when absent."""
        match = _STATUS_PATH_RE.search(string=urlparse(self.raw_url).path)
        return match.group("id") if match else ""


def post_url(handle: str, status_id: str) -> str:
    """Returns the canonical URL for a post, built from the author the payload named.

    Args:
        handle (str): The author's screen name, without the leading `@`.
        status_id (str): The post's numeric id.
    """
    return f"{_CANONICAL_TWITTER_ORIGIN}/{handle or 'i'}/status/{status_id}"


class TwitterOutput(PlatformOutput):
    """One Twitter post or reply.

    `comment_count` holds `conversation_count` on the target post and `reply_count` on embedded
    posts. `retweet_count` is omitted; syndication populates it only on embedded posts.
    """

    video_poster_urls: list[str] = Field(
        default_factory=list,
        description="Still frames for any video, in media order; what the card shows, since nothing here downloads the clip",
    )
    # Bounded to one level of nesting.
    quoted: "TwitterOutput | None" = Field(
        default=None,
        description="The post this one quotes, absent when it quotes nothing readable",
    )
    is_truncated: bool = Field(
        default=False,
        description="Whether the body is cut and FxTwitter could not serve the rest; only the post asked for can report it, see the module docstring",
        examples=[False],
    )


class TwitterConversation(PlatformConversation[TwitterOutput]):
    """A parsed Twitter post, its ancestor chain, and the first page of replies.

    `chain` holds readable ancestors from FxTwitter oldest-first (falling back to syndication's
    single embedded parent on failure) ending with the target. `reply_branches` holds FxTwitter's
    first reply page when enabled. `selected_comment_id` is always empty.
    """

    @property
    def parent(self) -> TwitterOutput | None:
        """Returns the post the target replies to (`chain[-2]`), or `None` when it replies to nothing readable."""
        return self.chain[-2] if len(self.chain) > 1 else None


class _TwitterPayload(BaseModel):
    """Base for syndication mirrors; absent keys default, explicit nulls require nullable types."""


class _User(_TwitterPayload):
    """The author block, present on every live post and on every embedded one."""

    screen_name: str = Field(default="", description="Handle, without the leading @")
    profile_image_url_https: str = Field(default="", description="Avatar URL, 48px")


class _VideoVariant(_TwitterPayload):
    """One video/GIF rendition from `mediaDetails[].video_info.variants` (carries `bitrate`)."""

    bitrate: int = Field(default=0, description="Bits per second; absent on the HLS manifest")
    content_type: str = Field(default="", description="MIME type of this rendition")
    url: str = Field(default="", description="Direct URL for this rendition")


class _VideoInfo(_TwitterPayload):
    """The rendition list for a video or animated GIF."""

    variants: list[_VideoVariant] = Field(default_factory=list, description="Available renditions")


class _MediaDetail(_TwitterPayload):
    """One attachment from `mediaDetails` (preserves order across mixed photo/video posts)."""

    type: str = Field(default="", description="`photo`, `video` or `animated_gif`")
    media_url_https: str = Field(default="", description="Still image, or a video's poster frame")
    video_info: _VideoInfo | None = Field(default=None, description="Renditions, on a video only")


class _UrlEntity(_TwitterPayload):
    """One `t.co` link and what it stands for."""

    url: str = Field(default="", description="The t.co form, as it appears in the body")
    expanded_url: str = Field(default="", description="Where it actually points")


class _Entities(_TwitterPayload):
    """The link and media spans inside the body. May be entirely absent on an older post."""

    urls: list[_UrlEntity] = Field(default_factory=list, description="External links in the body")


class _Tweet(_TwitterPayload):
    """Syndication tweet payload (target, embedded parent, or quoted post)."""

    id_str: str = Field(default="", description="The post's numeric id")
    text: str = Field(default="", description="Body, with every URL in t.co form")
    created_at: str = Field(default="", description="ISO-8601 Zulu publication time")
    favorite_count: int = Field(default=0, description="Likes")
    conversation_count: int = Field(default=0, description="Size of the thread under the post")
    # Populated on embedded parent/quoted tweets; the target carries `conversation_count` instead.
    reply_count: int = Field(default=0, description="Direct replies; embedded posts only")
    display_text_range: list[int] = Field(
        default_factory=list, description="Body span in UTF-16 code units, excluding media links"
    )
    entities: _Entities | None = Field(default=None, description="Link and media spans")
    user: _User | None = Field(default=None, description="Author")
    mediaDetails: list[_MediaDetail] = Field(  # noqa: N815 — mirrors the payload's own camelCase key
        default_factory=list, description="Every attachment, in post order"
    )
    note_tweet: dict[str, Any] | None = Field(
        default=None, description="Present when the body is truncated; carries only an opaque id"
    )
    parent: "_Tweet | None" = Field(
        default=None, description="The post replied to, when it is readable"
    )
    quoted_tweet: "_Tweet | None" = Field(default=None, description="The post this one quotes")


class _TombstoneText(_TwitterPayload):
    """Inner `tombstone.text` object holding the localized refusal string."""

    text: str = Field(
        default="", description="Why the post cannot be shown, in the request's lang"
    )


class _TombstoneBody(_TwitterPayload):
    """The `tombstone` block; the wording under it is the only member read."""

    text: _TombstoneText | None = Field(default=None, description="The worded refusal")


class _Tombstone(_TwitterPayload):
    """Unavailable-post tombstone payload; `reason` is localized and for logging only."""

    tombstone: _TombstoneBody | None = Field(default=None, description="The refusal, when worded")

    @property
    def reason(self) -> str:
        """Returns the refusal wording, or an empty string when the payload carried none."""
        if self.tombstone is None or self.tombstone.text is None:
            return ""
        return self.tombstone.text.text


class _TweetResult(_TwitterPayload):
    """Syndication envelope distinguishing live `Tweet` responses from `TweetTombstone`."""

    typename: str = Field(
        default="", alias="__typename", description="`Tweet` or `TweetTombstone`"
    )


def _body_text(tweet: _Tweet) -> str:
    """Returns the post body with the trailing media link removed and `t.co` links expanded."""
    text = tweet.text
    if len(tweet.display_text_range) == 2:
        start, end = tweet.display_text_range
        # Syndication's `display_text_range` is indexed in UTF-16 code units, not codepoints.
        encoded = text.encode("utf-16-le")
        text = encoded[start * 2 : end * 2].decode("utf-16-le", errors="ignore")
    for entity in tweet.entities.urls if tweet.entities else []:
        if entity.url and entity.expanded_url:
            text = text.replace(entity.url, entity.expanded_url)
    return text.strip()


def _media_urls(tweet: _Tweet) -> tuple[list[str], list[str], list[str]]:
    """Returns `(image_urls, video_urls, video_poster_urls)` from `mediaDetails` in order."""
    images: list[str] = []
    videos: list[str] = []
    posters: list[str] = []
    for media in tweet.mediaDetails[:_MAX_MEDIA_ITEMS]:
        if media.type == "photo":
            if media.media_url_https:
                images.append(f"{media.media_url_https}{_ORIGINAL_SIZE_QUERY}")
            continue
        # Skip HLS manifests; animated GIFs carry one MP4 rendition at bitrate 0.
        renditions = [
            variant
            for variant in (media.video_info.variants if media.video_info else [])
            if variant.content_type == "video/mp4" and variant.url
        ]
        if not renditions:
            continue
        videos.append(max(renditions, key=lambda variant: variant.bitrate).url)
        posters.append(media.media_url_https)
    return images, videos, posters


def _build_output(tweet: _Tweet, include_quoted: bool = True) -> TwitterOutput:
    """Builds a `TwitterOutput` from `tweet`, bounding quoted recursion to one level."""
    images, videos, posters = _media_urls(tweet=tweet)
    author = tweet.user or _User()
    quoted = (
        _build_output(tweet=tweet.quoted_tweet, include_quoted=False)
        if include_quoted and tweet.quoted_tweet
        else None
    )
    return TwitterOutput(
        text=_body_text(tweet=tweet),
        url=post_url(handle=author.screen_name, status_id=tweet.id_str),
        author_name=author.screen_name,
        author_icon_url=author.profile_image_url_https,
        image_urls=images,
        video_urls=videos,
        video_poster_urls=posters,
        like_count=tweet.favorite_count,
        comment_count=tweet.conversation_count or tweet.reply_count,
        taken_at=_time_of(value=tweet.created_at),
        quoted=quoted,
        is_truncated=tweet.note_tweet is not None,
    )


class _FxPayload(BaseModel):
    """Base model for FxTwitter payload mirrors."""


class _FxAuthor(_FxPayload):
    """The author block."""

    screen_name: str = Field(default="", description="Handle, without the leading @")
    avatar_url: str | None = Field(default=None, description="Avatar URL; nullable in the spec")


class _FxMedia(_FxPayload):
    """One attachment."""

    type: str = Field(default="", description="`photo`, `video` or `gif`")
    url: str = Field(default="", description="A still at original size, or a clip's best mp4")
    thumbnail_url: str = Field(default="", description="A clip's poster frame")


class _FxMediaBlock(_FxPayload):
    """The media block; `{}` on a post with none."""

    all: list[_FxMedia] = Field(
        default_factory=list, description="Every attachment, in post order"
    )


class _FxRawText(_FxPayload):
    """The body before FxTwitter's own rendering, which is where the display range lives."""

    display_text_range: list[int] = Field(
        default_factory=list, description="Body span in codepoints, past the leading mentions"
    )


class _FxReplyingTo(_FxPayload):
    """The post this one replies to; the whole block is null on a post that replies to nothing."""

    status: str = Field(default="", description="That post's id")


class _FxStatus(_FxPayload):
    """One post, in `thread` or `replies`."""

    type: str = Field(
        default="", description="`status`; the spec also allows `tombstone` and `substatus`"
    )
    id: str = Field(default="", description="The post's numeric id")
    text: str = Field(
        default="", description="Body, links expanded; leading mentions kept unless it is long"
    )
    raw_text: _FxRawText | None = Field(default=None, description="Carries the display range")
    author: _FxAuthor | None = Field(default=None, description="Author")
    likes: int = Field(default=0, description="Likes")
    replies: int = Field(default=0, description="Direct replies")
    created_timestamp: int | None = Field(
        default=None, description="Publication time, Unix seconds"
    )
    media: _FxMediaBlock | None = Field(default=None, description="Attachments")
    replying_to: _FxReplyingTo | None = Field(default=None, description="The post replied to")
    is_note_tweet: bool = Field(default=False, description="Whether this is a long post, whole")
    quote: dict[str, Any] | None = Field(
        default=None, description="The post this one quotes, raw so a malformed one costs itself"
    )


class _FxThreadPage(_FxPayload):
    """FxTwitter `/2/conversation` envelope; posts stay raw for per-item validation."""

    code: int = Field(default=0, description="FxTwitter's own status, 200 on success")
    thread: list[dict[str, Any]] | None = Field(
        default=None, description="Every ancestor plus the post itself, in no reliable order"
    )
    replies: list[dict[str, Any]] | None = Field(
        default=None, description="The first page of direct replies, most-liked first"
    )


def _fx_statuses(items: list[dict[str, Any]] | None) -> list[_FxStatus]:
    """Returns live `"status"` items that validate, dropping malformed items and tombstones."""
    statuses: list[_FxStatus] = []
    for item in items or []:
        try:
            status = _FxStatus.model_validate(item)
        except ValidationError as error:
            logfire.warn(
                "Skipping an FxTwitter post that does not validate",
                status_id=item.get("id"),
                _exc_info=error,
            )
            continue
        if status.type == "status":
            statuses.append(status)
    return statuses


def _fx_text(status: _FxStatus) -> str:
    """Returns one FxTwitter post's body, past its leading reply mentions."""
    # Only use range start (codepoints): the end indexes `raw_text` with unexpanded `t.co` URLs.
    start = (
        status.raw_text.display_text_range[0]
        if status.raw_text and status.raw_text.display_text_range
        else 0
    )
    return status.text[start:].strip()


def _fx_output(status: _FxStatus) -> TwitterOutput:
    """Turns one FxTwitter post into the shared output shape."""
    author = status.author or _FxAuthor()
    images: list[str] = []
    videos: list[str] = []
    posters: list[str] = []
    for media in (status.media.all if status.media else [])[:_MAX_MEDIA_ITEMS]:
        if not media.url:
            continue
        if media.type == "photo":
            images.append(media.url)
            continue
        videos.append(media.url)
        posters.append(media.thumbnail_url)
    return TwitterOutput(
        text=_fx_text(status=status),
        url=post_url(handle=author.screen_name, status_id=status.id),
        author_name=author.screen_name,
        author_icon_url=author.avatar_url or "",
        image_urls=images,
        video_urls=videos,
        video_poster_urls=posters,
        like_count=status.likes,
        comment_count=status.replies,
        taken_at=(
            datetime.fromtimestamp(status.created_timestamp, tz=UTC)
            if status.created_timestamp
            else None
        ),
    )


def _fx_ancestors(thread: list[_FxStatus], status_id: str) -> list[TwitterOutput]:
    """Returns the posts above `status_id` in `thread`, oldest first, up to the first one missing."""
    by_id = {status.id: status for status in thread}
    ancestors: list[TwitterOutput] = []
    seen = {status_id}
    current = by_id.get(status_id)
    while current is not None and current.replying_to and current.replying_to.status not in seen:
        seen.add(current.replying_to.status)
        current = by_id.get(current.replying_to.status)
        if current is not None:
            ancestors.append(_fx_output(status=current))
    return ancestors[::-1]


def _with_whole_text(post: TwitterOutput, whole: _FxStatus | None) -> TwitterOutput:
    """Replaces truncated target and quoted bodies with FxTwitter's full `is_note_tweet` text."""
    if whole is None:
        return post
    update: dict[str, Any] = {}
    if whole.is_note_tweet:
        update.update(text=_fx_text(status=whole), is_truncated=False)
    quotes = _fx_statuses(items=[whole.quote]) if whole.quote else []
    if post.quoted is not None and quotes and quotes[0].is_note_tweet:
        update["quoted"] = post.quoted.model_copy(
            update={"text": _fx_text(status=quotes[0]), "is_truncated": False}
        )
    return post.model_copy(update=update)


class TwitterDownloader(PlatformDownloader):
    """Reads a Twitter post via syndication and FxTwitter, writing nothing to disk."""

    read_replies: bool = Field(
        default=True,
        description="Whether the replies are read; without them FxTwitter is asked only for a reply, a quote or a long post",
    )

    def _fetch_tweet(self, status_id: str) -> dict[str, Any]:
        """Returns the syndication JSON payload for `status_id`."""
        try:
            response = requests.get(
                _SYNDICATION_URL,
                params={"id": status_id, "lang": "en", "token": _REQUEST_TOKEN},
                headers=_REQUEST_HEADERS,
                timeout=TWITTER_PAGE_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise link_fetch_error(
                error=error, url=f"{_SYNDICATION_URL}?id={status_id}"
            ) from error
        # Decode outside `RequestException`: `requests.JSONDecodeError` subclasses it.
        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError(f"Twitter served a non-JSON body for post {status_id}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"Twitter served an unexpected body for post {status_id}")
        return payload

    def _fetch_conversation(self, status_id: str) -> dict[str, Any]:
        """Returns FxTwitter's `/2/conversation` JSON payload for `status_id`."""
        url = _FXTWITTER_CONVERSATION_URL.format(status_id=status_id)
        try:
            response = requests.get(
                url, headers=_REQUEST_HEADERS, timeout=FXTWITTER_CONVERSATION_TIMEOUT_SECONDS
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise link_fetch_error(error=error, url=url) from error
        # Decode outside `RequestException`: `requests.JSONDecodeError` subclasses it.
        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError(f"FxTwitter served a non-JSON body for post {status_id}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"FxTwitter served an unexpected body for post {status_id}")
        return payload

    def _read_conversation(
        self, status_id: str
    ) -> tuple[list[_FxStatus], list[list[TwitterOutput]]]:
        """Returns `(thread, reply_branches)` from FxTwitter, or empty lists on any failure."""
        # Enforce a whole-request deadline; `requests` timeout only bounds per-socket reads.
        pool = ThreadPoolExecutor(max_workers=1)
        fetch = pool.submit(self._fetch_conversation, status_id=status_id)
        pool.shutdown(wait=False)
        # Broad on purpose: this read only adds to a post syndication already served, so nothing
        # it raises may cost that post.
        try:
            conversation = _FxThreadPage.model_validate(
                fetch.result(timeout=FXTWITTER_CONVERSATION_TIMEOUT_SECONDS)
            )
            if conversation.code != 200:
                raise RuntimeError(f"FxTwitter answered code {conversation.code}")
        except Exception as error:
            logfire.warn(
                "FxTwitter conversation unreadable; keeping the syndication read",
                status_id=status_id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            return [], []
        replies = {status.id: status for status in _fx_statuses(conversation.replies)}
        return (
            _fx_statuses(conversation.thread),
            thread_branches(
                comments={
                    reply_id: _fx_output(status=reply) for reply_id, reply in replies.items()
                },
                parents={
                    reply_id: reply.replying_to.status
                    for reply_id, reply in replies.items()
                    if reply.replying_to
                },
            ),
        )

    def parse_metadata(self, url: str) -> TwitterConversation:
        """Returns the parsed conversation for a Twitter post URL, writing nothing to disk.

        Returns an empty `TwitterConversation` for invalid ids, out-of-range snowflakes, or
        tombstones. Queries FxTwitter after syndication when `read_replies` is true or when the
        post is a reply, quote, or truncated note tweet.

        Args:
            url (str): The post URL, in any of the forms `TWITTER_URL_RE` matches.

        Raises:
            LinkReadError: If the syndication request fails with a classified HTTP/network error.
            RuntimeError: If the endpoint serves a non-JSON or non-object body.
        """
        status_id = TwitterURL(raw_url=url).status_id
        if not status_id or int(status_id) > _MAX_STATUS_ID:
            return TwitterConversation()

        payload = self._fetch_tweet(status_id=status_id)
        if _TweetResult.model_validate(payload).typename != "Tweet":
            # Tombstones return HTTP 200; log the localized reason without matching on it.
            logfire.info(
                "Twitter post is not readable",
                status_id=status_id,
                reason=_Tombstone.model_validate(payload).reason,
            )
            return TwitterConversation()

        tweet = _Tweet.model_validate(payload)
        post = _build_output(tweet=tweet)
        # Skip FxTwitter when replies are off and syndication already has the full standalone post.
        ancestors: list[TwitterOutput] = []
        reply_branches: list[list[TwitterOutput]] = []
        if self.read_replies or tweet.parent or tweet.quoted_tweet or post.is_truncated:
            thread, branches = self._read_conversation(status_id=status_id)
            ancestors = _fx_ancestors(thread=thread, status_id=status_id)
            reply_branches = branches if self.read_replies else []
            post = _with_whole_text(
                post=post, whole=next((item for item in thread if item.id == status_id), None)
            )
        if not ancestors and tweet.parent:
            ancestors = [_build_output(tweet=tweet.parent)]
        return TwitterConversation(chain=[*ancestors, post], reply_branches=reply_branches)


if __name__ == "__main__":
    # Kept for local manual smoke-testing; do not remove.
    from rich.console import Console

    console = Console()
    downloader = TwitterDownloader()
    conversation = downloader.parse_metadata(
        url="https://x.com/thsottiaux/status/2090887457915232269"
    )
    console.print(conversation)
