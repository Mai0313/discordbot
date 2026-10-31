"""Twitter (x.com) post URL parsing and reading, through the public syndication endpoint and FxTwitter.

**The page itself is unreadable, so this module does not fetch it.** Measured 2026-09-10:
`x.com/<user>/status/<id>` answers a pure SPA shell with no post text and no Open Graph tags at
all, for a browser User-Agent and for `facebookexternalhit` / `Twitterbot` / `Discordbot` alike.
Scraping the HTML for an embedded payload finds nothing here.

What does work logged out is `cdn.syndication.twimg.com/tweet-result`, the endpoint the official
embed widget calls. It needs an `id`, a `token` of any non-empty value, and any `User-Agent`
header; `lang` is optional and only picks the locale a refusal is worded in (measured: dropping
it returns a byte-identical body). Both of the required ones are traps rather than ceremony —
omitting the token answers HTTP 200 with a body of `{}`, and omitting the User-Agent answers 400
with an empty body — and neither failure says what is wrong.

**Success is `__typename`, never the status code.** A deleted, protected or suspended post comes
back as HTTP 200 carrying a well-formed `TweetTombstone`, while an id that never existed answers
404 with HTML. Reading either off the status gets it backwards. The refusal's own text is localized
by `lang`, so `_Tombstone` reads it for the log and nothing ever matches on its wording.

Two things syndication cannot do, so a second request goes to FxTwitter for them:

Replies are unreachable there. The payload carries `conversation_count` and no reply content, and
13 parameter variants (`replies`, `conversation`, `cursor`, `expansions`, ...) returned
byte-identical responses.

The post being replied to arrives embedded exactly one level deep. Where the payload's own
`in_reply_to_status_id_str` is set and `parent` is absent, the parent is unreadable — 10 of 10
sampled were deleted, protected or suspended.

**FxTwitter's `/2/conversation/<id>` fills both, and only ever adds to the syndication read.**
Measured 2026-10-31: no key, documented (docs.fxembed.com, with an OpenAPI spec), ~1.5s and ~80KB.
`thread` is every ancestor plus the post itself, in no reliable order (one sample came back
parent, root, post), so the chain is walked by `replying_to` ids rather than read in order;
`replies` is the first page of direct replies, most-liked first, an author's own continuation
among them. It is a third party reading X with accounts of its own, so it can stop working
without notice: any failure there is logged and the conversation keeps what syndication served —
the post, its quote and its one embedded parent — with no replies. Its `display_text_range` counts
codepoints, not the UTF-16 units syndication's does (`😬` is `[25, 26]`), and only its start is
read, to drop the leading reply mentions.

One more thing syndication cannot do, which the callers have to state rather than hide:

A long post is truncated at roughly 275 characters, with the remainder behind an opaque
`note_tweet.id` the endpoint will not resolve. Nothing in the text marks the cut, which is why
`TwitterOutput.is_truncated` exists: a caller that cannot see the cut presents a fragment as the
whole post. **That flag is only meaningful on the post asked for.** `note_tweet` is served in the
target position alone — an embedded parent or quote carries the identically cut `text`,
`display_text_range` and `entities` with the key simply absent (measured on both copies of post
1724892505647296620) — so an embedded post reports `is_truncated=False` whether or not it was
cut. Deriving it from the body's length instead is not sound: the ceiling sits under the classic
280-character limit, so an ordinary whole post reaches it too.
"""

import re
from typing import Any
from datetime import UTC, datetime
from functools import cached_property
from urllib.parse import urlparse

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

# Path-anchored rather than host-anchored, so no post filter is needed on top: `/status/<digits>`
# names a post and nothing else on the site does, so a profile, the home page or a list never
# matches in the first place. Both hosts are matched because `twitter.com` still 301s to `x.com`
# rather than being retired, and the mobile hosts are what a phone's share sheet copies.
# `/i/web/status/<id>` is the form a link with no handle takes.
#
# The username segment is matched but never trusted: x.com serves the same post under any handle,
# including one that does not exist (measured — it 307s to the real author's URL), so only the id
# is load-bearing and every URL this module republishes is built by `post_url` from the author the
# payload names. That also drops the `?s=46&t=<token>` tail, which is minted per share and names
# whoever passed the link on.
#
# The optional `/photo/1` or `/video/1` suffix is what x.com's own image lightbox copies. The query
# tail is matched so `?s=46&t=<token>` is consumed rather than left dangling, and it ends on
# `[A-Za-z0-9_-]` for the reason every pattern here does: a link written straight after Chinese or
# Japanese text stops at the terminator instead of swallowing it.
TWITTER_URL_RE = re.compile(
    pattern=rf"{URL_START_ANCHOR}https?://(?:www\.|mobile\.|m\.)?(?:x|twitter)\.com/"
    r"(?:i/web/status|[A-Za-z0-9_]{1,20}/status(?:es)?)/\d+"
    r"(?:/(?:photo|video)/\d+)?/?"
    r"(?:\?[A-Za-z0-9=&%_.-]*[A-Za-z0-9_-])?"
)

_STATUS_PATH_RE = re.compile(pattern=r"/status(?:es)?/(?P<id>\d+)")

# A status id is a signed 64-bit snowflake, and one past that range is the only input the endpoint
# answers 400 to (measured: 2**63 - 1 and every smaller nonsense id answer 404, which classifies as
# unavailable and earns the ⚠️ mark; a 400 does not classify at all and would reach the reader as
# the cross that means the bot broke). The pattern cannot carry the bound — a 19-digit prefix of a
# 21-digit id matches just as well — so the check lives where the id is read.
_MAX_STATUS_ID = 2**63 - 1

_SYNDICATION_URL = "https://cdn.syndication.twimg.com/tweet-result"
_FXTWITTER_CONVERSATION_URL = "https://api.fxtwitter.com/2/conversation/{status_id}"
_CANONICAL_TWITTER_ORIGIN = "https://x.com"

# The endpoint rejects a request with no User-Agent (HTTP 400, empty body) and answers `{}` to one
# with no token. The token's value is never checked — every non-empty string tested returned the
# same 920-byte payload — so this is a required-but-unvalidated parameter rather than a credential,
# and nothing here is authenticating as anyone.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
_REQUEST_TOKEN = "a"  # noqa: S105 — a parameter the endpoint requires and never reads, not a secret

# Twitter's own image CDN serves `medium` from a bare URL while the payload reports the original's
# dimensions beside it, so a caller that trusts the JSON and takes the bare URL gets a 900x1200
# image labelled 3024x4032. Measured: bare 57,759 bytes / 900x1200, `?name=orig` 978,797 bytes /
# 3024x4032. Where the original is smaller than the `large` bound the two are byte-identical, so
# this is never a downgrade.
_ORIGINAL_SIZE_QUERY = "?name=orig"

# Twitter's own per-post media limit. The reader keeps at most this many items, so every later
# cap on a Twitter post's media sits at or under it.
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
        """Returns the numeric post id the URL names, or an empty string when it names none.

        Read off the path rather than the last segment, so a `/photo/1` suffix does not become the
        id. This is the whole of what identifies a post: the handle in front of it is decorative.
        """
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
    """One post, the single shape a conversation is built from.

    The shared fields and `is_readable` come from `PlatformOutput`; what is declared here is
    Twitter's own reality. One inherited field means something narrower: `comment_count` is the
    size of the thread under the post rather than its direct replies, that being the only reply
    figure the endpoint reports for the post asked for.

    And there is deliberately no `retweet_count`. The endpoint publishes one for an embedded parent
    or quoted post and never for the post actually asked for, and a field a platform cannot fill is
    worse than an absent one.
    """

    video_poster_urls: list[str] = Field(
        default_factory=list,
        description="Still frames for any video, in media order; what the card shows, since nothing here downloads the clip",
    )
    # One level only, and never populated on a quoted post itself: 46 sampled quotes carried no
    # nested quote. `_build_output` bounds it anyway rather than trusting the sample.
    quoted: "TwitterOutput | None" = Field(
        default=None,
        description="The post this one quotes, absent when it quotes nothing readable",
    )
    is_truncated: bool = Field(
        default=False,
        description="Whether Twitter cut the body and will not serve the rest; only the post asked for can report it, see the module docstring",
        examples=[False],
    )


class TwitterConversation(PlatformConversation[TwitterOutput]):
    """A parsed Twitter post, the thread above it, and the first page of replies under it.

    The shared fields and accessors come from `PlatformConversation`. Three of them carry a
    Twitter-specific truth.

    `chain` is every readable ancestor FxTwitter served, oldest first, then the target; when that
    read failed it falls back to syndication's one embedded parent, so a chain of two does not
    mean the thread starts there.

    `reply_branches` holds one branch per direct reply on FxTwitter's first page, most-liked first,
    each followed by whatever replies to it that page carried, and nothing when the read failed.

    `selected_comment_id` is always empty: a reply has a post URL of its own, so a link to one
    makes it the target.
    """

    @property
    def parent(self) -> TwitterOutput | None:
        """Returns the post the target replies to (`chain[-2]`), or `None` when it replies to nothing readable."""
        return self.chain[-2] if len(self.chain) > 1 else None


class _TwitterPayload(BaseModel):
    """Base for the syndication payload mirrors.

    Every field below is optional with a default, because the payload omits rather than nulls most
    of what a given post lacks — `photos` is present-but-empty on a video post, `mediaDetails` is
    absent entirely on a text post, and `parent` only appears on a reply. A null is not absorbed:
    only a field typed to accept None takes one, and anywhere else it fails validation.
    """


class _User(_TwitterPayload):
    """The author block, present on every live post and on every embedded one."""

    screen_name: str = Field(default="", description="Handle, without the leading @")
    profile_image_url_https: str = Field(default="", description="Avatar URL, 48px")


class _VideoVariant(_TwitterPayload):
    """One rendition of a video.

    Read off `mediaDetails[].video_info.variants` rather than the top-level `video.variants`: the
    two describe the same renditions under different key names (`content_type`/`url` here,
    `type`/`src` there) and only this one carries `bitrate`, which is the whole basis for picking
    a rendition.

    A video leads with the HLS manifest, which has no bitrate and is not a file; an animated GIF
    carries no manifest at all and exactly one `video/mp4` at `bitrate: 0`. Filtering on the
    content type rather than skipping the first entry is what makes one reader serve both.
    """

    bitrate: int = Field(default=0, description="Bits per second; absent on the HLS manifest")
    content_type: str = Field(default="", description="MIME type of this rendition")
    url: str = Field(default="", description="Direct URL for this rendition")


class _VideoInfo(_TwitterPayload):
    """The rendition list for a video or animated GIF."""

    variants: list[_VideoVariant] = Field(default_factory=list, description="Available renditions")


class _MediaDetail(_TwitterPayload):
    """One attachment.

    `mediaDetails` is the only authoritative media list. The sibling `photos` and `video` keys are
    lossy convenience views: `photos` silently drops videos from a mixed post and loses their
    position, and `video` holds at most one.

    Mixed media is rare enough to miss: a 417-post sample contained none, and post
    1885011383672000609 is one — `mediaDetails` reads photo/video/photo/photo while `photos`
    returns three entries with the clip's place in the order gone. Both `photos` and `mediaDetails`
    are absent entirely on a post with no media, and `photos` is present-but-empty on a video, so
    every media field here defaults and none is required.
    """

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
    """One post, as the endpoint serves it — the target, an embedded parent, or a quoted post.

    One model serves all three positions, but the payload is not identical across them and the
    differences are load-bearing rather than trivia. An embedded parent or quote carries no
    `__typename`, no `conversation_count` and no `note_tweet`, and carries `reply_count` and
    `retweet_count` that the target never does. `reply_count` is mirrored and read, for the reason
    the comment below it gives; `retweet_count` deliberately is not (`TwitterOutput` owns why); and
    the missing `note_tweet` is why `is_truncated` cannot be trusted on an embedded post, which the
    module docstring owns.
    """

    id_str: str = Field(default="", description="The post's numeric id")
    text: str = Field(default="", description="Body, with every URL in t.co form")
    created_at: str = Field(default="", description="ISO-8601 Zulu publication time")
    favorite_count: int = Field(default=0, description="Likes")
    conversation_count: int = Field(default=0, description="Size of the thread under the post")
    # The endpoint reports the reply figure under two names depending on the post's position: the
    # post asked for carries `conversation_count` and never `reply_count`, while an embedded parent
    # or quoted post carries `reply_count` and never `conversation_count`. Reading only the first
    # renders every quoted post as having no replies.
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
    """The innermost block, where the wording finally sits.

    One level deeper than the key name suggests: `tombstone.text` is an object carrying `entities`,
    `rtl` and a `text` of its own, not the string. Mirroring it one level short fails validation on
    every real refusal while the rare wordless one passes, since there is then nothing to validate.
    """

    text: str = Field(
        default="", description="Why the post cannot be shown, in the request's lang"
    )


class _TombstoneBody(_TwitterPayload):
    """The `tombstone` block; the wording under it is the only member read."""

    text: _TombstoneText | None = Field(default=None, description="The worded refusal")


class _Tombstone(_TwitterPayload):
    """A post the platform will not serve: deleted, protected, suspended, or from a gone account.

    The wording is the only thing separating those four, and 4 of 133 sampled refusals carried an
    empty `tombstone` with no wording at all — so it is logged for an operator and never used to
    decide anything.
    """

    tombstone: _TombstoneBody | None = Field(default=None, description="The refusal, when worded")

    @property
    def reason(self) -> str:
        """Returns the refusal wording, or an empty string when the payload carried none."""
        if self.tombstone is None or self.tombstone.text is None:
            return ""
        return self.tombstone.text.text


class _TweetResult(_TwitterPayload):
    """The response envelope.

    `__typename` is the only readable success signal: a tombstone and a live post are both HTTP
    200, and the tombstone's text is localized so matching it would work in English and nowhere
    else.
    """

    typename: str = Field(
        default="", alias="__typename", description="`Tweet` or `TweetTombstone`"
    )


def _body_text(tweet: _Tweet) -> str:
    """Returns the post body with the trailing media link removed and external `t.co` links expanded.

    Two index bases live in one payload and mixing them corrupts any post containing an emoji:
    `display_text_range` is UTF-16 code units (measured: 31 of 31 discriminating samples), while
    `entities.*.indices` are codepoints (36 of 36). Python strings are indexed by codepoint, so
    slicing by `display_text_range` directly leaks the first characters of the trailing `t.co`
    into the body; encoding to UTF-16 first aligns the range before expanding `entities.urls`.
    """
    text = tweet.text
    if len(tweet.display_text_range) == 2:
        start, end = tweet.display_text_range
        encoded = text.encode("utf-16-le")
        text = encoded[start * 2 : end * 2].decode("utf-16-le", errors="ignore")
    for entity in tweet.entities.urls if tweet.entities else []:
        if entity.url and entity.expanded_url:
            text = text.replace(entity.url, entity.expanded_url)
    return text.strip()


def _media_urls(tweet: _Tweet) -> tuple[list[str], list[str], list[str]]:
    """Returns `(image_urls, video_urls, video_poster_urls)` from `tweet.mediaDetails` in post order.

    Walks `mediaDetails` rather than `photos` / `video`, which lose a mixed post's order; animated
    GIFs are treated as videos (served as MP4 with a single bitrate-0 rendition).
    """
    images: list[str] = []
    videos: list[str] = []
    posters: list[str] = []
    for media in tweet.mediaDetails[:_MAX_MEDIA_ITEMS]:
        if media.type == "photo":
            if media.media_url_https:
                images.append(f"{media.media_url_https}{_ORIGINAL_SIZE_QUERY}")
            continue
        renditions = [
            variant
            for variant in (media.video_info.variants if media.video_info else [])
            if variant.content_type == "video/mp4" and variant.url
        ]
        if not renditions:
            continue
        videos.append(max(renditions, key=lambda variant: variant.bitrate).url)
        if media.media_url_https:
            posters.append(media.media_url_https)
    return images, videos, posters


def _build_output(tweet: _Tweet, include_quoted: bool = True) -> TwitterOutput:
    """Turns one payload post into the shared output shape.

    `include_quoted` bounds the recursion at one level. The sample says a quoted post never carries
    its own quote (0 of 46), so this bounds what was already true rather than truncating anything —
    but the payload is not ours and a shape we do not control is not a shape to trust.
    """
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
    """Base for the FxTwitter payload mirrors; every field defaults, for the reason `_TwitterPayload` gives."""


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
    text: str = Field(default="", description="Body, links expanded, leading mentions kept")
    raw_text: _FxRawText | None = Field(default=None, description="Carries the display range")
    author: _FxAuthor | None = Field(default=None, description="Author")
    likes: int = Field(default=0, description="Likes")
    replies: int = Field(default=0, description="Direct replies")
    created_timestamp: int | None = Field(
        default=None, description="Publication time, Unix seconds"
    )
    media: _FxMediaBlock | None = Field(default=None, description="Attachments")
    replying_to: _FxReplyingTo | None = Field(default=None, description="The post replied to")


class _FxThreadPage(_FxPayload):
    """The `/2/conversation` envelope.

    Its posts stay raw here and are validated one at a time by `_fx_statuses`, so one malformed
    reply costs itself rather than the whole thread.
    """

    code: int = Field(default=0, description="FxTwitter's own status, 200 on success")
    thread: list[dict[str, Any]] | None = Field(
        default=None, description="Every ancestor plus the post itself, in no reliable order"
    )
    replies: list[dict[str, Any]] | None = Field(
        default=None, description="The first page of direct replies, most-liked first"
    )


def _fx_statuses(items: list[dict[str, Any]] | None) -> list[_FxStatus]:
    """Returns the items that validate as live posts, dropping the rest.

    A tombstone is dropped rather than kept as an empty post, so the ancestor walk stops at a
    deleted post instead of showing a blank one.
    """
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


def _fx_output(status: _FxStatus) -> TwitterOutput:
    """Turns one FxTwitter post into the shared output shape.

    Only the range's start is read: it sits past the leading mentions, which are identical in
    `text` and `raw_text.text`, while the end counts `raw_text.text`'s `t.co` links and would cut
    the expanded `text` short.
    """
    author = status.author or _FxAuthor()
    start = (
        status.raw_text.display_text_range[0]
        if status.raw_text and status.raw_text.display_text_range
        else 0
    )
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
        if media.thumbnail_url:
            posters.append(media.thumbnail_url)
    return TwitterOutput(
        text=status.text[start:].strip(),
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


class TwitterDownloader(PlatformDownloader):
    """Reads a Twitter post through the public syndication endpoint.

    Holds no state and writes nothing to disk, so one instance serves every caller. There is
    deliberately no `parse`: nothing here downloads media, and every image, poster and clip comes
    back as a URL.
    """

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
        # Decoded outside the clause above rather than inside it: `requests.JSONDecodeError`
        # subclasses `RequestException`, so a body that is not JSON would otherwise be classified
        # as a failed fetch and reported as one, of a request that landed fine.
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
        # Outside the clause above for the reason `_fetch_tweet` gives.
        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError(f"FxTwitter served a non-JSON body for post {status_id}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"FxTwitter served an unexpected body for post {status_id}")
        return payload

    def _read_conversation(
        self, status_id: str
    ) -> tuple[list[TwitterOutput], list[list[TwitterOutput]]]:
        """Returns `(ancestors, reply_branches)` from FxTwitter, both empty when it could not be read.

        `replies` is not only direct replies: a reply to one of them follows the reply it answers
        (6 of 40 on one sample, three deep), so the branches are grouped by `replying_to`.
        """
        # Broad on purpose: this read only adds to a post syndication already served, so nothing
        # it raises may cost that post.
        try:
            conversation = _FxThreadPage.model_validate(
                self._fetch_conversation(status_id=status_id)
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
            _fx_ancestors(thread=_fx_statuses(conversation.thread), status_id=status_id),
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
        """Parses a Twitter post URL into the conversation, writing nothing to disk.

        The chain is the readable ancestors oldest first, then the post, and empty when the post
        could not be read at all — a deleted, protected or suspended post, or an id that never
        existed or could not be one. An empty conversation is the ordinary outcome for a pasted
        link rather than an error, so it comes back rather than raising, and the caller marks it
        unreadable. FxTwitter is asked only once syndication has served the post.

        Args:
            url (str): The post URL, in any of the forms `TWITTER_URL_RE` matches.

        Returns:
            The parsed conversation; empty when there is no readable post.

        Raises:
            LinkReadError: If the request itself failed in a way HTTP named.
            RuntimeError: If the endpoint served something other than a JSON object.
        """
        status_id = TwitterURL(raw_url=url).status_id
        if not status_id or int(status_id) > _MAX_STATUS_ID:
            return TwitterConversation()

        payload = self._fetch_tweet(status_id=status_id)
        if _TweetResult.model_validate(payload).typename != "Tweet":
            # A tombstone is a 200 with a well-formed body, so this is the only place the
            # difference is visible. Its own text says which of deleted / protected / suspended it
            # was, and is localized by `lang`, so it is logged rather than matched on.
            logfire.info(
                "Twitter post is not readable",
                status_id=status_id,
                reason=_Tombstone.model_validate(payload).reason,
            )
            return TwitterConversation()

        tweet = _Tweet.model_validate(payload)
        ancestors, reply_branches = self._read_conversation(status_id=status_id)
        if not ancestors and tweet.parent:
            ancestors = [_build_output(tweet=tweet.parent)]
        return TwitterConversation(
            chain=[*ancestors, _build_output(tweet=tweet)], reply_branches=reply_branches
        )


if __name__ == "__main__":
    """
    Keep this for self development and testing.
    DO NOT REMOVE THIS FOR ANY REASON.
    """
    from rich.console import Console

    console = Console()
    downloader = TwitterDownloader()
    conversation = downloader.parse_metadata(
        url="https://x.com/thsottiaux/status/2090887457915232269"
    )
    console.print(conversation)
