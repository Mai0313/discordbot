"""Twitter (x.com) post URL parsing and reading, through the public syndication endpoint.

Shared by `parse_twitter` (which expands a pasted link into embeds) and `gen_reply` (which reads
the post into answer context), the same split `facebook.py` and `instagram.py` serve.

**The page itself is unreadable, so this module does not fetch it.** Measured 2026-09-10:
`x.com/<user>/status/<id>` answers a pure SPA shell with no post text and no Open Graph tags at
all, for a browser User-Agent and for `facebookexternalhit` / `Twitterbot` / `Discordbot` alike.
The scrape-the-HTML approach the other two take has nothing to scrape here.

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

Three things this source cannot do, each of which the callers have to state rather than hide:

Replies are unreachable. The payload carries `conversation_count` and no reply content, and 13
parameter variants (`replies`, `conversation`, `cursor`, `expansions`, ...) returned byte-identical
responses. So `TwitterConversation.reply_branches` is permanently empty — the field is carried only
so the surface means the same as the other three sources.

The post being replied to arrives embedded, exactly one level deep, and nothing deeper. Walking
further is one request per hop, which this module deliberately does not spend: `chain` is
`[parent, target]` or `[target]`. Where the payload's own `in_reply_to_status_id_str` is set and
`parent` is absent, the parent is unreadable — 10 of 10 sampled were deleted, protected or
suspended — so neither key is mirrored: a chain of one is the whole answer either way.

A long post is TRUNCATED at roughly 275 characters, with the remainder behind an opaque
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
from pydantic import Field, BaseModel, computed_field
import requests

from discordbot.utils.urls import URL_START_ANCHOR
from discordbot.typings.timeouts import TWITTER_PAGE_TIMEOUT_SECONDS
from discordbot.utils.link_errors import link_fetch_error
from discordbot.services.platforms.base import (
    PlatformOutput,
    PlatformDownloader,
    PlatformConversation,
)

# Path-anchored rather than host-anchored, so unlike Facebook, Instagram and Douyin no `url_filter`
# is needed on top: `/status/<digits>` names a post and nothing else on the site does, so a profile,
# the home page or a list never matches in the first place. Both hosts are matched because
# `twitter.com` still 301s to `x.com` rather than being retired, and the mobile hosts are what a
# phone's share sheet copies. `/i/web/status/<id>` is the form a link with no handle takes.
#
# The username segment is matched but never trusted: x.com serves the same post under ANY handle,
# including one that does not exist (measured — it 307s to the real author's URL), so only the id
# is load-bearing and every URL this module republishes is built by `post_url` from the author the
# payload names. That also drops the `?s=46&t=<token>` tail, which is minted per share and names
# whoever passed the link on — the same trap `threads.py` documents for `?xmt=`.
#
# The optional `/photo/1` or `/video/1` suffix is what x.com's own image lightbox copies. The query
# tail is matched so `?s=46&t=<token>` is consumed rather than left dangling, and it ends on
# `[A-Za-z0-9_-]` for the reason every pattern here does: a link written straight after Chinese or
# Japanese text stops at the terminator instead of swallowing it.
TWITTER_URL_RE = re.compile(
    rf"{URL_START_ANCHOR}https?://(?:www\.|mobile\.|m\.)?(?:x|twitter)\.com/"
    r"(?:i/web/status|[A-Za-z0-9_]{1,20}/status(?:es)?)/\d+"
    r"(?:/(?:photo|video)/\d+)?/?"
    r"(?:\?[A-Za-z0-9=&%_.-]*[A-Za-z0-9_-])?"
)

_STATUS_PATH_RE = re.compile(r"/status(?:es)?/(?P<id>\d+)")

# A status id is a signed 64-bit snowflake, and one past that range is the only input the endpoint
# answers 400 to (measured: 2**63 - 1 and every smaller nonsense id answer 404, which classifies as
# unavailable and earns the ⚠️ mark; a 400 does not classify at all and would reach the reader as
# the cross that means the bot broke). The pattern cannot carry the bound — a 19-digit prefix of a
# 21-digit id matches just as well — so the check lives where the id is read.
_MAX_STATUS_ID = 2**63 - 1

_SYNDICATION_URL = "https://cdn.syndication.twimg.com/tweet-result"
_CANONICAL_TWITTER_ORIGIN = "https://x.com"

# The endpoint rejects a request with no User-Agent (HTTP 400, empty body) and answers `{}` to one
# with no token. The token's VALUE is never checked — every non-empty string tested returned the
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

# Twitter's own image CDN serves `medium` from a bare URL while the payload reports the ORIGINAL's
# dimensions beside it, so a caller that trusts the JSON and takes the bare URL gets a 900x1200
# image labelled 3024x4032. Measured: bare 57,759 bytes / 900x1200, `?name=orig` 978,797 bytes /
# 3024x4032. Where the original is smaller than the `large` bound the two are byte-identical, so
# this is never a downgrade.
_ORIGINAL_SIZE_QUERY = "?name=orig"

# Twitter's own cap. Four is also what the Facebook and Instagram cards allow, which is coincidence
# rather than coupling: theirs is a rendered-message bound and this is the platform's own limit.
_MAX_MEDIA_ITEMS = 4


def _time_of(*, value: str) -> datetime | None:
    """The ISO-8601 Zulu timestamp as an aware datetime, or None when it is absent or malformed.

    Parsed rather than trusted: the field is a string in the payload and every other reader here
    goes through a pydantic mirror, but a date is the one value a mirror cannot validate into the
    shape a caller wants without also deciding what a bad one costs. A bad one costs the timestamp
    and nothing else.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


class TwitterURL(BaseModel):
    """Parses and normalises a Twitter post URL.

    Attributes:
        raw_url: Original URL provided by the caller.
    """

    raw_url: str = Field(..., description="Original Twitter URL provided by the caller")

    @computed_field
    @cached_property
    def status_id(self) -> str:
        """The numeric post id the URL names, or an empty string when it names none.

        Read off the path rather than the last segment, so a `/photo/1` suffix does not become the
        id. This is the whole of what identifies a post: the handle in front of it is decorative.

        Returns:
            The status id, or an empty string when the URL is not a post URL.
        """
        match = _STATUS_PATH_RE.search(string=urlparse(self.raw_url).path)
        return match.group("id") if match else ""


def post_url(*, handle: str, status_id: str) -> str:
    """The canonical URL for a post, built from the author the payload named.

    Args:
        handle: The author's screen name, without the leading `@`.
        status_id: The post's numeric id.

    Returns:
        The canonical post URL.
    """
    return f"{_CANONICAL_TWITTER_ORIGIN}/{handle or 'i'}/status/{status_id}"


class TwitterOutput(PlatformOutput):
    """One post, the single shape a conversation is built from.

    The nine shared fields and `is_readable` come from `PlatformOutput`; what is declared here is
    Twitter's own reality. One inherited field means something narrower: `comment_count` is the
    size of the thread under the post rather than its direct replies, that being the only reply
    figure the endpoint reports for the post asked for.

    And there is deliberately no `retweet_count`. The endpoint publishes one for an embedded parent
    or quoted post and never for the post actually asked for, and a field a platform cannot fill is
    worse than an absent one — the same rule that keeps `share_count` off Instagram.

    Attributes:
        video_poster_urls: Still frames for any video, in media order; what the card shows, since
            nothing here downloads the clip.
        quoted: The post this one quotes, when it quotes a readable one.
        is_truncated: Whether the body is a fragment Twitter cut and will not serve in full. Only
            the post asked for can report it; see the module docstring.
    """

    video_poster_urls: list[str] = Field(
        default_factory=list, description="Still frames for any video, in media order"
    )
    # One level only, and never populated on a quoted post itself: 46 sampled quotes carried no
    # nested quote. `_build_output` bounds it anyway rather than trusting the sample.
    quoted: "TwitterOutput | None" = Field(
        default=None,
        description="The post this one quotes, absent when it quotes nothing readable",
    )
    is_truncated: bool = Field(
        default=False,
        description="Whether Twitter cut the body and will not serve the rest",
        examples=[False],
    )


class TwitterConversation(PlatformConversation[TwitterOutput]):
    """A parsed Twitter post, and the post it replies to when there is a readable one.

    The three fields and the four accessors come from `PlatformConversation`. Two of them carry a
    Twitter-specific truth.

    `chain` is `[parent, target]` when the post replies to a readable one and `[target]` otherwise.
    It never runs deeper: the parent arrives inside the same response, free, while a grandparent is
    a second request, and this module does not spend one.

    `reply_branches` is permanently empty and `selected_comment_id` with it, because the endpoint
    serves no reply content at all — only a count, which rides on `target.comment_count`. Both are
    carried so a caller written against Threads, Facebook or Instagram reads this without learning
    a second set of rules.
    """


class _TwitterPayload(BaseModel):
    """Base for the syndication payload mirrors, tolerating the nulls the endpoint serves.

    Every field below is optional with a default, because the payload omits rather than nulls most
    of what a given post lacks — `photos` is present-but-empty on a video post, `mediaDetails` is
    absent entirely on a text post, and `parent` only appears on a reply.
    """

    model_config = {"extra": "ignore"}


class _User(_TwitterPayload):
    """The author block, present on every live post and on every embedded one."""

    screen_name: str = Field(default="", description="Handle, without the leading @")
    profile_image_url_https: str = Field(default="", description="Avatar URL, 48px")


class _VideoVariant(_TwitterPayload):
    """One rendition of a video.

    Read off `mediaDetails[].video_info.variants` rather than the top-level `video.variants`: the
    two describe the same renditions under DIFFERENT key names (`content_type`/`url` here,
    `type`/`src` there) and only this one carries `bitrate`, which is the whole basis for picking
    a rendition. The first entry is always the HLS manifest, which has no bitrate and is not a file.
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
    position, and `video` holds at most one. Measured on a post carrying photo/video/photo/photo,
    where `photos` returned three entries and the video's place in the order was gone.
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
    # The endpoint reports the reply figure under two names depending on the post's POSITION: the
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

    One level deeper than the key name suggests: `tombstone.text` is an OBJECT carrying `entities`,
    `rtl` and a `text` of its OWN, not the string. Mirroring it one level short fails validation on
    every real refusal while the rare wordless one passes, since there is then nothing to validate
    — which is exactly how it looks when only the empty case was tried.
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
        """The refusal wording, or an empty string when the payload carried none."""
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

    model_config = {"extra": "ignore", "populate_by_name": True}


def _body_text(*, tweet: _Tweet) -> str:
    """The post body: the trailing media link removed, and every other link expanded.

    Two index bases live in one payload and mixing them corrupts any post containing an emoji.
    `display_text_range` is UTF-16 CODE UNITS (measured: 31 of 31 discriminating samples), while
    `entities.*.indices` are CODEPOINTS (36 of 36). Python strings are indexed by codepoint, so
    slicing by `display_text_range` directly leaks the first characters of the trailing `t.co`
    into the body — `'...en htt'` on a real post. Encoding to UTF-16 first is what makes the two
    agree.

    The trailing link that range excludes is the post's OWN media link, which points at the post
    we are already rendering. Every other link stays and is expanded from its `t.co` form, since
    the body is what the model reads and a `t.co` tells it nothing.
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


def _media_urls(*, tweet: _Tweet) -> tuple[list[str], list[str], list[str]]:
    """Splits the attachments into images, playable videos and their poster frames.

    Walks `mediaDetails` rather than `photos` / `video`, which lose a mixed post's order. An
    animated GIF is a video here, as Twitter models it: it is served as an mp4 with a single
    bitrate-0 rendition.

    Returns:
        `(image_urls, video_urls, video_poster_urls)`, each in post order.
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


def _build_output(*, tweet: _Tweet, include_quoted: bool = True) -> TwitterOutput:
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


class TwitterDownloader(PlatformDownloader):
    """Reads a Twitter post through the public syndication endpoint.

    Holds no state and writes nothing to disk, so one instance serves every caller — the Facebook
    and Instagram shape. There is deliberately no `parse`: nothing here downloads media. The card
    hands Discord the image and poster URLs to fetch itself, and the clip rides as a link.

    Attributes:
        timeout: Per-request bound in seconds.
    """

    timeout: float = Field(
        default=TWITTER_PAGE_TIMEOUT_SECONDS, description="Per-request timeout in seconds."
    )

    def _fetch_tweet(self, *, status_id: str) -> dict[str, Any]:
        """Fetches one post's payload.

        Raises:
            LinkReadError: Classified from the HTTP failure where the status says unambiguously
                what happened; a bare `RuntimeError` otherwise. A 404 here means no such post.
            RuntimeError: If the body is not the JSON object the endpoint documents.
        """
        try:
            response = requests.get(
                _SYNDICATION_URL,
                params={"id": status_id, "lang": "en", "token": _REQUEST_TOKEN},
                headers=_REQUEST_HEADERS,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise link_fetch_error(
                error=error, url=f"{_SYNDICATION_URL}?id={status_id}"
            ) from error
        # Decoded outside the clause above rather than inside it: `requests.JSONDecodeError`
        # subclasses `RequestException`, so a body that is not JSON would otherwise be classified
        # as a failed FETCH and reported as one, of a request that landed fine.
        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError(f"Twitter served a non-JSON body for post {status_id}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"Twitter served an unexpected body for post {status_id}")
        return payload

    def parse_metadata(self, *, url: str) -> TwitterConversation:
        """Parses a Twitter post URL into the conversation, writing nothing to disk.

        The chain is `[parent, target]` when the post replies to a readable one, `[target]`
        otherwise, and empty when the post could not be read at all — a deleted, protected or
        suspended post, or an id that never existed or could not be one. An empty conversation is
        the ordinary outcome for a pasted link rather than an error, so it comes back rather than
        raising, and the caller marks it unreadable.

        Args:
            url: The post URL, in any of the forms `TWITTER_URL_RE` matches.

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
        target = _build_output(tweet=tweet)
        chain = [_build_output(tweet=tweet.parent), target] if tweet.parent else [target]
        return TwitterConversation(chain=chain)


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
