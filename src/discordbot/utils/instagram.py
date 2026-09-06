"""Instagram post URL parsing and page extraction.

Shared by `parse_instagram` (which expands a pasted link into embeds) and `gen_reply` (which
reads the post into answer context), the same split `utils/threads.py` and `utils/facebook.py`
serve. The three modules deliberately agree on both halves of their surface.

`parse_metadata` is the entry point on all three and means the same thing on each: parse the
post, write nothing to disk. Only Threads additionally has `parse`, because only Threads
downloads a file that then has to be cleaned up.

`InstagramConversation` mirrors `ThreadsConversation`: a `chain` whose last element is the
post the link names, and `reply_branches` holding the discussion under it, one branch per
top-level comment with its replies threaded behind it. Instagram has no ancestor posts, so the
chain is always exactly one element — the shape is kept anyway so a caller that renders a
Threads conversation renders this one without learning a second set of rules.

Logged out, Instagram is the most generous of the three. The page ships the whole post — full
caption, every carousel image at original resolution, the counters, AND the complete comment
list — inside `<script type="application/json">` blocks, under a module whose own name says
who it is for (`PolarisLoggedOutDesktopWWWMedia`). Measured 2026-09-07 against a 9-image post
reporting 11 comments: 9 images and all 11 comments came back. That is where this source beats
Facebook, whose comments are a preload of a much longer thread.

Three things shape the parser.

The page carries OTHER posts by the same author, in a "more posts" rail, and those nodes look
like the target: same `code` key, same `caption`, same `carousel_media_count`. They differ by
what they omit — no `carousel_media` list, no `like_count`, no `taken_at` — but the reliable
test is the shortcode from the URL, so `_find_media` matches on that and never on position.

A comment URL cannot be fetched. `/p/<code>/c/<comment_id>/` answers HTTP 200 with a page
carrying no post payload at all (measured: 503KB and no media node, against 805KB and a full
one for the bare post), so `clean_url` strips the comment segment and the id is carried
separately in `comment_id`. Everything in the query goes the same way, `img_index` included;
`stkn` is the reason it has to, being a per-share token that names whoever passed the link on,
exactly like Facebook's `rdid`.

Image candidates are ordered widest first and carry no dimensions of their own, so
`candidates[0]` is the original: the ones after it have a size in their `stp=` segment
(`p720x720`, `s640x640`) and the first does not.
"""

import re
import json
from typing import Any
from datetime import UTC, datetime
from functools import cached_property
from urllib.parse import urlparse
from collections.abc import Iterator

import logfire
from pydantic import Field, BaseModel, computed_field
import requests

from discordbot.utils.urls import URL_START_ANCHOR
from discordbot.typings.timeouts import INSTAGRAM_PAGE_TIMEOUT_SECONDS

_CANONICAL_INSTAGRAM_ORIGIN = "https://www.instagram.com"

# Host-anchored like the Douyin and Facebook patterns, because the path alone cannot separate a
# post from a profile: `/p/<code>/`, `/reel/<code>/` and `/tv/<code>/` are posts, `/<user>/` is
# not, and `/<user>/p/<code>/` (the form Instagram's own `og:url` reports) is.
# `is_instagram_post_url` makes that call on the parsed path. The tail class mirrors
# `THREADS_URL_RE`, so a link written straight after Chinese or Japanese text is matched without
# swallowing the terminator.
INSTAGRAM_URL_RE = re.compile(
    rf"{URL_START_ANCHOR}https?://(?:[a-z0-9-]+\.)*instagram\.com/"
    r"[A-Za-z0-9_.?=&%/~:+-]*[A-Za-z0-9_-]/?"
)

# The three post shapes, each optionally prefixed by the author's handle. The shortcode charset
# is Instagram's base64url alphabet. `/reels/audio/<id>/` is a sound page rather than a post and
# is refused here, since it otherwise parses with `audio` as the shortcode.
_POST_PATH_RE = re.compile(
    r"^/(?:(?P<user>[A-Za-z0-9_.]+)/)?(?P<kind>p|reel|reels|tv)/(?!audio/)(?P<code>[A-Za-z0-9_-]+)"
)

# A comment permalink hangs off the post path. It is parsed but never fetched: see the module
# docstring for what that URL answers with.
_COMMENT_PATH_RE = re.compile(r"/c/(?P<comment>[0-9]+)")

# What the page will only hand to a browser, the same set `utils/facebook.py` needs. A crawler
# UA does get a page here (unlike Facebook, which answers 400), but it is a 650KB shell whose
# post payload is missing the carousel and every comment.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_JSON_SCRIPT_RE = re.compile(r'<script type="application/json"[^>]*>(.*?)</script>', re.DOTALL)

# Instagram's own media-type enum, as it appears on both a post and a carousel child.
_MEDIA_TYPE_VIDEO = 2

# What a parsed JSON payload can hold. Spelled out rather than left as a bare `Any`, which the
# project's checker refuses, and mirroring the union `utils/facebook.py` walks with.
JsonValue = dict[str, Any] | list[Any] | str | float | None


def _walk(*, node: JsonValue) -> Iterator[dict[str, Any]]:
    """Yields every dict in a parsed JSON tree, outermost first."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(node=value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(node=value)


def _deep_get(node: JsonValue, *keys: str) -> JsonValue:
    """Follows a chain of dict keys, returning None as soon as one is missing."""
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _str_of(*, value: JsonValue) -> str:
    """A string field, or an empty one when the page served a null or another type."""
    return value if isinstance(value, str) else ""


def _int_of(*, value: JsonValue) -> int:
    """An integer field, or zero when the page served something else."""
    return value if isinstance(value, int) else 0


def _time_of(*, value: JsonValue) -> datetime | None:
    """A unix timestamp as an aware datetime, or None when the page omitted it."""
    return datetime.fromtimestamp(value, tz=UTC) if isinstance(value, int) and value else None


def is_instagram_post_url(*, url: str) -> bool:
    """Whether a matched Instagram URL names one post rather than a profile or the home page.

    The regex matches the host, so this is where `/`, `/<user>/`, `/explore/...` and the rest
    are refused. It matters on both paths: the expansion cog would otherwise put a failure
    reaction on an ordinary profile link, and the reply pipeline would spend a full page fetch
    to learn there is no post.

    Args:
        url: A URL `INSTAGRAM_URL_RE` matched.

    Returns:
        True when the URL names a post.
    """
    return bool(InstagramURL(raw_url=url).shortcode)


class InstagramURL(BaseModel):
    """Parses and normalises an Instagram post URL.

    Unlike a Threads or Facebook share link, every accepted spelling names its post directly, so
    there is no redirect to resolve and `shortcode` is empty only when the URL is not a post.

    Attributes:
        raw_url: Original Instagram URL provided by the caller.
    """

    raw_url: str = Field(..., description="Original Instagram URL provided by the caller")

    @computed_field
    @cached_property
    def shortcode(self) -> str:
        """The post's shortcode, or an empty string when the URL names no post."""
        match = _POST_PATH_RE.match(string=urlparse(self.raw_url).path)
        return match.group("code") if match else ""

    @computed_field
    @cached_property
    def comment_id(self) -> str:
        """The comment a `/c/<id>/` permalink singles out, empty when it names none."""
        match = _COMMENT_PATH_RE.search(string=urlparse(self.raw_url).path)
        return match.group("comment") if match else ""

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        """The bare post URL: the fetchable form, and the one safe to publish back.

        Everything else goes. The query carries `stkn`, a per-share token naming whoever passed
        the link on (Facebook's `rdid` by another name), plus `img_index` and `utm_source`; the
        `/c/<id>/` segment has to go too, because that URL answers with a page carrying no post
        payload at all. The author handle is dropped with them: `/p/<code>/` and
        `/<user>/p/<code>/` return the same page, and the shorter one cannot go stale when a
        handle changes.
        """
        match = _POST_PATH_RE.match(string=urlparse(self.raw_url).path)
        if not match:
            return self.raw_url
        kind = "reel" if match.group("kind") in {"reel", "reels"} else match.group("kind")
        return f"{_CANONICAL_INSTAGRAM_ORIGIN}/{kind}/{match.group('code')}/"


class FetchedPage(BaseModel):
    """One page fetch: its HTML and the URL the request ended on.

    Attributes:
        html: The fetched page's HTML body.
        final_url: The URL the request ended on after redirects.
    """

    html: str = Field(..., description="The fetched page's HTML body")
    final_url: str = Field(..., description="The URL the request ended on after redirects")

    @computed_field
    @cached_property
    def is_login_wall(self) -> bool:
        """Whether the fetch was redirected to a login or challenge page."""
        path = urlparse(self.final_url).path.lower()
        return path.startswith(("/accounts/login", "/challenge", "/accounts/suspended"))


class InstagramOutput(BaseModel):
    """One post OR one comment, the single shape a conversation is built from.

    Deliberately one type for both, exactly as `ThreadsOutput` is: a caller that walks a
    Threads conversation walks this one with the same code. A comment simply leaves the fields
    a comment has no version of empty — it carries no media of its own and no comment count.

    Attributes:
        text: The caption, or the comment body.
        url: The permalink. For a comment this is the post's, since a comment permalink is not
            fetchable (see the module docstring) and would only publish a dead link.
        author_name: The author's handle, without the leading `@`.
        author_full_name: The author's display name, which comments do not carry.
        author_icon_url: The author's profile picture URL.
        image_urls: Original-resolution image URLs, in carousel order.
        video_urls: Playable video URLs, in carousel order.
        like_count: Likes this post or comment carries.
        comment_count: Comments the POST reports; zero on a comment.
        taken_at: When it was published.
        comment_id: The comment's own numeric id; empty on the post itself.
    """

    text: str = Field(default="", description="The caption, or the comment body")
    url: str = Field(default="", description="The permalink, the post's in both cases")
    author_name: str = Field(default="", description="The author's handle")
    author_full_name: str = Field(default="", description="The author's display name")
    author_icon_url: str = Field(default="", description="The author's profile picture URL")
    image_urls: list[str] = Field(
        default_factory=list, description="Original-resolution image URLs in carousel order"
    )
    video_urls: list[str] = Field(
        default_factory=list, description="Playable video URLs in carousel order"
    )
    like_count: int = Field(default=0, description="Likes this post or comment carries")
    comment_count: int = Field(default=0, description="Comments the post reports; 0 on a comment")
    taken_at: datetime | None = Field(default=None, description="When it was published")
    comment_id: str = Field(default="", description="The comment's own id; empty on the post")

    @computed_field
    @cached_property
    def is_readable(self) -> bool:
        """Whether enough came back to be worth showing."""
        return bool(self.text or self.image_urls or self.video_urls)


class InstagramConversation(BaseModel):
    """One Instagram post and the discussion under it, shaped like `ThreadsConversation`.

    Attributes:
        chain: The post the link names. Always exactly one element — Instagram has no ancestor
            posts — but kept as a list so `target` means the same here as it does on Threads.
        reply_branches: One branch per top-level comment, each ordered from that comment
            outward through its replies.
        selected_comment_id: The comment a `/c/<id>/` permalink named, empty when none did.
    """

    chain: list[InstagramOutput] = Field(
        default_factory=list, description="The post the link names, as a one-element chain"
    )
    reply_branches: list[list[InstagramOutput]] = Field(
        default_factory=list, description="One branch per top-level comment, replies behind it"
    )
    selected_comment_id: str = Field(
        default="", description="The comment a `/c/<id>/` permalink named"
    )

    @computed_field
    @cached_property
    def target(self) -> InstagramOutput | None:
        """The post the link named, or None when the page carried none."""
        return self.chain[-1] if self.chain else None

    @property
    def comments(self) -> list[InstagramOutput]:
        """Every comment, flattened out of the branches in page order.

        A plain property rather than a computed field, on all three sources: it re-slices data
        `reply_branches` already carries, so serializing it would put every comment in a dump
        twice. The two computed fields resolve a POINTER instead, which a dump cannot derive on
        its own and which is what a hand test wants to see.
        """
        return [comment for branch in self.reply_branches for comment in branch]

    @property
    def posts(self) -> list[InstagramOutput]:
        """Everything the page yielded: the post first, then its comments in page order."""
        return [*self.chain, *self.comments]

    @computed_field
    @cached_property
    def selected_comment(self) -> InstagramOutput | None:
        """The comment the URL singled out, or None when it named none or it was not on the page."""
        if not self.selected_comment_id:
            return None
        return next(
            (
                comment
                for comment in self.comments
                if comment.comment_id == self.selected_comment_id
            ),
            None,
        )


class InstagramDownloader(BaseModel):
    """Reads a public Instagram post out of its page.

    Holds no state and writes nothing to disk, so one instance serves every caller; it is a
    class rather than a function so a test can replace `_fetch_page`, which is the one seam
    that touches the network.
    """

    def _fetch_page(self, *, url: str) -> FetchedPage:
        """Fetches a page with the browser headers the full payload needs.

        Raises:
            RuntimeError: The page could not be fetched.
        """
        try:
            response = requests.get(
                url=url, headers=_BROWSER_HEADERS, timeout=INSTAGRAM_PAGE_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return FetchedPage(html=response.text, final_url=response.url)
        except requests.RequestException as error:
            raise RuntimeError(f"Failed to fetch HTML from {url}: {error}") from error

    @staticmethod
    def _json_payloads(*, html: str) -> Iterator[Any]:
        """Yields every embedded JSON block on the page, skipping the ones that do not parse."""
        for match in _JSON_SCRIPT_RE.finditer(string=html):
            try:
                yield json.loads(s=match.group(1))
            except ValueError:
                logfire.debug("Skipped an unparsable Instagram JSON block", _exc_info=True)
                continue

    @staticmethod
    def _find_media(*, payloads: list[Any], shortcode: str) -> dict[str, Any] | None:
        """The media node for the wanted post, or None when the page carries none.

        Matched on the shortcode alone. The page's "more posts by this author" rail serialises
        nodes carrying the same keys, so anything positional would read a neighbouring post's
        caption as this one's; those nodes also omit `carousel_media` and `image_versions2`,
        which is what the second test below leans on when several nodes share the code.
        """
        fallback: dict[str, Any] | None = None
        for payload in payloads:
            for node in _walk(node=payload):
                if node.get("code") != shortcode:
                    continue
                if "carousel_media" in node or "image_versions2" in node:
                    return node
                fallback = fallback or node
        return fallback

    @staticmethod
    def _media_urls_of(*, media: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Original-resolution image URLs and playable video URLs, in carousel order.

        A carousel keeps its items under `carousel_media`; a single image or video is the node
        itself. `candidates[0]` is the original — see the module docstring for how that is known
        when the candidates carry no dimensions.
        """
        children = media.get("carousel_media")
        items = children if isinstance(children, list) else [media]
        image_urls: list[str] = []
        video_urls: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("media_type") == _MEDIA_TYPE_VIDEO:
                versions = item.get("video_versions")
                first = versions[0] if isinstance(versions, list) and versions else None
                url = _str_of(value=_deep_get(first, "url"))
                if url:
                    video_urls.append(url)
                    continue
            candidates = _deep_get(item, "image_versions2", "candidates")
            first = candidates[0] if isinstance(candidates, list) and candidates else None
            url = _str_of(value=_deep_get(first, "url"))
            if url and url not in image_urls:
                image_urls.append(url)
        return image_urls, video_urls

    @staticmethod
    def _comment_branches(*, payloads: list[Any], post_url: str) -> list[list[InstagramOutput]]:
        """Every comment on the page, grouped into branches the way Threads groups replies.

        A comment carries `parent_comment_id` when it answers another one, so a reply is
        threaded behind the comment it answers instead of standing alone; everything else opens
        its own branch. On a public post this really is the whole discussion rather than a
        preload — measured against a post reporting 11 comments, 11 came back — which is the
        one place this source has more to give than Facebook's.

        A comment is recognised by carrying both `comment_like_count` and `text`, which no other
        node on the page does.
        """
        found: dict[str, InstagramOutput] = {}
        parents: dict[str, str] = {}
        for payload in payloads:
            for node in _walk(node=payload):
                if "comment_like_count" not in node or "text" not in node:
                    continue
                comment_id = str(node.get("pk") or "")
                if not comment_id.isdigit() or comment_id in found:
                    continue
                parent = str(node.get("parent_comment_id") or "")
                if parent.isdigit():
                    parents[comment_id] = parent
                found[comment_id] = InstagramOutput(
                    text=_str_of(value=node.get("text")),
                    url=post_url,
                    author_name=_str_of(value=_deep_get(node, "user", "username")),
                    author_icon_url=_str_of(value=_deep_get(node, "user", "profile_pic_url")),
                    like_count=_int_of(value=node.get("comment_like_count")),
                    taken_at=_time_of(value=node.get("created_at")),
                    comment_id=comment_id,
                )
        branches: list[list[InstagramOutput]] = []
        index: dict[str, list[InstagramOutput]] = {}
        for comment_id, comment in found.items():
            parent = parents.get(comment_id)
            branch = index.get(parent) if parent else None
            if branch is None:
                branch = [comment]
                branches.append(branch)
            else:
                branch.append(comment)
            index[comment_id] = branch
        return branches

    def parse_metadata(self, *, url: str) -> InstagramConversation:
        """Reads one public Instagram post and the comments under it.

        Named to match `ThreadsDownloader.parse_metadata` and `FacebookDownloader.parse_metadata`,
        and meaning the same on all three: parse the post and write nothing to disk. There is no
        `parse` counterpart here because nothing is downloaded — the images ride out as URLs.

        A private account, a deleted post and a login wall are one outcome from outside, and all
        three come back as an empty conversation rather than an error.

        Args:
            url: The Instagram post URL in any accepted form.

        Returns:
            The parsed conversation; its `chain` is empty when the post could not be read.

        Raises:
            RuntimeError: The page could not be fetched at all.
        """
        instagram_url = InstagramURL(raw_url=url)
        if not instagram_url.shortcode:
            logfire.info("An Instagram URL names no post; treating it as unreadable", url=url)
            return InstagramConversation()

        fetched = self._fetch_page(url=instagram_url.clean_url)
        if fetched.is_login_wall:
            logfire.info(
                "An Instagram post is not public; treating it as unreadable",
                url=instagram_url.clean_url,
            )
            return InstagramConversation()

        payloads = list(self._json_payloads(html=fetched.html))
        media = self._find_media(payloads=payloads, shortcode=instagram_url.shortcode)
        if media is None:
            logfire.info(
                "An Instagram page carried no post payload; treating it as unreadable",
                url=instagram_url.clean_url,
                html_length=len(fetched.html),
            )
            return InstagramConversation()

        image_urls, video_urls = self._media_urls_of(media=media)
        post = InstagramOutput(
            text=_str_of(value=_deep_get(media, "caption", "text")),
            url=instagram_url.clean_url,
            author_name=_str_of(value=_deep_get(media, "user", "username")),
            author_full_name=_str_of(value=_deep_get(media, "user", "full_name")),
            author_icon_url=_str_of(value=_deep_get(media, "user", "profile_pic_url")),
            image_urls=image_urls,
            video_urls=video_urls,
            like_count=_int_of(value=media.get("like_count")),
            comment_count=_int_of(value=media.get("comment_count")),
            taken_at=_time_of(value=media.get("taken_at")),
        )
        return InstagramConversation(
            chain=[post],
            reply_branches=self._comment_branches(
                payloads=payloads, post_url=instagram_url.clean_url
            ),
            selected_comment_id=instagram_url.comment_id,
        )


if __name__ == "__main__":
    """
    Keep this for self development and testing.
    DO NOT REMOVE THIS FOR ANY REASON.
    """
    from rich.console import Console

    console = Console()
    downloader = InstagramDownloader()
    metadata = downloader.parse_metadata(
        url="https://www.instagram.com/p/Dc5eNjYkoZE/c/17946527169275440/?img_index=1"
    )
    console.print(metadata)
