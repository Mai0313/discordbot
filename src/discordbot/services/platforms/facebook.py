"""Facebook post URL parsing and page extraction.

Shared by `parse_facebook` (which expands a pasted link into embeds) and `gen_reply` (which
reads the post into answer context), the same split `services/platforms/threads.py` serves.

Unlike its Threads counterpart this module downloads nothing. Both callers work from the image
URLs alone: the cog hands them to Discord, which fetches them itself, and the reply builder
passes them to `load_image_bytes`, which is where every link source already gets image bytes
(`services/platforms/threads.py` keeps a downloader only for the video files that path writes to disk, and
there are none here). So there is no scratch directory anywhere in this feature.

Two things about this source differ from Threads and shape everything below.

The page is only served to something that looks like a browser. A bare `User-Agent` gets an
HTTP 400 and a crawler-shaped one gets a 348KB shell carrying nothing but Open Graph tags,
whose description is truncated at ~190 characters. The full ~950KB page, with the post's own
GraphQL payload in it, needs the complete header set in `_BROWSER_HEADERS`: measured
2026-09-06, dropping `Sec-Fetch-Mode` alone is enough to lose it. The mobile hosts
(`m.`, `mbasic.`, `touch.`) redirect to a login page and are never worth trying.

And the payload is not a schema this module can mirror the way `services/platforms/threads.py` mirrors
Threads'. The post sits somewhere inside one of ~58 `<script type="application/json">` blocks
whose shape is dominated by Facebook's own module loader, repeated two or three times per
page, so the parser walks for a node carrying the fields a story has rather than validating a
structure. Only the extracted result is modelled.

Video is deliberately out of scope: a logged-out `Video` node carries `permalink_url` and
`captions_url` but no `playable_url`, so there is no file to fetch. `FacebookOutput.video_urls`
therefore holds permalinks for a caller to link to, never something to download.
"""

import re
import json
import base64
from typing import Any
from datetime import UTC, datetime
from functools import cached_property
from urllib.parse import parse_qs, urlparse, urlunparse
from collections.abc import Iterator

import logfire
from pydantic import Field, BaseModel, computed_field
import requests

from discordbot.utils.urls import URL_START_ANCHOR, host_matches_domain
from discordbot.typings.timeouts import FACEBOOK_PAGE_TIMEOUT_SECONDS
from discordbot.utils.link_errors import link_fetch_error
from discordbot.services.platforms.base import (
    PlatformOutput,
    PlatformDownloader,
    PlatformConversation,
)

# Every host Facebook serves posts on. `fb.watch` and `fb.com` are the short forms its own share
# sheet emits; the mobile hosts are matched so a pasted one is recognised as a post URL, and
# `FacebookURL.clean_url` then aims the fetch at `www` where the payload actually is.
_FACEBOOK_DOMAINS = frozenset({"facebook.com", "fb.com", "fb.watch"})
_CANONICAL_FACEBOOK_ORIGIN = "https://www.facebook.com"

# Deliberately host-anchored rather than path-anchored, the shape `services/platforms/douyin.py` uses: a
# Facebook post is spelled at least six ways (`/share/p/<code>`, `/groups/<id>/posts/<id>`,
# `/groups/<id>/permalink/<id>`, `/<page>/posts/<id>`, `/permalink.php?story_fbid=`, and a
# group feed carrying `?multi_permalinks=`), and a path pattern covering all six would also
# match the profile and group-home URLs that are not posts at all. `is_facebook_post_url` makes
# that call on the parsed URL instead, where the query is readable. The tail class mirrors
# `THREADS_URL_RE`: ASCII URL characters ending on one that a real id or query value ends on,
# so a link written mid-sentence in Chinese or Japanese is matched without swallowing the
# terminator.
FACEBOOK_URL_RE = re.compile(
    rf"{URL_START_ANCHOR}https?://(?:[a-z0-9-]+\.)*(?:facebook\.com|fb\.com|fb\.watch)/"
    r"[A-Za-z0-9_.?=&%/~:+-]*[A-Za-z0-9_-]/?"
)

# The path shapes that name a post on their own. A `share/p/<code>` link names one only through
# the redirect it answers with, exactly as a Threads share link does.
_GROUP_POST_PATH_RE = re.compile(r"^/groups/(?P<group>[^/]+)/(?:posts|permalink)/(?P<post>[0-9]+)")
_PAGE_POST_PATH_RE = re.compile(r"^/(?:[^/]+)/(?:posts|videos)/(?P<post>[0-9]+)")
_NUMERIC_POST_PATH_RE = re.compile(r"^/(?:[0-9]+)/posts/(?P<post>[0-9]+)")
_SHARE_PATH_RE = re.compile(r"^/share/(?:p|v|r)/[A-Za-z0-9]+")

# A comment id is the whole point of the `?comment_id=` form, so it survives `clean_url` while
# every other query parameter is dropped. `rdid` and `share_url` are the reason the rest go:
# both are minted per share, so echoing them names whoever sent the link to the channel — the
# same trap `services/platforms/threads.py` documents for the `?xmt=` token.
_COMMENT_ID_PARAM = "comment_id"
_POST_ID_PARAMS = ("story_fbid", "multi_permalinks", "fbid")

# Kept by `clean_url` but never read as a post id. `id` names the page or profile that OWNS the
# post, and `permalink.php?story_fbid=<post>` cannot resolve without it — dropping it aims the
# fetch at a URL naming no owner, and the post comes back unreadable. Deliberately NOT in
# `_POST_ID_PARAMS`: `profile.php?id=<n>` carries the same parameter and is not a post at all.
_OWNER_ID_PARAMS = ("id",)

# What the page will only hand to a browser. Measured 2026-09-06: this exact set returns the
# full payload, a bare `User-Agent` returns HTTP 400, and a crawler UA returns an Open Graph
# shell with the post text cut at ~190 characters.
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

# A comment node's id is base64 of `comment:<post_id>_<comment_id>`, which is what makes both
# "the comment this URL names" and "the post this comment hangs off" exact matches rather than
# guesses. `legacy_fbid` carries the trailing id directly and wins for that half; the decode is
# the fallback for a node without it, and the only source for the leading one.
_COMMENT_ID_RE = re.compile(r"^comment:(?P<post>[0-9]+)_(?P<comment>[0-9]+)")


def is_facebook_post_url(*, url: str) -> bool:
    """Whether a matched Facebook URL names one post rather than a profile or a feed.

    The regex matches the host, so this is where a profile, a group home page, a marketplace
    listing or a bare `facebook.com` link is refused. Refusing them matters twice over: the
    expansion cog would otherwise put a failure reaction on an ordinary link, and the reply
    pipeline would spend a full page fetch to learn there is no post.

    Args:
        url: A URL `FACEBOOK_URL_RE` matched.

    Returns:
        True when the URL names a readable post, whether directly or through a share redirect.
    """
    facebook_url = FacebookURL(raw_url=url)
    return bool(facebook_url.post_id or facebook_url.is_share_link)


class FacebookURL(BaseModel):
    """Parses and normalises a Facebook post URL.

    Only some of the accepted shapes name their post: a `share/p/<code>` link resolves to one
    through its redirect, which is why `post_id` being empty is not the same as the URL being
    unreadable (see `is_share_link`).

    Attributes:
        raw_url: Original Facebook URL provided by the caller.
    """

    raw_url: str = Field(..., description="Original Facebook URL provided by the caller")

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        """The URL aimed at `www.facebook.com`, carrying only the query that names content.

        Everything but `comment_id` and the post-id parameters is dropped, because `rdid` and
        `share_url` are minted per share and name whoever passed the link on. That makes this
        the form safe to publish back into a channel as well as the form to fetch.
        """
        parsed = urlparse(self.raw_url)
        kept: list[str] = []
        query = parse_qs(parsed.query)
        for name in (*_POST_ID_PARAMS, *_OWNER_ID_PARAMS, _COMMENT_ID_PARAM):
            values = query.get(name)
            if values and values[0]:
                kept.append(f"{name}={values[0]}")
        host = (parsed.hostname or "").lower()
        origin = (
            _CANONICAL_FACEBOOK_ORIGIN
            if any(host_matches_domain(host=host, domain=d) for d in _FACEBOOK_DOMAINS)
            else f"{parsed.scheme}://{parsed.netloc}"
        )
        path = parsed.path if parsed.path == "/" else parsed.path.rstrip("/")
        return urlunparse(("", "", f"{origin}{path}", "", "&".join(kept), ""))

    @computed_field
    @cached_property
    def post_id(self) -> str:
        """The post id the URL names, or an empty string when only a redirect can name it."""
        parsed = urlparse(self.raw_url)
        for pattern in (_GROUP_POST_PATH_RE, _NUMERIC_POST_PATH_RE, _PAGE_POST_PATH_RE):
            match = pattern.match(string=parsed.path)
            if match:
                return match.group("post")
        query = parse_qs(parsed.query)
        for name in _POST_ID_PARAMS:
            values = query.get(name)
            if values and values[0].isdigit():
                return values[0]
        return ""

    @computed_field
    @cached_property
    def group_id(self) -> str:
        """The group the post sits in, or an empty string for a page or profile post.

        Read so the group's NAME can be matched on the page, which also serialises the groups
        it recommends beside the one being read.
        """
        match = _GROUP_POST_PATH_RE.match(string=urlparse(self.raw_url).path)
        if match:
            return match.group("group")
        feed = re.match(pattern=r"^/groups/(?P<group>[^/?]+)", string=urlparse(self.raw_url).path)
        return feed.group("group") if feed else ""

    @computed_field
    @cached_property
    def comment_id(self) -> str:
        """The comment id the URL singles out, or an empty string when it names no comment."""
        values = parse_qs(urlparse(self.raw_url).query).get(_COMMENT_ID_PARAM)
        return values[0] if values and values[0].isdigit() else ""

    @computed_field
    @cached_property
    def is_share_link(self) -> bool:
        """Whether the URL is a share form, which names its post only through the redirect."""
        return bool(_SHARE_PATH_RE.match(string=urlparse(self.raw_url).path))


class FetchedPage(BaseModel):
    """One page fetch: its HTML and the URL the request actually ended on.

    Where it landed is part of the result because a share link names its post only there, and
    because a redirect to a login page is how Facebook says the post is not public.

    Attributes:
        html: The fetched page's HTML body.
        final_url: The URL the request ended on after redirects.
    """

    html: str = Field(..., description="The fetched page's HTML body")
    final_url: str = Field(..., description="The URL the request ended on after redirects")

    @computed_field
    @cached_property
    def is_login_wall(self) -> bool:
        """Whether the fetch was redirected to a login or checkpoint page."""
        path = urlparse(self.final_url).path.lower()
        return path.startswith(("/login", "/checkpoint", "/recover"))


class FacebookOutput(PlatformOutput):
    """One post OR one comment, the single shape a conversation is built from.

    Deliberately one type for both, exactly as `ThreadsOutput` and `InstagramOutput` are: a
    caller that walks one platform's conversation walks the others with the same code. A comment
    leaves empty the fields it has no version of — it carries no media, no group and no counts
    of its own.

    Two of the inherited fields mean something slightly narrower here. `url` is the POST's
    permalink even on a comment, since Facebook's own comment permalink is a query on it rather
    than a page of its own; and `comment_count` is what the post REPORTS, which exceeds what the
    page preloads.

    Attributes:
        group_name: The group the POST was made in, empty for a page post and on every comment.
        share_count: Shares the post reports; zero on a comment.
        comment_id: The comment's own numeric id; empty on the post itself.
    """

    group_name: str = Field(default="", description="The group the post was made in, if any")
    share_count: int = Field(default=0, description="Shares the post reports; 0 on a comment")
    comment_id: str = Field(default="", description="The comment's own id; empty on the post")


class FacebookConversation(PlatformConversation[FacebookOutput]):
    """One Facebook post and the discussion under it, shaped like `ThreadsConversation`.

    What the three inherited fields mean on Facebook. `chain` always has exactly one element —
    Facebook serves no ancestor posts — and is a list only so `target` means the same here as it
    does on Threads. `reply_branches` holds one branch per top-level comment, and what the page
    preloads is a handful of a much longer thread, which is the one thing a caller must not
    present as the whole discussion. `selected_comment_id` is whatever a `?comment_id=` URL named.
    """

    @computed_field
    @cached_property
    def selected_comment(self) -> FacebookOutput | None:
        """The comment the URL singled out, or None when it named none or was not preloaded.

        A named comment that is not on the page is the ordinary miss rather than an error: the
        page preloads only the first handful, so a link to an old comment resolves to nothing
        and the caller shows the post alone.
        """
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


# What a parsed JSON payload can hold. Spelled out rather than left as a bare `Any`, which the
# project's checker refuses outright, and mirroring the union `services/platforms/threads.py` walks with. Every
# read below goes through one of the narrowing helpers under it, so a page that serves an
# unexpected shape yields an empty field instead of raising into a caller mid-expansion.
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


def _text_of(*, value: JsonValue) -> str:
    """Reads a `{"text": ...}` node's string, tolerating the null the page sometimes serves."""
    return _str_of(value=value.get("text")) if isinstance(value, dict) else ""


def _count_of(*, value: JsonValue) -> int:
    """Reads a count the page serves as a bare value, a `{"count": n}` wrapper, or "1,017".

    An int rather than the page's own formatted string, so every platform's counters are the
    same type and whoever renders them picks the formatting once.
    """
    if isinstance(value, dict):
        value = value.get("count")
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = value.replace(",", "").strip()
        return int(digits) if digits.isdigit() else 0
    return 0


def _time_of(*, value: JsonValue) -> datetime | None:
    """A unix timestamp as an aware datetime, or None when the page omitted it."""
    return datetime.fromtimestamp(value, tz=UTC) if isinstance(value, int) and value else None


def _comment_ids_of(*, node: dict[str, Any]) -> tuple[str, str]:
    """The post a comment hangs off and the comment's own id, either half empty when unreadable.

    Both come out of the base64 node id, which is what lets a group feed's other posts be told
    apart from the one being read; `legacy_fbid` carries only the comment's own half and wins
    for it, being there even on a node whose id does not decode.
    """
    legacy = node.get("legacy_fbid")
    comment_id = str(legacy) if isinstance(legacy, (int, str)) and str(legacy).isdigit() else ""
    node_id = node.get("id")
    if not isinstance(node_id, str):
        return "", comment_id
    try:
        # Padded because Facebook serves these unpadded; the extra `=` are ignored when the
        # length is already a multiple of four.
        decoded = base64.b64decode(node_id + "==").decode(encoding="utf-8", errors="replace")
    except ValueError:
        return "", comment_id
    match = _COMMENT_ID_RE.match(string=decoded)
    if match is None:
        return "", comment_id
    return match.group("post"), comment_id or match.group("comment")


class FacebookDownloader(PlatformDownloader):
    """Reads a public Facebook post out of its page.

    Holds no state and writes nothing to disk, so one instance serves every caller; it is a
    class rather than a function so a test can replace `_fetch_page` the way the Threads tests
    do, which is the seam that keeps every test off the network.
    """

    def _fetch_page(self, *, url: str) -> FetchedPage:
        """Fetches a page with the browser headers Facebook will only answer in full to.

        Raises:
            LinkRetryableError: The platform refused the request or never answered.
            LinkUnavailableError: The platform answered that there is no such page.
            RuntimeError: The fetch failed in a way HTTP does not classify.
        """
        try:
            response = requests.get(
                url=url, headers=_BROWSER_HEADERS, timeout=FACEBOOK_PAGE_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return FetchedPage(html=response.text, final_url=response.url)
        except requests.RequestException as error:
            raise link_fetch_error(error=error, url=url) from error

    @staticmethod
    def _json_payloads(*, html: str) -> Iterator[Any]:
        """Yields every embedded JSON block on the page, skipping the ones that do not parse.

        Skipping rather than failing is what keeps one truncated block, of the roughly sixty a
        page carries, from costing the post.
        """
        for match in _JSON_SCRIPT_RE.finditer(string=html):
            try:
                yield json.loads(s=match.group(1))
            except ValueError:
                # `json.JSONDecodeError` subclasses this, as does the int-string conversion
                # limit a very large embedded number can trip.
                logfire.debug("Skipped an unparsable Facebook JSON block", _exc_info=True)
                continue

    @staticmethod
    def _find_story(*, payloads: list[Any], post_id: str) -> dict[str, Any] | None:
        """The story node for the wanted post, or None when the page carries none.

        A story is recognised by the fields it carries rather than by its path: the page nests
        the same node under several route-dependent keys (`data.node_v2`, `data.node`) and
        repeats it two or three times. Matching `post_id` is what makes a group feed usable —
        it serialises several posts, only one of which the URL named — and a feed's other
        entries come through as stubs carrying no message at all.
        """
        fallback: dict[str, Any] | None = None
        for payload in payloads:
            for node in _walk(node=payload):
                if "post_id" not in node or "creation_time" not in node:
                    continue
                if post_id and str(node.get("post_id")) != post_id:
                    continue
                if _deep_get(
                    node,
                    "comet_sections",
                    "content",
                    "story",
                    "comet_sections",
                    "message_container",
                    "story",
                    "message",
                ):
                    return node
                fallback = fallback or node
        # A post with no text at all is legitimate (an image-only post), so a node that matched
        # the id but carried no message is still the answer when nothing richer turned up.
        return fallback

    @staticmethod
    def _media_of(*, story: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Full-resolution image URLs and video permalinks carried by the story's attachments.

        `viewer_image` is the full-resolution rendition and `image` the feed thumbnail, so the
        first is preferred; a post with one attachment serves it directly under `media` while a
        gallery nests every item under `all_subattachments`. A video yields its permalink and
        never a file: logged out, the node carries no `playable_url` to fetch.
        """
        image_urls: list[str] = []
        video_urls: list[str] = []
        for attachment in story.get("attachments") or []:
            style = _deep_get(attachment, "styles", "attachment")
            nodes = _deep_get(style, "all_subattachments", "nodes")
            single = _deep_get(style, "media")
            items: list[JsonValue] = list(nodes) if isinstance(nodes, list) else []
            if single is not None:
                items.append({"media": single})
            for item in items:
                media = item.get("media") if isinstance(item, dict) else None
                if not isinstance(media, dict):
                    continue
                if media.get("__typename") == "Video":
                    permalink = media.get("permalink_url") or media.get("url")
                    if isinstance(permalink, str) and permalink:
                        video_urls.append(permalink)
                    continue
                rendition = media.get("viewer_image") or media.get("image") or {}
                uri = rendition.get("uri") if isinstance(rendition, dict) else None
                if isinstance(uri, str) and uri and uri not in image_urls:
                    image_urls.append(uri)
        return image_urls, video_urls

    @staticmethod
    def _comment_branches(
        *, payloads: list[Any], post_url: str, post_id: str
    ) -> list[list[FacebookOutput]]:
        """Every preloaded comment on THIS post, grouped into branches the way Threads groups replies.

        A reply names its own parent in `comment_direct_parent`, so it is threaded behind that
        comment rather than behind whichever one the page happened to serialise before it; a
        comment whose parent is absent from the page opens a branch of its own. What comes back
        is only what the page chose to preload — a handful — never the whole comment section.

        The scan is page-wide because the comments are not nested under the story node, which on
        a group feed means walking past the neighbouring posts' comments too; each comment's own
        id says which post it belongs to, and that is what keeps them out.

        The page serialises each comment two or three times, once fully and once as a stub for
        its reply expander, so the fullest version of each wins.
        """
        found: dict[str, FacebookOutput] = {}
        parents: dict[str, str] = {}
        for payload in payloads:
            for node in _walk(node=payload):
                if node.get("__typename") != "Comment":
                    continue
                owner_id, comment_id = _comment_ids_of(node=node)
                text = _text_of(value=node.get("body"))
                if not comment_id or not text or comment_id in found:
                    continue
                if owner_id and post_id and owner_id != post_id:
                    continue
                parent = node.get("comment_direct_parent")
                if isinstance(parent, dict):
                    parents[comment_id] = _comment_ids_of(node=parent)[1]
                found[comment_id] = FacebookOutput(
                    text=text,
                    url=post_url,
                    author_name=_str_of(value=_deep_get(node, "author", "name")),
                    author_icon_url=_str_of(
                        value=_deep_get(node, "author", "profile_picture", "uri")
                    ),
                    taken_at=_time_of(value=node.get("created_time")),
                    comment_id=comment_id,
                )
        branches: list[list[FacebookOutput]] = []
        index: dict[str, list[FacebookOutput]] = {}
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

    @staticmethod
    def _group_name_of(*, payloads: list[Any], group_id: str) -> str:
        """The name of the group the post sits in, empty for a page or profile post.

        Matched on the URL's own group id where there is one, since a group page also serialises
        the groups it recommends alongside the one being read.
        """
        fallback = ""
        for payload in payloads:
            for node in _walk(node=payload):
                if node.get("__typename") != "Group":
                    continue
                name = node.get("name")
                if not isinstance(name, str) or not name:
                    continue
                if group_id and str(node.get("id")) == group_id:
                    return name
                fallback = fallback or name
        return "" if group_id else fallback

    def parse_metadata(self, *, url: str) -> FacebookConversation:
        """Reads one public Facebook post and the comments the page preloaded with it.

        Named to match `ThreadsDownloader.parse_metadata` and `InstagramDownloader.parse_metadata`,
        and meaning the same on all three: parse the post and write nothing to disk. There is no
        `parse` counterpart here because nothing is downloaded — the images ride out as URLs.

        A share link names its post only through the redirect it answers with, so the id is
        read off where the fetch landed, exactly as `ThreadsDownloader.extract_post_data` does.
        A redirect to the login wall means the post is not public, which is a normal outcome
        rather than a failure and comes back as an empty conversation.

        Args:
            url: The Facebook post URL in any accepted form.

        Returns:
            The parsed conversation; its `chain` is empty when the post could not be read.

        Raises:
            LinkReadError: The page could not be fetched, in the shape `link_fetch_error`
                classified it as; `RuntimeError` for a failure HTTP does not classify.
        """
        facebook_url = FacebookURL(raw_url=url)
        fetched = self._fetch_page(url=facebook_url.clean_url)
        if fetched.is_login_wall:
            logfire.info(
                "A Facebook post is not public; treating it as unreadable",
                url=facebook_url.clean_url,
            )
            return FacebookConversation()
        landed = FacebookURL(raw_url=fetched.final_url)
        post_id = facebook_url.post_id or landed.post_id
        payloads = list(self._json_payloads(html=fetched.html))
        story = self._find_story(payloads=payloads, post_id=post_id)
        if story is None:
            logfire.info(
                "A Facebook page carried no post payload; treating it as unreadable",
                url=facebook_url.clean_url,
                html_length=len(fetched.html),
            )
            return FacebookConversation()

        content_story = _deep_get(story, "comet_sections", "content", "story")
        actors_value = _deep_get(content_story, "actors") or story.get("actors")
        actors = actors_value if isinstance(actors_value, list) else []
        actor = actors[0] if actors and isinstance(actors[0], dict) else {}
        message = _deep_get(
            content_story, "comet_sections", "message_container", "story", "message"
        )
        image_urls, video_urls = self._media_of(story=story)
        permalink = story.get("permalink_url")
        # The permalink the page reports is preferred over the caller's URL for the reason
        # `clean_url` exists: a pasted share link carries the tokens that name whoever shared it.
        post_url = permalink if isinstance(permalink, str) and permalink else landed.clean_url
        post = FacebookOutput(
            text=_text_of(value=message),
            url=post_url,
            author_name=_str_of(value=actor.get("name")),
            author_icon_url=_str_of(value=_deep_get(actor, "profile_picture", "uri")),
            group_name=self._group_name_of(
                payloads=payloads, group_id=facebook_url.group_id or landed.group_id
            ),
            image_urls=image_urls,
            video_urls=video_urls,
            like_count=_count_of(value=_deep_get(story, "feedback", "reaction_count")),
            comment_count=_comment_total(story=story),
            share_count=_count_of(value=_deep_get(story, "feedback", "share_count")),
            taken_at=_time_of(value=story.get("creation_time")),
        )
        return FacebookConversation(
            chain=[post],
            reply_branches=self._comment_branches(
                payloads=payloads, post_url=post_url, post_id=post_id
            ),
            selected_comment_id=facebook_url.comment_id,
        )


def _comment_total(*, story: dict[str, Any]) -> int:
    """The post's total comment count, which the page nests differently per surface."""
    for path in (
        ("feedback", "total_comment_count"),
        ("feedback", "comment_rendering_instance", "comments", "total_count"),
        ("comet_sections", "feedback", "story", "feedback_context", "total_comment_count"),
    ):
        value = _deep_get(story, *path)
        if isinstance(value, int):
            return value
    return 0


if __name__ == "__main__":
    """
    Keep this for self development and testing.
    DO NOT REMOVE THIS FOR ANY REASON.
    """
    from rich.console import Console

    console = Console()
    downloader = FacebookDownloader()
    metadata = downloader.parse_metadata(
        url="https://www.facebook.com/groups/1176671326743489/posts/1730774811333135/?comment_id=1730777104666239"
    )
    console.print(metadata)
