"""Facebook post URL parsing and page extraction.

Reads public posts from embedded GraphQL JSON blocks using `BROWSER_HEADERS`; mobile hosts
redirect to login and are normalized to `www.facebook.com`. Downloads nothing — image URLs and
video permalinks (logged-out `Video` nodes carry no `playable_url`) are returned directly.
"""

import re
import base64
from typing import Any
from functools import cached_property
from urllib.parse import parse_qs, urlparse, urlunparse

import logfire
from pydantic import Field, BaseModel, computed_field

from discordbot.utils.urls import URL_START_ANCHOR, host_matches_domain
from discordbot.typings.timeouts import FACEBOOK_PAGE_TIMEOUT_SECONDS
from discordbot.services.platforms.base import (
    LinkableComments,
    PlatformDownloader,
    LinkableCommentOutput,
    thread_branches,
)
from discordbot.services.platforms.page_json import (
    BROWSER_HEADERS,
    JsonValue,
    FetchedPage,
    walk,
    str_of,
    time_of,
    count_of,
    deep_get,
    fetch_page,
    json_payloads,
)

# `fb.watch` is excluded: logged-out redirects land on `/<page>/videos/<id>` with no post payload.
_FACEBOOK_DOMAINS = frozenset({"facebook.com", "fb.com"})
_CANONICAL_FACEBOOK_ORIGIN = "https://www.facebook.com"

# Host-anchored (`is_facebook_post_url` filters non-posts); excludes `/<page>/videos/` (no
# logged-out payload), with an ASCII tail ending on `[A-Za-z0-9_-]` to stop at CJK punctuation.
FACEBOOK_URL_RE = re.compile(
    pattern=rf"{URL_START_ANCHOR}https?://(?:[a-z0-9-]+\.)*(?:facebook\.com|fb\.com)/"
    r"(?![^/?\s]+/videos/)"
    r"[A-Za-z0-9_.?=&%/~:+-]*[A-Za-z0-9_-]/?"
)

# Post paths only; `share/v/` and `share/r/` carry no logged-out post payload.
_GROUP_POST_PATH_RE = re.compile(
    pattern=r"^/groups/(?P<group>[^/]+)/(?:posts|permalink)/(?P<post>[0-9]+)"
)
_PAGE_POST_PATH_RE = re.compile(pattern=r"^/(?:[^/]+)/(?:posts|videos)/(?P<post>[0-9]+)")
_NUMERIC_POST_PATH_RE = re.compile(pattern=r"^/(?:[0-9]+)/posts/(?P<post>[0-9]+)")
_SHARE_PATH_RE = re.compile(pattern=r"^/share/p/[A-Za-z0-9]+")

# Query params preserved by `clean_url`; share-tracking params (`rdid`, `share_url`) are stripped.
_COMMENT_ID_PARAM = "comment_id"
_POST_ID_PARAMS = ("story_fbid", "multi_permalinks", "fbid")

# Preserved by `clean_url` for `permalink.php?story_fbid=`, but excluded from `_POST_ID_PARAMS`
# so `profile.php?id=<n>` is not accepted as a post.
_OWNER_ID_PARAMS = ("id",)

_LOGIN_WALL_PATHS = ("/login", "/checkpoint", "/recover")


# Comment node ids decode as base64 `comment:<post_id>_<comment_id>`.
_COMMENT_ID_RE = re.compile(pattern=r"^comment:(?P<post>[0-9]+)_(?P<comment>[0-9]+)")


def is_facebook_post_url(url: str) -> bool:
    """Returns whether a matched Facebook URL names one post rather than a profile or a feed.

    `FACEBOOK_URL_RE` matches the host, so profile, group home, marketplace, and bare
    `facebook.com` links are rejected here.

    Args:
        url (str): A URL `FACEBOOK_URL_RE` matched.
    """
    facebook_url = FacebookURL(raw_url=url)
    return bool(facebook_url.post_id or facebook_url.is_share_link or facebook_url.is_pfbid_link)


class FacebookURL(BaseModel):
    """Parses and normalises a Facebook post URL; share and `pfbid` links leave `post_id` empty."""

    raw_url: str = Field(..., description="Original Facebook URL provided by the caller")

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        """Returns the `www.facebook.com` URL keeping only post-, owner-, and comment-id params.

        Strips per-share tracking parameters (`rdid`, `share_url`), making the URL safe to publish.
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
        """Returns the numeric post id the URL names, or an empty string for a share or `pfbid` link."""
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
        """Returns the group id the post sits in, or an empty string for a page or profile post."""
        match = _GROUP_POST_PATH_RE.match(string=urlparse(self.raw_url).path)
        if match:
            return match.group("group")
        feed = re.match(pattern=r"^/groups/(?P<group>[^/?]+)", string=urlparse(self.raw_url).path)
        return feed.group("group") if feed else ""

    @computed_field
    @cached_property
    def comment_id(self) -> str:
        """Returns the comment id the URL singles out, or an empty string when it names no comment."""
        values = parse_qs(urlparse(self.raw_url).query).get(_COMMENT_ID_PARAM)
        return values[0] if values and values[0].isdigit() else ""

    @computed_field
    @cached_property
    def is_share_link(self) -> bool:
        """Returns whether the URL is the post share form, which names its post only through the redirect."""
        return bool(_SHARE_PATH_RE.match(string=urlparse(self.raw_url).path))

    @computed_field
    @cached_property
    def is_pfbid_link(self) -> bool:
        """Returns whether the URL uses a `pfbid` post id (excludes group `multi_permalinks`)."""
        parsed = urlparse(self.raw_url)
        values = parse_qs(parsed.query).get("story_fbid")
        return bool(values and values[0].startswith("pfbid")) or bool(
            re.match(pattern=r"^/[^/]+/posts/pfbid[0-9A-Za-z]+", string=parsed.path)
        )


class FacebookOutput(LinkableCommentOutput):
    """One Facebook post or comment.

    `url` holds the post permalink even on comments, and `comment_count` is the total count
    reported by the post (which exceeds the preloaded comments). Comments carry no media, group,
    or counts.
    """

    group_name: str = Field(
        default="",
        description="The group the POST was made in, empty for a page post and on every comment",
    )
    share_count: int = Field(default=0, description="Shares the post reports; 0 on a comment")


class FacebookConversation(LinkableComments[FacebookOutput]):
    """One Facebook post and the preloaded discussion under it.

    `chain` has one element (no ancestors), `reply_branches` holds only the comments preloaded
    with the page, and `selected_comment_id` comes from `?comment_id=`.
    """


def _text_of(value: JsonValue) -> str:
    """Reads a `{"text": ...}` node's string, tolerating the null the page sometimes serves."""
    return str_of(value=value.get("text")) if isinstance(value, dict) else ""


def _comment_ids_of(node: dict[str, Any]) -> tuple[str, str]:
    """Returns `(post_id, comment_id)` from `legacy_fbid` and base64 node id, or empty strings."""
    legacy = node.get("legacy_fbid")
    comment_id = str(legacy) if isinstance(legacy, (int, str)) and str(legacy).isdigit() else ""
    node_id = node.get("id")
    if not isinstance(node_id, str):
        return "", comment_id
    try:
        # Facebook omits base64 padding; extra `=` are ignored when already aligned.
        decoded = base64.b64decode(node_id + "==").decode(encoding="utf-8", errors="replace")
    except ValueError:
        return "", comment_id
    match = _COMMENT_ID_RE.match(string=decoded)
    if match is None:
        return "", comment_id
    return match.group("post"), comment_id or match.group("comment")


class FacebookDownloader(PlatformDownloader):
    """Reads a public Facebook post and its preloaded comments from the page HTML."""

    def _fetch_page(self, url: str) -> FetchedPage:
        """Fetches a page with the browser headers Facebook will only answer in full to."""
        return fetch_page(url=url, headers=BROWSER_HEADERS, timeout=FACEBOOK_PAGE_TIMEOUT_SECONDS)

    @staticmethod
    def _find_story(payloads: list[Any], post_id: str) -> dict[str, Any] | None:
        """Returns the story node matching `post_id` (or the page's first story when empty)."""
        fallback: dict[str, Any] | None = None
        for payload in payloads:
            for node in walk(node=payload):
                if "post_id" not in node or "creation_time" not in node:
                    continue
                # Empty `post_id` occurs on `pfbid` links; take the page's first story.
                if post_id and str(node.get("post_id")) != post_id:
                    continue
                if deep_get(
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
        # Fall back to a message-less story node for image-only posts.
        return fallback

    @staticmethod
    def _media_of(story: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Returns `(image_urls, video_urls)` extracted from `story`'s attachments."""
        image_urls: list[str] = []
        video_urls: list[str] = []
        for attachment in story.get("attachments") or []:
            style = deep_get(attachment, "styles", "attachment")
            nodes = deep_get(style, "all_subattachments", "nodes")
            single = deep_get(style, "media")
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
                # Prefer full-resolution `viewer_image` over feed thumbnail `image`.
                rendition = media.get("viewer_image") or media.get("image") or {}
                uri = rendition.get("uri") if isinstance(rendition, dict) else None
                if isinstance(uri, str) and uri and uri not in image_urls:
                    image_urls.append(uri)
        return image_urls, video_urls

    @staticmethod
    def _comment_branches(
        payloads: list[Any], post_url: str, post_id: str
    ) -> list[list[FacebookOutput]]:
        """Returns preloaded comments scoped to `post_id`, grouped by `thread_branches`."""
        found: dict[str, FacebookOutput] = {}
        parents: dict[str, str] = {}
        for payload in payloads:
            for node in walk(node=payload):
                if node.get("__typename") != "Comment":
                    continue
                owner_id, comment_id = _comment_ids_of(node=node)
                text = _text_of(value=node.get("body"))
                if not comment_id or not text or comment_id in found:
                    continue
                # Scope page-wide scan to this post; group feeds include neighbouring comments.
                if owner_id and post_id and owner_id != post_id:
                    continue
                parent = node.get("comment_direct_parent")
                if isinstance(parent, dict):
                    parents[comment_id] = _comment_ids_of(node=parent)[1]
                found[comment_id] = FacebookOutput(
                    text=text,
                    url=post_url,
                    author_name=str_of(value=deep_get(node, "author", "name")),
                    author_icon_url=str_of(
                        value=deep_get(node, "author", "profile_picture", "uri")
                    ),
                    taken_at=time_of(value=node.get("created_time")),
                    comment_id=comment_id,
                )
        return thread_branches(comments=found, parents=parents)

    @staticmethod
    def _group_name_of(payloads: list[Any], group_id: str) -> str:
        """Returns the display name matching `group_id`, or an empty string when absent."""
        # Require an explicit `group_id`; pages also serialize recommended groups.
        if not group_id:
            return ""
        for payload in payloads:
            for node in walk(node=payload):
                if node.get("__typename") != "Group":
                    continue
                name = node.get("name")
                if not isinstance(name, str) or not name:
                    continue
                if str(node.get("id")) == group_id:
                    return name
        return ""

    def parse_metadata(self, url: str) -> FacebookConversation:
        """Returns the parsed conversation for a public Facebook post, writing nothing to disk.

        Returns an empty `FacebookConversation` on a login-wall redirect or missing story payload.

        Args:
            url (str): The Facebook post URL in any accepted form.

        Raises:
            LinkReadError: If the page fetch fails with a classified HTTP/network error.
            RuntimeError: If the fetch fails in an unclassified way.
        """
        facebook_url = FacebookURL(raw_url=url)
        fetched = self._fetch_page(url=facebook_url.clean_url)
        if fetched.landed_on(path_prefixes=_LOGIN_WALL_PATHS):
            logfire.info(
                "A Facebook post is not public; treating it as unreadable",
                url=facebook_url.clean_url,
            )
            return FacebookConversation()
        landed = FacebookURL(raw_url=fetched.final_url)
        post_id = facebook_url.post_id or landed.post_id
        payloads = list(json_payloads(html=fetched.html, platform="Facebook"))
        story = self._find_story(payloads=payloads, post_id=post_id)
        if story is None:
            logfire.info(
                "A Facebook page carried no post payload; treating it as unreadable",
                url=facebook_url.clean_url,
                html_length=len(fetched.html),
            )
            return FacebookConversation()

        content_story = deep_get(story, "comet_sections", "content", "story")
        actors_value = deep_get(content_story, "actors") or story.get("actors")
        actors = actors_value if isinstance(actors_value, list) else []
        actor = actors[0] if actors and isinstance(actors[0], dict) else {}
        message = deep_get(
            content_story, "comet_sections", "message_container", "story", "message"
        )
        image_urls, video_urls = self._media_of(story=story)
        permalink = story.get("permalink_url")
        # Prefer the page's permalink so per-share tracking tokens are never echoed.
        post_url = permalink if isinstance(permalink, str) and permalink else landed.clean_url
        post = FacebookOutput(
            text=_text_of(value=message),
            url=post_url,
            author_name=str_of(value=actor.get("name")),
            author_icon_url=str_of(value=deep_get(actor, "profile_picture", "uri")),
            group_name=self._group_name_of(
                payloads=payloads, group_id=facebook_url.group_id or landed.group_id
            ),
            image_urls=image_urls,
            video_urls=video_urls,
            # Raw integer, unlike locale-formatted `i18n_reaction_count` (e.g. "1.7萬").
            like_count=count_of(value=deep_get(story, "feedback", "reaction_count")),
            comment_count=_comment_total(story=story),
            share_count=count_of(value=deep_get(story, "feedback", "share_count")),
            taken_at=time_of(value=story.get("creation_time")),
        )
        return FacebookConversation(
            chain=[post],
            # A `pfbid` link names no id, so the story's own keeps other stories' comments out.
            reply_branches=self._comment_branches(
                payloads=payloads,
                post_url=post_url,
                post_id=post_id or str(story.get("post_id") or ""),
            ),
            selected_comment_id=facebook_url.comment_id,
        )


def _comment_total(story: dict[str, Any]) -> int:
    """Returns the post's total comment count across Facebook's feedback paths."""
    for path in (
        ("feedback", "total_comment_count"),
        ("feedback", "comment_rendering_instance", "comments", "total_count"),
        ("comet_sections", "feedback", "story", "feedback_context", "total_comment_count"),
    ):
        value = deep_get(story, *path)
        if isinstance(value, int):
            return value
    return 0


if __name__ == "__main__":
    # Kept for local manual smoke-testing; do not remove.
    from rich.console import Console

    console = Console()
    downloader = FacebookDownloader()
    metadata = downloader.parse_metadata(
        url="https://www.facebook.com/groups/1176671326743489/posts/1730774811333135/?comment_id=1730777104666239"
    )
    console.print(metadata)
