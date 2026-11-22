"""Instagram post URL parsing and page extraction.

Reads logged-out posts and their full comment lists from embedded JSON script blocks using
`BROWSER_HEADERS`. Downloads nothing — image and video URLs are returned directly.

Comment permalinks (`/p/<code>/c/<id>/`) return a shell without the post payload, so
`InstagramURL.clean_url` strips `/c/<id>/` and query tracking parameters (`stkn`) while
preserving `comment_id` separately.
"""

import re
from typing import Any
from functools import cached_property
from urllib.parse import urlparse

import logfire
from pydantic import Field, BaseModel, computed_field

from discordbot.utils.urls import URL_START_ANCHOR, PostLink
from discordbot.typings.timeouts import INSTAGRAM_PAGE_TIMEOUT_SECONDS
from discordbot.services.platforms.base import (
    LinkableComments,
    PlatformDownloader,
    LinkableCommentOutput,
    thread_branches,
)
from discordbot.services.platforms.page_json import (
    BROWSER_HEADERS,
    FetchedPage,
    walk,
    str_of,
    time_of,
    count_of,
    deep_get,
    fetch_page,
    json_payloads,
)

_CANONICAL_INSTAGRAM_ORIGIN = "https://www.instagram.com"

# Host-anchored (`is_instagram_post_url` filters non-posts); ASCII tail stops at CJK punctuation.
INSTAGRAM_URL_RE = re.compile(
    pattern=rf"{URL_START_ANCHOR}https?://(?:[a-z0-9-]+\.)*instagram\.com/"
    r"[A-Za-z0-9_.?=&%/~:+-]*[A-Za-z0-9_-]/?"
)

# Matches post paths (with optional author handle); excludes `/reels/audio/<id>/` sound pages.
_POST_PATH_RE = re.compile(
    pattern=r"^/(?:(?P<user>[A-Za-z0-9_.]+)/)?(?P<kind>p|reel|reels|tv)/(?!audio/)(?P<code>[A-Za-z0-9_-]+)"
)

_COMMENT_PATH_RE = re.compile(pattern=r"/c/(?P<comment>[0-9]+)")


_MEDIA_TYPE_VIDEO = 2

_LOGIN_WALL_PATHS = ("/accounts/login", "/challenge", "/accounts/suspended")


def is_instagram_post_url(url: str) -> bool:
    """Returns whether a matched Instagram URL names one post rather than a profile or home page.

    `INSTAGRAM_URL_RE` matches the host, so `/`, `/<user>/`, `/explore/...`, and other non-post
    paths are rejected here before any page fetch.

    Args:
        url (str): A URL `INSTAGRAM_URL_RE` matched.
    """
    return bool(InstagramURL(raw_url=url).shortcode)


INSTAGRAM_POST_LINK = PostLink(pattern=INSTAGRAM_URL_RE, accept=is_instagram_post_url)


class InstagramURL(BaseModel):
    """Parses and normalises an Instagram post URL; `shortcode` is empty for non-post URLs."""

    raw_url: str = Field(..., description="Original Instagram URL provided by the caller")

    @computed_field
    @cached_property
    def shortcode(self) -> str:
        """Returns the post's shortcode, or an empty string when the URL names no post."""
        match = _POST_PATH_RE.match(string=urlparse(self.raw_url).path)
        return match.group("code") if match else ""

    @computed_field
    @cached_property
    def comment_id(self) -> str:
        """Returns the comment id a `/c/<id>/` permalink singles out, or an empty string when absent."""
        match = _COMMENT_PATH_RE.search(string=urlparse(self.raw_url).path)
        return match.group("comment") if match else ""

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        """Returns the canonical `https://www.instagram.com/<kind>/<code>/` URL.

        Drops the author handle, `/c/<id>/` segment (unfetchable), and all query parameters
        including share-tracking `stkn`.
        """
        match = _POST_PATH_RE.match(string=urlparse(self.raw_url).path)
        if not match:
            return self.raw_url
        kind = "reel" if match.group("kind") in {"reel", "reels"} else match.group("kind")
        return f"{_CANONICAL_INSTAGRAM_ORIGIN}/{kind}/{match.group('code')}/"


class InstagramOutput(LinkableCommentOutput):
    """One Instagram post or comment.

    `url` is always the post permalink even on comments (comment permalinks are not fetchable),
    and comments carry no media or `comment_count`.
    """

    author_full_name: str = Field(
        default="", description="The author's display name, which comments do not carry"
    )


class InstagramConversation(LinkableComments[InstagramOutput]):
    """One Instagram post and its comment threads.

    `chain` has one element (no ancestors), `reply_branches` holds the full comment list rather
    than a preload, and `selected_comment_id` comes from `/c/<id>/`.
    """


class InstagramDownloader(PlatformDownloader):
    """Reads a public Instagram post and its comments from the page HTML."""

    def _fetch_page(self, url: str) -> FetchedPage:
        """Fetches a page with the browser headers the full payload needs."""
        return fetch_page(url=url, headers=BROWSER_HEADERS, timeout=INSTAGRAM_PAGE_TIMEOUT_SECONDS)

    @staticmethod
    def _find_media(payloads: list[Any], shortcode: str) -> dict[str, Any] | None:
        """Returns the media node matching `shortcode`, or `None` when absent."""
        # Match shortcode and prefer nodes with media fields over "more posts" rail stubs.
        fallback: dict[str, Any] | None = None
        for payload in payloads:
            for node in walk(node=payload):
                if node.get("code") != shortcode:
                    continue
                if "carousel_media" in node or "image_versions2" in node:
                    return node
                fallback = fallback or node
        return fallback

    @staticmethod
    def _media_urls_of(media: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Returns `(image_urls, video_urls)` in carousel order."""
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
                url = str_of(value=deep_get(first, "url"))
                if url:
                    video_urls.append(url)
                    continue
            # Candidates are ordered widest-first without dimensions; index 0 is the original.
            candidates = deep_get(item, "image_versions2", "candidates")
            first = candidates[0] if isinstance(candidates, list) and candidates else None
            url = str_of(value=deep_get(first, "url"))
            if url and url not in image_urls:
                image_urls.append(url)
        return image_urls, video_urls

    @staticmethod
    def _comment_branches(payloads: list[Any], post_url: str) -> list[list[InstagramOutput]]:
        """Returns every comment on the page, grouped into branches by `thread_branches`."""
        found: dict[str, InstagramOutput] = {}
        parents: dict[str, str] = {}
        for payload in payloads:
            for node in walk(node=payload):
                # Comment nodes are identified by carrying both `comment_like_count` and `text`.
                if "comment_like_count" not in node or "text" not in node:
                    continue
                comment_id = str(node.get("pk") or "")
                if not comment_id.isdecimal() or comment_id in found:
                    continue
                parent = str(node.get("parent_comment_id") or "")
                if parent.isdecimal():
                    parents[comment_id] = parent
                found[comment_id] = InstagramOutput(
                    text=str_of(value=node.get("text")),
                    url=post_url,
                    author_name=str_of(value=deep_get(node, "user", "username")),
                    author_icon_url=str_of(value=deep_get(node, "user", "profile_pic_url")),
                    like_count=count_of(value=node.get("comment_like_count")),
                    taken_at=time_of(value=node.get("created_at")),
                    comment_id=comment_id,
                )
        return thread_branches(comments=found, parents=parents)

    def parse_metadata(self, url: str) -> InstagramConversation:
        """Returns the parsed conversation for a public Instagram post, writing nothing to disk.

        Returns an empty `InstagramConversation` for non-post URLs, login walls, private accounts,
        or missing payloads.

        Args:
            url (str): The Instagram post URL in any accepted form.

        Raises:
            LinkReadError: If the page fetch fails with a classified HTTP/network error.
            RuntimeError: If the fetch fails in an unclassified way.
        """
        instagram_url = InstagramURL(raw_url=url)
        if not instagram_url.shortcode:
            logfire.info("An Instagram URL names no post; treating it as unreadable", url=url)
            return InstagramConversation()

        fetched = self._fetch_page(url=instagram_url.clean_url)
        if fetched.landed_on(path_prefixes=_LOGIN_WALL_PATHS):
            logfire.info(
                "An Instagram post is not public; treating it as unreadable",
                url=instagram_url.clean_url,
            )
            return InstagramConversation()

        payloads = list(json_payloads(html=fetched.html, platform="Instagram"))
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
            text=str_of(value=deep_get(media, "caption", "text")),
            url=instagram_url.clean_url,
            author_name=str_of(value=deep_get(media, "user", "username")),
            author_full_name=str_of(value=deep_get(media, "user", "full_name")),
            author_icon_url=str_of(value=deep_get(media, "user", "profile_pic_url")),
            image_urls=image_urls,
            video_urls=video_urls,
            like_count=count_of(value=media.get("like_count")),
            comment_count=count_of(value=media.get("comment_count")),
            taken_at=time_of(value=media.get("taken_at")),
        )
        return InstagramConversation(
            chain=[post],
            reply_branches=self._comment_branches(
                payloads=payloads, post_url=instagram_url.clean_url
            ),
            selected_comment_id=instagram_url.comment_id,
        )


if __name__ == "__main__":
    # Kept for local manual smoke-testing; do not remove.
    from rich.console import Console

    console = Console()
    downloader = InstagramDownloader()
    metadata = downloader.parse_metadata(
        url="https://www.instagram.com/p/Dc5eNjYkoZE/c/17946527169275440/?img_index=1"
    )
    console.print(metadata)
