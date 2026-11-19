"""Threads URL parsing, API models, and media download helpers."""

import re
import json
import time
from typing import Any
from pathlib import Path
from datetime import UTC, datetime
from functools import cached_property
from urllib.parse import urlparse

import logfire
from pydantic import (
    Field,
    BaseModel,
    ValidationInfo,
    ValidationError,
    computed_field,
    field_validator,
)
import requests
from pydantic_core.core_schema import ValidatorFunctionWrapHandler

from discordbot.utils.urls import URL_START_ANCHOR
from discordbot.typings.timeouts import (
    THREADS_PAGE_TIMEOUT_SECONDS,
    THREADS_MEDIA_READ_TIMEOUT_SECONDS,
    THREADS_EMPTY_PAGE_RETRY_DEADLINE_SECONDS,
)
from discordbot.utils.link_errors import (
    LinkRetryableError,
    LinkUnavailableError,
    is_retryable_fetch_failure,
)
from discordbot.services.platforms.base import (
    PlatformOutput,
    PlatformDownloader,
    PlatformConversation,
)
from discordbot.services.platforms.page_json import FetchedPage, walk, fetch_page
from discordbot.services.platforms.file_downloads import stream_to_file

# Matches `/@user/post/<code>` and `/share/<code>` on threads.net/com; ASCII tail stops at CJK.
THREADS_URL_RE = re.compile(
    pattern=rf"{URL_START_ANCHOR}https?://(?:www\.)?threads\.(?:net|com)/(?:@[^/]+/post|share)/"
    r"[A-Za-z0-9_.?=&%-]*[A-Za-z0-9_-]/?"
)

# Canonical post path only; `/share/<code>` carries an unrelated code resolved via redirect.
_POST_PATH_RE = re.compile(pattern=r"^/@[^/]+/post/([^/]+)/?$")

# Hosts rewritten to `www.threads.com` (the only host Threads serves without a 301).
_THREADS_HOSTS = frozenset({"threads.com", "www.threads.com", "threads.net", "www.threads.net"})
_CANONICAL_THREADS_ORIGIN = "https://www.threads.com"


class _ThreadsModel(BaseModel):
    """Base model coercing explicit JSON nulls on `str` fields to empty strings."""

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_null_string(cls, value: object, info: ValidationInfo) -> object:
        """Maps a null value to an empty string for str-typed fields."""
        field_name = info.field_name
        if value is None and field_name and cls.model_fields[field_name].annotation is str:
            return ""
        return value


class ThreadsURL(BaseModel):
    """Parses and normalises a Threads post URL; `post_code` is empty for `/share/<code>` links."""

    raw_url: str = Field(..., description="Original Threads URL provided by the caller")

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        """Returns the `www.threads.com` URL with query parameters and trailing slash stripped.

        Non-Threads hosts keep their original scheme and host.
        """
        parsed = urlparse(self.raw_url)
        path = parsed.path.rstrip("/")
        if parsed.netloc.lower() in _THREADS_HOSTS:
            return f"{_CANONICAL_THREADS_ORIGIN}{path}"
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    @computed_field
    @cached_property
    def post_code(self) -> str:
        """Returns the short code from `/@<user>/post/<code>`, or empty string for share links."""
        match = _POST_PATH_RE.match(string=urlparse(self.raw_url).path)
        return match.group(1) if match else ""


class User(_ThreadsModel):
    """Represents a Threads user."""

    username: str = Field(default="", description="Username handle")
    profile_pic_url: str = Field(default="", description="Profile picture URL")


class Caption(_ThreadsModel):
    """Represents caption text attached to a Threads post."""

    text: str = Field(default="", description="Caption text content")


class VideoVersion(_ThreadsModel):
    """Represents an available video rendition."""

    url: str = Field(default="", description="Video file URL")


class ImageCandidate(_ThreadsModel):
    """Represents an available image rendition."""

    url: str = Field(default="", description="Image URL")


class ImageVersions2(_ThreadsModel):
    """Holds available image renditions for a media object."""

    candidates: list[ImageCandidate] = Field(
        default_factory=list, description="Available image resolutions"
    )


class CarouselMedia(_ThreadsModel):
    """Represents one media item in a Threads carousel."""

    video_versions: list[VideoVersion] | None = Field(
        default=None, description="Available video versions"
    )
    image_versions2: ImageVersions2 | None = Field(
        default=None, description="Available image versions"
    )


class MediaContainer(_ThreadsModel):
    """Contains media fields shared by posts and linked inline media."""

    carousel_media: list[CarouselMedia] | None = Field(
        default=None, description="Carousel media items"
    )
    video_versions: list[VideoVersion] | None = Field(
        default=None, description="Available video versions"
    )
    image_versions2: ImageVersions2 | None = Field(
        default=None, description="Available image versions"
    )

    @property
    def media_urls(self) -> list[str]:
        """Returns the first media URL from each carousel item, or the first standalone video or image URL."""
        urls: list[str] = []
        if self.carousel_media:
            for item in self.carousel_media:
                if item.video_versions:
                    urls.append(item.video_versions[0].url)
                elif item.image_versions2 and item.image_versions2.candidates:
                    urls.append(item.image_versions2.candidates[0].url)
        elif self.video_versions:
            urls.append(self.video_versions[0].url)
        elif self.image_versions2 and self.image_versions2.candidates:
            urls.append(self.image_versions2.candidates[0].url)
        return [u for u in urls if u]


class Fragment(_ThreadsModel):
    """Represents one text fragment from Threads structured text."""

    plaintext: str = Field(default="", description="Plain text content of the fragment")


class TextFragments(_ThreadsModel):
    """Holds ordered structured text fragments."""

    fragments: list[Fragment] = Field(
        default_factory=list, description="Ordered list of text fragments"
    )


class LinkPreviewAttachment(_ThreadsModel):
    """Represents metadata for a link preview attachment."""

    title: str = Field(default="", description="Title shown in the link preview")
    image_url: str = Field(default="", description="Image shown in the link preview")
    url: str = Field(default="", description="Original link preview URL")


class LinkedInlineMedia(MediaContainer):
    """Represents media attached through a link preview."""

    code: str = Field(default="", description="Linked media short code")
    caption: Caption | None = Field(default=None, description="Linked media caption")


def _isolate(value: object, handler: ValidatorFunctionWrapHandler, message: str) -> object | None:
    """Validates `value`, or logs `message` and drops it to None when it fails the schema."""
    try:
        return handler(value)
    except ValidationError:
        logfire.warn(message, _exc_info=True)
        return None


class ShareInfo(_ThreadsModel):
    """What a post quotes or reposts.

    `quoted_post` is a full `Post` (forming a forward-reference cycle resolved by pydantic at
    import). `quoted_attachment_post` and `reposted_post` are intentionally omitted.
    """

    quoted_post: "Post | None" = Field(
        default=None, description="The post this one quotes, absent when it quotes nothing"
    )

    @field_validator("quoted_post", mode="wrap")
    @classmethod
    def _isolate_quoted_post(
        cls, value: object, handler: ValidatorFunctionWrapHandler
    ) -> object | None:
        """Drops an unparsable quoted post instead of failing validation of the enclosing post."""
        return _isolate(
            value=value,
            handler=handler,
            message="A quoted Threads post no longer matches the parser schema; dropping just it",
        )


class PostEdge(_ThreadsModel):
    """One edge of a connection of posts."""

    node: "Post | None" = Field(default=None, description="The post this edge carries")

    @field_validator("node", mode="wrap")
    @classmethod
    def _isolate_post(cls, value: object, handler: ValidatorFunctionWrapHandler) -> object | None:
        """Drops a post that fails validation instead of failing the connection holding it."""
        return _isolate(
            value=value,
            handler=handler,
            message="A Threads post no longer matches the parser schema; dropping just it",
        )


class PostConnection(_ThreadsModel):
    """A connection of posts, in the order the page serialised them."""

    edges: list[PostEdge] = Field(
        default_factory=list, description="The connection's edges, in page order"
    )

    @property
    def posts(self) -> list["Post"]:
        """Returns every edge's post in page order, with empty edges dropped."""
        return [edge.node for edge in self.edges if edge.node]


class PostThread(_ThreadsModel):
    """One thread of posts: the chain above the target, or one branch of replies below it."""

    posts: PostConnection | None = Field(
        default=None, description="The thread's posts, oldest first"
    )


class ThreadEdge(_ThreadsModel):
    """One edge of a connection of threads."""

    node: PostThread | None = Field(default=None, description="The thread this edge carries")

    @field_validator("node", mode="wrap")
    @classmethod
    def _isolate_thread(
        cls, value: object, handler: ValidatorFunctionWrapHandler
    ) -> object | None:
        """Drops a branch that fails validation instead of failing every branch beside it."""
        return _isolate(
            value=value,
            handler=handler,
            message="A Threads reply branch no longer matches the parser schema; dropping just it",
        )


class ThreadConnection(_ThreadsModel):
    """A connection of threads, in the order the page ranked them."""

    edges: list[ThreadEdge] = Field(
        default_factory=list, description="The connection's edges, in page order"
    )

    @property
    def threads(self) -> list[list["Post"]]:
        """Returns one post list per non-empty thread, ordered from the direct reply outward."""
        return [
            edge.node.posts.posts
            for edge in self.edges
            if edge.node and edge.node.posts and edge.node.posts.posts
        ]


class TextPostAppInfo(_ThreadsModel):
    """Represents Threads-specific post metadata and engagement fields."""

    direct_reply_count: int | None = Field(default=None, description="Number of direct replies")
    repost_count: int | None = Field(default=None, description="Number of reposts")
    quote_count: int | None = Field(default=None, description="Number of quote posts")
    reshare_count: int | None = Field(default=None, description="Total reshare count")
    text_fragments: TextFragments | None = Field(
        default=None, description="Structured text fragments with links/mentions"
    )
    link_preview_attachment: LinkPreviewAttachment | None = Field(
        default=None, description="Preview metadata for shared links"
    )
    linked_inline_media: LinkedInlineMedia | None = Field(
        default=None, description="Inline media attached through a link preview"
    )
    # Nullable because Threads sends explicit null and `_ThreadsModel` only coerces `str` fields.
    is_post_unavailable: bool | None = Field(
        default=None, description="True when the post this info belongs to is deleted or private"
    )
    reply_to_author: User | None = Field(
        default=None, description="User this post is directly replying to, if any"
    )
    share_info: ShareInfo | None = Field(
        default=None, description="What this post quotes or reposts"
    )
    # Arrive on separate SJS fragments joined by `Post.id` in `_thread_around`.
    containing_thread: PostThread | None = Field(
        default=None, description="The posts above this one in its thread, this one excluded"
    )
    direct_replies: ThreadConnection | None = Field(
        default=None, description="One thread per branch of replies under this post"
    )


class Post(MediaContainer):
    """Represents a single Threads post parsed from the API JSON."""

    # Shared join key across split SJS fragments for one post.
    id: str = Field(default="", description="Media id, `<post_pk>_<author_pk>`")
    code: str = Field(default="", description="Post short code used in URLs")
    caption: Caption | None = Field(default=None, description="Post caption")
    user: User | None = Field(default=None, description="Post author")
    text_post_app_info: TextPostAppInfo | None = Field(
        default=None, description="Threads-specific post info and engagement metrics"
    )
    like_count: int | None = Field(default=None, description="Number of likes")
    taken_at: int | None = Field(default=None, description="Post creation timestamp (Unix epoch)")

    @property
    def caption_text(self) -> str:
        """Returns structured text fragments, caption text, link preview title, or an empty string in that priority order."""
        if self.text_post_app_info and self.text_post_app_info.text_fragments:
            fragments_text = "".join(
                f.plaintext for f in self.text_post_app_info.text_fragments.fragments
            )
            if fragments_text:
                return fragments_text
        if self.caption and self.caption.text:
            return self.caption.text
        if self.text_post_app_info and self.text_post_app_info.link_preview_attachment:
            title = self.text_post_app_info.link_preview_attachment.title
            if title:
                return title
        return ""

    @property
    def author_name(self) -> str:
        """Returns the author username, or an empty string when author data is missing."""
        return self.user.username if self.user else ""

    @property
    def author_icon_url(self) -> str:
        """Returns the author profile picture URL, or an empty string when author data is missing."""
        return self.user.profile_pic_url if self.user else ""

    @property
    def reply_count(self) -> int:
        """Returns the direct reply count, or `0` when engagement data is missing."""
        return (self.text_post_app_info.direct_reply_count or 0) if self.text_post_app_info else 0

    @property
    def repost_count(self) -> int:
        """Returns the repost count, or `0` when engagement data is missing."""
        return (self.text_post_app_info.repost_count or 0) if self.text_post_app_info else 0

    @property
    def quote_count(self) -> int:
        """Returns the quote post count, or `0` when engagement data is missing."""
        return (self.text_post_app_info.quote_count or 0) if self.text_post_app_info else 0

    @property
    def reshare_count(self) -> int:
        """Returns the total reshare count, or `0` when engagement data is missing."""
        return (self.text_post_app_info.reshare_count or 0) if self.text_post_app_info else 0

    @property
    def reply_to_username(self) -> str:
        """Returns the username from `text_post_app_info.reply_to_author`, or an empty string when absent."""
        if self.text_post_app_info and self.text_post_app_info.reply_to_author:
            return self.text_post_app_info.reply_to_author.username
        return ""

    @property
    def quoted_post(self) -> "Post | None":
        """Returns the quoted `Post` payload, or `None` when this post quotes nothing."""
        if self.text_post_app_info and self.text_post_app_info.share_info:
            return self.text_post_app_info.share_info.quoted_post
        return None

    @property
    def is_unavailable(self) -> bool:
        """Returns whether Threads flags this post as deleted or private.

        Unavailable quoted posts arrive as tombstones carrying only `id`, `pk`, and this flag;
        `share_info.quoted_attachment_post_unavailable` is unreliable and unused.
        """
        return bool(self.text_post_app_info and self.text_post_app_info.is_post_unavailable)

    @property
    def is_readable(self) -> bool:
        """Returns whether this post is available and carries an author, code, text, or media.

        Broader than `PlatformOutput.is_readable`: honours `is_unavailable` and accepts a bare
        author or shortcode as having a permalink to link to.
        """
        if self.is_unavailable:
            return False
        return bool(self.author_name or self.code or self.caption_text or self.media_urls)

    @property
    def media_urls(self) -> list[str]:
        """Returns deduplicated non-empty media URLs from post media, linked inline media, or the link preview image fallback."""
        urls = super().media_urls
        app_info = self.text_post_app_info
        if app_info and app_info.linked_inline_media:
            urls.extend(app_info.linked_inline_media.media_urls)
        if not urls and app_info and app_info.link_preview_attachment:
            urls.append(app_info.link_preview_attachment.image_url)
        return [u for u in dict.fromkeys(urls) if u]


class ThreadsPage(BaseModel):
    """Raw target chain, reply branches, and canonical URL joined from one Threads page."""

    chain: list[Post] = Field(
        default_factory=list,
        description="The chain ending at the target, ordered `[root, ..., parent, target]`; empty when the page carried no such post",
    )
    reply_branches: list[list[Post]] = Field(
        default_factory=list,
        description="One list per reply branch under the target, each ordered from the direct reply outward, so an item's index in its branch is its nesting depth",
    )
    url: str = Field(
        default="",
        description="The canonical post URL the page was read from, query stripped; for a share link, where its redirect landed. Empty when a share link led to no post",
    )


class ParsedPage(BaseModel):
    """Parsed page output paired with whether any SJS block held post JSON.

    `carried_post_json=False` indicates a soft-throttled shell rather than a real miss, unless
    `_served_route` already matched a known post-refusal route.
    """

    page: ThreadsPage = Field(
        ..., description="The posts this page yielded; empty when it held no such post"
    )
    carried_post_json: bool = Field(
        ...,
        description="Whether any script block on the page held post JSON at all",
        examples=[True],
    )


class ThreadsOutput(PlatformOutput):
    """Output model for a single Threads post."""

    video_paths: list[Path] = Field(
        default_factory=list, description="Local paths of downloaded videos"
    )
    reply_to_username: str = Field(
        default="", description="Username this post replies to, empty when it replies to nobody"
    )
    repost_count: int = Field(default=0, description="Number of reposts")
    quote_count: int = Field(default=0, description="Number of quote posts")
    share_count: int = Field(
        default=0, description="Total reshare count, the same concept Facebook counts as shares"
    )
    # Quoted post media stays URL-only (`video_paths` is always empty).
    quoted: "ThreadsOutput | None" = Field(
        default=None,
        description="The post this one quotes, absent when it quotes nothing readable",
    )
    quoted_unavailable: bool = Field(
        default=False,
        description="Whether this post quotes a post Threads reports as deleted or private",
        examples=[False],
    )

    def counters_line(self) -> str:
        """Returns the five counters Threads publishes formatted as one line."""
        return (
            f"❤️ {self.like_count:,} | 💬 {self.comment_count:,} | 🔁 {self.repost_count:,} | "
            f"🔗 {self.quote_count:,} | ↗️ {self.share_count:,}"
        )


class ThreadsConversation(PlatformConversation[ThreadsOutput]):
    """A parsed Threads post, its ancestor `chain`, and `reply_branches`.

    `selected_comment_id` is always empty; every Threads reply has its own post URL.
    """


_SJS_PATTERN = re.compile(
    pattern=r'<script type="application/json"[^>]*data-sjs>(.*?)</script>', flags=re.DOTALL
)

# Present on every real Threads response (including invalid-post feed redirects) and absent on
# soft-throttle shells.
_POST_PAYLOAD_MARKER = "text_post_app_info"

# Non-localized route name rendered by Threads, used to distinguish post refusals from throttles.
_SERVED_ROUTE_RE = re.compile(pattern=r'"canonicalRouteName":"(?P<route>[^"]+)"')

# Matched explicitly: geo-block shells lack `_POST_PAYLOAD_MARKER` just like soft throttles.
_GEO_BLOCK_ROUTE = "comet.barcelonawebloggedout.BarcelonaGeoBlockRoute"

# Retry count for soft-throttled shells carrying no post JSON; real misses return immediately.
THREADS_EMPTY_PAGE_RETRIES = 2

# Short pause between soft-throttle retries on the reply critical path.
THREADS_EMPTY_PAGE_RETRY_DELAY_SECONDS = 0.8


class ThreadsDownloader(PlatformDownloader):
    """A downloader for extracting text and media from Threads posts."""

    output_folder: str = Field(
        ..., description="Directory where downloaded media files are written"
    )

    def _fetch_page(self, url: str) -> FetchedPage:
        """Fetches `url`, returning its HTML and final post-redirect URL."""
        return fetch_page(
            url=url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"},
            timeout=THREADS_PAGE_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _collect_fragments(data: dict[str, Any] | list[Any], post_code: str) -> list[Post]:
        """Returns all valid `Post` fragments in one SJS payload, skipping malformed ones."""
        fragments: list[Post] = []
        for node in walk(node=data):
            media = node.get("media")
            if not isinstance(media, dict):
                continue
            try:
                fragments.append(Post.model_validate(obj=media))
            except ValidationError:
                logfire.warn(
                    "Threads payload no longer matches the parser schema; skipping one fragment",
                    post_code=post_code,
                    _exc_info=True,
                )
        return fragments

    @staticmethod
    def _thread_around(
        fragments: list[Post], media_id: str
    ) -> tuple[list[Post], list[list[Post]]]:
        """Returns `(ancestors, reply_branches)` joined by `media_id` across `fragments`."""
        ancestors: list[Post] = []
        branches: list[list[Post]] = []
        if not media_id:
            return ancestors, branches
        # Keep first non-empty fragment; later fragments for the same post may carry empty lists.
        for fragment in fragments:
            info = fragment.text_post_app_info
            if fragment.id != media_id or not info:
                continue
            if not ancestors and info.containing_thread and info.containing_thread.posts:
                ancestors = info.containing_thread.posts.posts
            if not branches and info.direct_replies:
                branches = info.direct_replies.threads
        return ancestors, branches

    @staticmethod
    def _served_route(html: str) -> str:
        """Returns the route Threads rendered this page with, or an empty string when absent."""
        match = _SERVED_ROUTE_RE.search(string=html)
        return match.group("route") if match else ""

    def _parse_page_from_html(self, html: str, post_code: str) -> ParsedPage:
        """Parses the target post, its ancestors, and its replies from the SJS script tags."""
        carried_post_json = False
        fragments: list[Post] = []
        for match in _SJS_PATTERN.finditer(string=html):
            text = match.group(1)
            if _POST_PAYLOAD_MARKER not in text:
                continue

            try:
                data = json.loads(s=text)
            except json.JSONDecodeError:
                logfire.debug(
                    "Skipped a non-JSON Threads SJS block", post_code=post_code, _exc_info=True
                )
                continue
            except ValueError:
                # `json.loads` also raises plain `ValueError` (e.g. int-string digit limit).
                logfire.warn(
                    "Skipped an unparsable Threads SJS block", post_code=post_code, _exc_info=True
                )
                continue

            # Set only after JSON parses so truncated blocks still trigger soft-throttle retry.
            carried_post_json = True
            fragments.extend(self._collect_fragments(data=data, post_code=post_code))

        # Scan all blocks first; ancestor and reply fragments may precede or follow the target.
        target = next((fragment for fragment in fragments if fragment.code == post_code), None)
        if not target:
            return ParsedPage(page=ThreadsPage(), carried_post_json=carried_post_json)
        ancestors, branches = self._thread_around(fragments=fragments, media_id=target.id)
        return ParsedPage(
            page=ThreadsPage(chain=[*ancestors, target], reply_branches=branches),
            carried_post_json=True,
        )

    @staticmethod
    def _is_video(media_url: str) -> bool:
        """Returns whether `media_url` looks like a video rather than an image."""
        path_lower = urlparse(media_url).path.lower()
        if any(extension in path_lower for extension in (".jpg", ".jpeg", ".webp", ".png")):
            return False
        return ".mp4" in path_lower or "video" in media_url or "mp4" in media_url

    def download_media(self, url: str, filename: str) -> Path:
        """Downloads media from `url` into `output_folder/filename` and returns its path.

        `output_folder` must already exist; removing it aborts an in-flight download via
        `stream_to_file`.

        Args:
            url (str): The URL of the media to download.
            filename (str): The name to save the file as.

        Returns:
            The `Path` to the downloaded file.

        Raises:
            LinkRetryableError: If the CDN refused, timed out, or dropped the connection.
            RuntimeError: If the transfer failed in an unclassified way.
            OSError: If the file cannot be written, including `FileNotFoundError` when
                `output_folder` was removed mid-download.
        """
        headers = {"User-Agent": "Mozilla/5.0", "Referer": f"{_CANONICAL_THREADS_ORIGIN}/"}
        try:
            return stream_to_file(
                url=url,
                filepath=Path(self.output_folder) / filename,
                headers=headers,
                timeout=THREADS_MEDIA_READ_TIMEOUT_SECONDS,
            )
        # Classify CDN throttles/timeouts as retryable rather than unreadable.
        except requests.RequestException as error:
            message = f"Failed to download media from {url}: {error}"
            if is_retryable_fetch_failure(error=error):
                raise LinkRetryableError(message) from error
            raise RuntimeError(message) from error

    def extract_post_data(self, url: str) -> ThreadsPage:
        """Returns the target post, ancestors, and replies parsed from a Threads URL.

        Share links resolve their post code from the redirect target; a share link landing on a
        non-post URL returns an empty `ThreadsPage`. Soft-throttled shells carrying no post JSON
        are retried up to `THREADS_EMPTY_PAGE_RETRIES` within
        `THREADS_EMPTY_PAGE_RETRY_DEADLINE_SECONDS`; real misses return immediately.

        Args:
            url (str): The raw Threads post URL, canonical or share form.

        Raises:
            LinkUnavailableError: If Threads served an explicit refusal route (e.g. geo-block).
            LinkRetryableError: If every attempt returned a shell with no post JSON.
            LinkReadError: If a fetch failed with a classified HTTP/network error.
            RuntimeError: If a fetch failed in an unclassified way.
        """
        threads_url = ThreadsURL(raw_url=url)
        fetch_url = threads_url.clean_url
        post_code = threads_url.post_code
        deadline = time.monotonic() + THREADS_EMPTY_PAGE_RETRY_DEADLINE_SECONDS
        for attempt in range(THREADS_EMPTY_PAGE_RETRIES + 1):
            fetched = self._fetch_page(url=fetch_url)
            if not post_code:
                resolved = ThreadsURL(raw_url=fetched.final_url)
                post_code = resolved.post_code
                if not post_code:
                    logfire.info(
                        "A Threads share link did not lead to a post; treating it as unreadable",
                        url=threads_url.clean_url,
                        # Strip `?xmt=` share-redirect tracking token before logging.
                        final_url=resolved.clean_url,
                    )
                    return ThreadsPage()
                fetch_url = resolved.clean_url
            if self._served_route(html=fetched.html) == _GEO_BLOCK_ROUTE:
                # Short-circuit geo-block before parsing; retrying cannot clear it.
                logfire.info(
                    "Threads will not serve this post from here; reporting it unreadable",
                    post_code=post_code,
                )
                raise LinkUnavailableError(f"Threads refuses {post_code} from this region")
            parsed = self._parse_page_from_html(html=fetched.html, post_code=post_code)
            if parsed.page.chain or parsed.carried_post_json:
                return parsed.page.model_copy(update={"url": fetch_url})
            if attempt == THREADS_EMPTY_PAGE_RETRIES or time.monotonic() >= deadline:
                break
            logfire.info(
                "Threads answered without any post JSON; fetching the page again",
                post_code=post_code,
                attempt=attempt + 1,
                html_length=len(fetched.html),
            )
            time.sleep(THREADS_EMPTY_PAGE_RETRY_DELAY_SECONDS)
        # Exhausted retries with no post JSON indicates a soft throttle, not a missing post.
        logfire.warn(
            "Threads kept answering without any post JSON; treating it as a throttle",
            post_code=post_code,
            attempts=attempt + 1,
        )
        raise LinkRetryableError(
            f"Threads answered {attempt + 1} times with no post JSON for {post_code}"
        )

    def resolve_clean_url(self, url: str) -> str:
        """Returns the canonical post URL for `url` without parsing post JSON or downloading media.

        Canonical URLs are stripped locally without a network request; share links follow one
        redirect and drop per-share `?xmt=` tokens. Returns an empty string when a share link
        does not land on a post URL.

        Args:
            url (str): The Threads post URL, canonical or share form.

        Raises:
            LinkReadError: If fetching a share link fails with a classified HTTP/network error.
            RuntimeError: If the fetch fails in an unclassified way.
        """
        threads_url = ThreadsURL(raw_url=url)
        if threads_url.post_code:
            return threads_url.clean_url
        resolved = ThreadsURL(raw_url=self._fetch_page(url=threads_url.clean_url).final_url)
        if not resolved.post_code:
            logfire.info(
                "A Threads share link did not lead to a post; it has no clean URL to give",
                url=threads_url.clean_url,
                # Strip `?xmt=` share-tracking token before logging.
                final_url=resolved.clean_url,
            )
            return ""
        return resolved.clean_url

    @staticmethod
    def _post_url(post: Post) -> str:
        """Reconstructs a canonical Threads URL from a post's author handle and code."""
        username = post.author_name
        code = post.code
        if username and code:
            return f"{_CANONICAL_THREADS_ORIGIN}/@{username}/post/{code}"
        return ""

    def _build_output(
        self, post: Post, url: str, download: bool, include_quoted: bool = True
    ) -> ThreadsOutput:
        """Builds a `ThreadsOutput` from `post`, bounding quoted recursion to one level."""
        post_code = post.code or "unknown"
        image_urls: list[str] = []
        video_urls: list[str] = []
        video_paths: list[Path] = []

        for i, media_url in enumerate(post.media_urls):
            if self._is_video(media_url=media_url):
                video_urls.append(media_url)
                if download:
                    filename = f"threads_{post_code}_{i}.mp4"
                    video_paths.append(self.download_media(url=media_url, filename=filename))
            else:
                image_urls.append(media_url)

        taken_at = datetime.fromtimestamp(post.taken_at, tz=UTC) if post.taken_at else None

        # Threads degrades quote-of-a-quote payloads to username-only stubs, so stop at depth 1.
        quoted_post = post.quoted_post if include_quoted else None
        quoted = (
            self._build_output(
                post=quoted_post,
                url=self._post_url(post=quoted_post),
                download=False,
                include_quoted=False,
            )
            if quoted_post is not None and quoted_post.is_readable
            else None
        )

        return ThreadsOutput(
            text=post.caption_text,
            url=url,
            image_urls=image_urls,
            video_urls=video_urls,
            video_paths=video_paths,
            author_name=post.author_name,
            author_icon_url=post.author_icon_url,
            reply_to_username=post.reply_to_username,
            like_count=post.like_count or 0,
            comment_count=post.reply_count,
            repost_count=post.repost_count,
            quote_count=post.quote_count,
            share_count=post.reshare_count,
            taken_at=taken_at,
            quoted=quoted,
            quoted_unavailable=quoted_post is not None and quoted is None,
        )

    def _build_conversation(self, url: str, download: bool) -> ThreadsConversation:
        """Returns the `ThreadsConversation` for `url`, downloading target videos if `download`."""
        page = self.extract_post_data(url=url)
        if not page.chain:
            return ThreadsConversation()
        target_index = len(page.chain) - 1
        # Reconstruct canonical URLs (falling back to stripped `page.url`) so `?xmt=` never leaks.
        chain = [
            self._build_output(
                post=post,
                url=self._post_url(post=post) or (page.url if index == target_index else ""),
                download=download and index == target_index,
            )
            for index, post in enumerate(page.chain)
        ]
        reply_branches = [
            [
                self._build_output(post=reply, url=self._post_url(post=reply), download=False)
                for reply in branch
            ]
            for branch in page.reply_branches
        ]
        return ThreadsConversation(chain=chain, reply_branches=reply_branches)

    def parse(self, url: str) -> ThreadsConversation:
        """Returns the parsed conversation for `url`, downloading only the target post's videos.

        Writes target videos into `output_folder`, which the caller creates and cleans up.
        Returns an empty `ThreadsConversation` when no post is found.

        Args:
            url (str): The Threads post URL.
        """
        return self._build_conversation(url=url, download=True)

    def parse_metadata(self, url: str) -> ThreadsConversation:
        """Returns the parsed conversation for `url` without downloading any media.

        Returns an empty `ThreadsConversation` when no post is found.

        Args:
            url (str): The Threads post URL.
        """
        return self._build_conversation(url=url, download=False)


if __name__ == "__main__":
    # Kept for local manual smoke-testing; do not remove.
    from rich.console import Console

    console = Console()
    downloader = ThreadsDownloader(output_folder="./data")
    metadata = downloader.parse_metadata(url="https://www.threads.com/share/D1CytHQmC/")
    console.print(metadata)
