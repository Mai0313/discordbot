"""Threads URL parsing, API models, and media download helpers."""

import re
import json
from typing import TYPE_CHECKING
import asyncio
from pathlib import Path
from datetime import UTC, datetime
from functools import cached_property
import contextlib
from collections import OrderedDict
from urllib.parse import urlparse
from collections.abc import Iterator

from pydantic import (
    Field,
    BaseModel,
    ConfigDict,
    PrivateAttr,
    ValidationInfo,
    computed_field,
    field_validator,
)
import requests

if TYPE_CHECKING:
    from nextcord import Message


# Shared by the parse_threads cog (auto-expansion) and the reply pipeline (so it can wait
# for that expansion); one definition keeps both sides matching the same post-URL shape.
THREADS_URL_RE = re.compile(r"https?://(?:www\.)?threads\.(?:net|com)/@[^/]+/post/[^\s\"'<>)]+")


class _ThreadsModel(BaseModel):
    """Base model tolerating an explicit JSON null on plain string fields.

    Threads sometimes serialises an optional string field (e.g. a link
    preview's image_url) as an explicit null instead of omitting it. Those
    fields are declared as str with an empty default, so the null would raise a
    ValidationError, which the parser treats as a corrupt block and silently
    drops the whole post. Coercing null to the empty string keeps the default
    semantics for both absent and null values.
    """

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_null_string(cls, value: object, info: ValidationInfo) -> object:
        """Maps a null value to an empty string for str-typed fields."""
        field_name = info.field_name
        if value is None and field_name and cls.model_fields[field_name].annotation is str:
            return ""
        return value


class ThreadsURL(BaseModel):
    """Parses and normalises a Threads post URL.

    Attributes:
        raw_url: Original Threads URL provided by the caller.
    """

    raw_url: str = Field(..., description="Original Threads URL provided by the caller")

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        """The cleaned and normalised URL.

        Returns:
            URL with `threads.com` hosts normalised to `www.threads.net` and
            query parameters removed.
        """
        parsed = urlparse(self.raw_url)
        netloc = parsed.netloc
        if netloc in ("www.threads.com", "threads.com"):
            netloc = "www.threads.net"
        return f"{parsed.scheme}://{netloc}{parsed.path}"

    @computed_field
    @cached_property
    def post_code(self) -> str:
        """The post short code extracted from the URL.

        Returns:
            The last path segment from the raw URL, or an empty string for an
            empty path.
        """
        parsed = urlparse(self.raw_url)
        path_parts = parsed.path.strip("/").split("/")
        return path_parts[-1] if path_parts else ""


class User(_ThreadsModel):
    """Represents a Threads user.

    Attributes:
        username: User handle.
        profile_pic_url: Profile picture URL.
    """

    username: str = Field(default="", description="Username handle")
    profile_pic_url: str = Field(default="", description="Profile picture URL")


class Caption(_ThreadsModel):
    """Represents caption text attached to a Threads post.

    Attributes:
        text: Caption text content.
    """

    text: str = Field(default="", description="Caption text content")


class VideoVersion(_ThreadsModel):
    """Represents an available video rendition.

    Attributes:
        url: Video file URL.
    """

    url: str = Field(default="", description="Video file URL")


class ImageCandidate(_ThreadsModel):
    """Represents an available image rendition.

    Attributes:
        url: Image URL.
    """

    url: str = Field(default="", description="Image URL")


class ImageVersions2(_ThreadsModel):
    """Holds available image renditions for a media object.

    Attributes:
        candidates: Available image resolutions.
    """

    candidates: list[ImageCandidate] = Field(
        default_factory=list, description="Available image resolutions"
    )


class CarouselMedia(_ThreadsModel):
    """Represents one media item in a Threads carousel.

    Attributes:
        video_versions: Available video renditions.
        image_versions2: Available image renditions.
    """

    video_versions: list[VideoVersion] | None = Field(
        default=None, description="Available video versions"
    )
    image_versions2: ImageVersions2 | None = Field(
        default=None, description="Available image versions"
    )


class MediaContainer(_ThreadsModel):
    """Contains media fields shared by posts and linked inline media.

    Attributes:
        carousel_media: Carousel media items.
        video_versions: Available video renditions.
        image_versions2: Available image renditions.
    """

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
        """The list of media URLs extracted from the container.

        Returns:
            First media URL from each carousel item, or the first standalone
            video or image URL, with empty values removed.
        """
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
    """Represents one text fragment from Threads structured text.

    Attributes:
        plaintext: Plain text content of the fragment.
    """

    plaintext: str = Field(default="", description="Plain text content of the fragment")


class TextFragments(_ThreadsModel):
    """Holds ordered structured text fragments.

    Attributes:
        fragments: Ordered list of text fragments.
    """

    fragments: list[Fragment] = Field(
        default_factory=list, description="Ordered list of text fragments"
    )


class LinkPreviewAttachment(_ThreadsModel):
    """Represents metadata for a link preview attachment.

    Attributes:
        title: Title shown in the link preview.
        image_url: Image shown in the link preview.
        url: Original link preview URL.
    """

    title: str = Field(default="", description="Title shown in the link preview")
    image_url: str = Field(default="", description="Image shown in the link preview")
    url: str = Field(default="", description="Original link preview URL")


class LinkedInlineMedia(MediaContainer):
    """Represents media attached through a link preview.

    Attributes:
        code: Linked media short code.
        caption: Linked media caption.
    """

    code: str = Field(default="", description="Linked media short code")
    caption: Caption | None = Field(default=None, description="Linked media caption")


class TextPostAppInfo(_ThreadsModel):
    """Represents Threads-specific post metadata and engagement fields.

    Attributes:
        direct_reply_count: Number of direct replies.
        repost_count: Number of reposts.
        quote_count: Number of quote posts.
        reshare_count: Total reshare count.
        text_fragments: Structured text fragments with links or mentions.
        link_preview_attachment: Preview metadata for shared links.
        linked_inline_media: Inline media attached through a link preview.
        is_reply: Whether this post is a reply to another post.
        reply_to_author: User this post is directly replying to, if any.
    """

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
    is_reply: bool | None = Field(
        default=None, description="True when this post is a reply to another post"
    )
    reply_to_author: User | None = Field(
        default=None, description="User this post is directly replying to"
    )


class Post(MediaContainer):
    """Represents a single Threads post parsed from the API JSON.

    Attributes:
        code: Post short code used in URLs.
        caption: Post caption.
        user: Post author.
        text_post_app_info: Threads-specific post info and engagement metrics.
        like_count: Number of likes.
        taken_at: Post creation timestamp as a Unix epoch.
    """

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
        """The extracted caption text or fallback link preview title.

        Returns:
            Structured text fragments, caption text, link preview title, or an
            empty string in that priority order.
        """
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
        """The username of the post author.

        Returns:
            Author username, or an empty string when author data is missing.
        """
        return self.user.username if self.user else ""

    @property
    def author_icon_url(self) -> str:
        """The profile picture URL of the post author.

        Returns:
            Author profile picture URL, or an empty string when author data is
            missing.
        """
        return self.user.profile_pic_url if self.user else ""

    @property
    def reply_count(self) -> int:
        """The number of direct replies to the post.

        Returns:
            Direct reply count, or 0 when engagement data is missing.
        """
        return (self.text_post_app_info.direct_reply_count or 0) if self.text_post_app_info else 0

    @property
    def repost_count(self) -> int:
        """The number of reposts.

        Returns:
            Repost count, or 0 when engagement data is missing.
        """
        return (self.text_post_app_info.repost_count or 0) if self.text_post_app_info else 0

    @property
    def quote_count(self) -> int:
        """The number of quote posts.

        Returns:
            Quote post count, or 0 when engagement data is missing.
        """
        return (self.text_post_app_info.quote_count or 0) if self.text_post_app_info else 0

    @property
    def reshare_count(self) -> int:
        """The total reshare count.

        Returns:
            Reshare count, or 0 when engagement data is missing.
        """
        return (self.text_post_app_info.reshare_count or 0) if self.text_post_app_info else 0

    @property
    def is_reply(self) -> bool:
        """Whether this post is a reply to another post.

        Returns:
            True when `text_post_app_info.is_reply` is set; False otherwise.
        """
        return bool(self.text_post_app_info and self.text_post_app_info.is_reply)

    @property
    def reply_to_username(self) -> str:
        """The username of the post being directly replied to.

        Returns:
            Username from `text_post_app_info.reply_to_author`, or an empty
            string when this post is not a reply or the field is missing.
        """
        if self.text_post_app_info and self.text_post_app_info.reply_to_author:
            return self.text_post_app_info.reply_to_author.username
        return ""

    @property
    def media_urls(self) -> list[str]:
        """The list of media URLs, including inline media and link preview images.

        Returns:
            Deduplicated non-empty media URLs from post media, linked inline
            media, or the link preview image fallback.
        """
        urls = super().media_urls
        app_info = self.text_post_app_info
        if app_info and app_info.linked_inline_media:
            urls.extend(app_info.linked_inline_media.media_urls)
        if not urls and app_info and app_info.link_preview_attachment:
            urls.append(app_info.link_preview_attachment.image_url)
        return [u for u in dict.fromkeys(urls) if u]


class ThreadItem(_ThreadsModel):
    """Represents one item in a Threads reply chain.

    Attributes:
        post: Parsed post for this thread item.
    """

    post: Post | None = Field(default=None, description="Parsed post for this thread item")


class ThreadData(_ThreadsModel):
    """Represents Threads reply-chain data extracted from HTML JSON.

    Attributes:
        thread_items: Ordered thread items from the embedded JSON.
    """

    thread_items: list[ThreadItem] = Field(
        default_factory=list, description="Ordered thread items from the embedded JSON"
    )

    def find_post_with_parents(self, post_code: str) -> tuple[Post | None, list[Post]]:
        """Returns the matching post and the chronologically-ordered ancestors before it.

        Threads stores an entire reply chain (root → direct parent → target) in a single
        `thread_items` list, oldest first. Everything appearing before the target item is
        therefore an ancestor of it.

        Args:
            post_code: The short code of the target post.

        Returns:
            A tuple containing:
                - The matching Post instance if found, else None.
                - A list of ancestor Post instances, ordered oldest to newest.
        """
        for index, item in enumerate(self.thread_items):
            if item.post and item.post.code == post_code:
                parents = [t.post for t in self.thread_items[:index] if t.post]
                return item.post, parents
        return None, []


class ThreadsOutput(BaseModel):
    """Output model for a single Threads post.

    Attributes:
        text: Extracted post text.
        url: Source Threads URL.
        image_urls: Image URLs extracted from the post.
        video_urls: Video URLs extracted from the post.
        video_paths: Local paths of downloaded videos.
        author_name: Post author username.
        author_icon_url: Post author profile picture URL.
        like_count: Number of likes.
        reply_count: Number of direct replies.
        repost_count: Number of reposts.
        quote_count: Number of quote posts.
        reshare_count: Total reshare count.
        taken_at: Post creation time.
    """

    text: str = Field(default="", description="Extracted post text")
    url: str = Field(default="", description="Source Threads URL")
    image_urls: list[str] = Field(
        default_factory=list, description="Image URLs extracted from the post"
    )
    video_urls: list[str] = Field(
        default_factory=list, description="Video URLs extracted from the post"
    )
    video_paths: list[Path] = Field(
        default_factory=list, description="Local paths of downloaded videos"
    )
    author_name: str = Field(default="", description="Post author username")
    author_icon_url: str = Field(default="", description="Post author profile picture URL")
    like_count: int = Field(default=0, description="Number of likes")
    reply_count: int = Field(default=0, description="Number of direct replies")
    repost_count: int = Field(default=0, description="Number of reposts")
    quote_count: int = Field(default=0, description="Number of quote posts")
    reshare_count: int = Field(default=0, description="Total reshare count")
    taken_at: datetime | None = Field(default=None, description="Post creation time")

    def unlink(self) -> None:
        """Deletes downloaded video files for this post."""
        for path in self.video_paths:
            path.unlink(missing_ok=True)


_SJS_PATTERN = re.compile(
    r'<script type="application/json"[^>]*data-sjs>(.*?)</script>', re.DOTALL
)


class ThreadsDownloader(BaseModel):
    """A downloader for extracting text and media from Threads.net posts.

    Attributes:
        output_folder: Directory where downloaded media files are written.
    """

    output_folder: str = Field(
        ..., description="Directory where downloaded media files are written"
    )

    def _fetch_html(self, url: str) -> str:
        """Fetches the HTML content of the given URL."""
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
        try:
            response = requests.get(url=url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch HTML from {url}: {e}") from e

    @staticmethod
    def _find_keys(obj: dict | list | str | float | None, key: str) -> list:
        """Recursively searches for all occurrences of a key in a JSON-like object."""
        results: list = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key and isinstance(v, list | dict):
                    results.append(v)
                else:
                    results.extend(ThreadsDownloader._find_keys(obj=v, key=key))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(ThreadsDownloader._find_keys(obj=item, key=key))
        return results

    def _parse_post_from_html(self, html: str, post_code: str) -> tuple[Post | None, list[Post]]:
        """Parses the post and its parents from the SJS script tags in the HTML."""
        for match in _SJS_PATTERN.finditer(string=html):
            text = match.group(1)
            if "thread_items" not in text:
                continue

            try:
                data = json.loads(s=text)
                raw_lists = self._find_keys(obj=data, key="thread_items")
                for raw_items in raw_lists:
                    if not isinstance(raw_items, list):
                        continue
                    thread_data = ThreadData(thread_items=raw_items)
                    post, parents = thread_data.find_post_with_parents(post_code=post_code)
                    if post:
                        return post, parents
            except (json.JSONDecodeError, ValueError):
                continue

        return None, []

    @staticmethod
    def _determine_extension(media_url: str) -> str:
        """Determines the file extension from a media URL."""
        path_lower = urlparse(media_url).path.lower()
        if ".jpg" in path_lower or ".jpeg" in path_lower:
            return "jpg"
        if ".webp" in path_lower:
            return "webp"
        if ".png" in path_lower:
            return "png"
        if ".mp4" in path_lower:
            return "mp4"
        if "video" not in media_url and "mp4" not in media_url:
            return "jpg"
        return "mp4"

    def download_media(self, url: str, filename: str) -> Path | None:
        """Downloads media from the given URL to the output folder.

        Args:
            url: The URL of the media to download.
            filename: The name to save the file as.

        Returns:
            The Path to the downloaded file.

        Raises:
            RuntimeError: If the download fails.
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.threads.net/"}
            response = requests.get(url=url, headers=headers, stream=True, timeout=15)
            response.raise_for_status()

            filepath = Path(self.output_folder) / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with filepath.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return filepath
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download media from {url}: {e}") from e

    def extract_post_data(self, url: str) -> tuple[Post | None, list[Post]]:
        """Extracts post data and its parents from a Threads URL.

        Args:
            url: The raw Threads post URL.

        Returns:
            A tuple containing:
                - The extracted Post instance if found, else None.
                - A list of ancestor Post instances.
        """
        threads_url = ThreadsURL(raw_url=url)
        html = self._fetch_html(url=threads_url.clean_url)
        return self._parse_post_from_html(html=html, post_code=threads_url.post_code)

    @staticmethod
    def _post_url(post: Post) -> str:
        """Reconstructs a canonical Threads URL from a post's author handle and code."""
        username = post.author_name
        code = post.code
        if username and code:
            return f"https://www.threads.com/@{username}/post/{code}"
        return ""

    def _build_output(self, post: Post, url: str, download: bool) -> ThreadsOutput:
        """Builds a ThreadsOutput object from a Post object."""
        post_code = post.code or "unknown"
        image_urls: list[str] = []
        video_urls: list[str] = []
        video_paths: list[Path] = []

        for i, media_url in enumerate(post.media_urls):
            ext = self._determine_extension(media_url=media_url)
            if ext == "mp4":
                video_urls.append(media_url)
                if download:
                    filename = f"threads_{post_code}_{i}.{ext}"
                    filepath = self.download_media(url=media_url, filename=filename)
                    if filepath:
                        video_paths.append(filepath)
            else:
                image_urls.append(media_url)

        taken_at = datetime.fromtimestamp(post.taken_at, tz=UTC) if post.taken_at else None

        return ThreadsOutput(
            text=post.caption_text,
            url=url,
            image_urls=image_urls,
            video_urls=video_urls,
            video_paths=video_paths,
            author_name=post.author_name,
            author_icon_url=post.author_icon_url,
            like_count=post.like_count or 0,
            reply_count=post.reply_count,
            repost_count=post.repost_count,
            quote_count=post.quote_count,
            reshare_count=post.reshare_count,
            taken_at=taken_at,
        )

    @contextlib.contextmanager
    def parse(self, url: str) -> Iterator[list[ThreadsOutput]]:
        """Parses a Threads post URL and yields the reply chain.

        The yielded list is ordered `[root, ..., direct_parent, target]`. The
        target post (last element) has its videos downloaded into
        `output_folder`; ancestor posts do not. Downloaded video files are
        removed when the context manager exits.

        Args:
            url: The Threads post URL.

        Yields:
            Ordered posts in the reply chain. Empty when no post is found.
        """
        post, parent_posts = self.extract_post_data(url=url)
        results: list[ThreadsOutput] = []
        if post:
            for parent in parent_posts:
                results.append(
                    self._build_output(
                        post=parent, url=self._post_url(post=parent), download=False
                    )
                )
            results.append(self._build_output(post=post, url=url, download=True))
        try:
            yield results
        finally:
            for output in results:
                output.unlink()


# Caps the relay's pending-future map. parse_threads registers an entry for every Threads
# message (whether or not the bot was tagged), so the bound stops links nobody asked the bot
# about from growing it without limit; 128 comfortably covers concurrent in-flight expansions.
_MAX_PENDING_EXPANSIONS = 128


class ThreadsExpansionRelay(BaseModel):
    """Hands the embed reply parse_threads posts to the reply pipeline, keyed by source message.

    parse_threads and the reply pipeline both fire on the same message: parse_threads
    publishes the expanded reply it posts (or None when the post is private / unparsable)
    and the reply pipeline awaits it, so the answer model can read the post without a second
    fetch. Futures are created in the running loop on first touch by either side (race-safe),
    and the map is bounded so expansions nobody awaits cannot grow it without limit.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _pending: "OrderedDict[int, asyncio.Future[Message | None]]" = PrivateAttr(
        default_factory=OrderedDict
    )

    def get_or_create(self, *, message_id: int) -> "asyncio.Future[Message | None]":
        """Returns the pending future for a message, creating it in the running loop if absent.

        A future bound to a different loop (a fresh test loop reusing the module-level
        singleton) is treated as absent and replaced, so the relay stays correct across runs.
        """
        loop = asyncio.get_running_loop()
        existing = self._pending.get(message_id)
        if existing is not None and existing.get_loop() is loop:
            self._pending.move_to_end(message_id)
            return existing
        future: asyncio.Future[Message | None] = loop.create_future()
        self._pending[message_id] = future
        self._pending.move_to_end(message_id)
        while len(self._pending) > _MAX_PENDING_EXPANSIONS:
            self._pending.popitem(last=False)
        return future

    def resolve(self, *, message_id: int, message: "Message | None") -> None:
        """Resolves a pending future with the posted expansion, or None when there is none."""
        future = self._pending.get(message_id)
        if (
            future is not None
            and not future.done()
            and future.get_loop() is asyncio.get_running_loop()
        ):
            future.set_result(message)


# Module-level singleton shared by the two cogs (same pattern as the DB engines): both import
# this instance so a publish on one side is visible to the awaiter on the other.
threads_expansion_relay = ThreadsExpansionRelay()


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    downloader = ThreadsDownloader(output_folder="./tmp")
    url = "https://www.threads.com/@chengweilai2/post/DZZImVsCWU-?xmt=AQG0MLHN7M4RJOdbdF1HzSG5Qm-9a1b2aOB5HN4ksjCrhQ"
    with downloader.parse(url=url) as outputs:
        console.print(outputs)
