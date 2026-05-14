"""Threads URL parsing, API models, and media download helpers."""

import re
import json
from pathlib import Path
from datetime import UTC, datetime
from functools import cached_property
import contextlib
from urllib.parse import urlparse
from collections.abc import Iterator

from rich import get_console
from pydantic import Field, BaseModel, computed_field
import requests

console = get_console()


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


class ThreadsURL(BaseModel):
    """Parses and normalises a Threads post URL.

    Attributes:
        raw_url: Original Threads URL provided by the caller.
    """

    raw_url: str

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        """The cleaned and normalised URL.

        Returns:
            URL with ``threads.com`` hosts normalised to ``www.threads.net`` and
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


# ---------------------------------------------------------------------------
# API data models
# ---------------------------------------------------------------------------


class User(BaseModel):
    """Represents a Threads user.

    Attributes:
        username: User handle.
        profile_pic_url: Profile picture URL.
    """

    username: str = Field(default="", description="Username handle")
    profile_pic_url: str = Field(default="", description="Profile picture URL")


class Caption(BaseModel):
    """Represents caption text attached to a Threads post.

    Attributes:
        text: Caption text content.
    """

    text: str = Field(default="", description="Caption text content")


class VideoVersion(BaseModel):
    """Represents an available video rendition.

    Attributes:
        url: Video file URL.
    """

    url: str = Field(default="", description="Video file URL")


class ImageCandidate(BaseModel):
    """Represents an available image rendition.

    Attributes:
        url: Image URL.
    """

    url: str = Field(default="", description="Image URL")


class ImageVersions2(BaseModel):
    """Holds available image renditions for a media object.

    Attributes:
        candidates: Available image resolutions.
    """

    candidates: list[ImageCandidate] = Field(
        default_factory=list, description="Available image resolutions"
    )


class CarouselMedia(BaseModel):
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


class MediaContainer(BaseModel):
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


class Fragment(BaseModel):
    """Represents one text fragment from Threads structured text.

    Attributes:
        plaintext: Plain text content of the fragment.
    """

    plaintext: str = Field(default="", description="Plain text content of the fragment")


class TextFragments(BaseModel):
    """Holds ordered structured text fragments.

    Attributes:
        fragments: Ordered list of text fragments.
    """

    fragments: list[Fragment] = Field(
        default_factory=list, description="Ordered list of text fragments"
    )


class LinkPreviewAttachment(BaseModel):
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


class TextPostAppInfo(BaseModel):
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

    # -- derived properties ---------------------------------------------------

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
            True when ``text_post_app_info.is_reply`` is set; False otherwise.
        """
        return bool(self.text_post_app_info and self.text_post_app_info.is_reply)

    @property
    def reply_to_username(self) -> str:
        """The username of the post being directly replied to.

        Returns:
            Username from ``text_post_app_info.reply_to_author``, or an empty
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


# ---------------------------------------------------------------------------
# HTML → Post extraction models
# ---------------------------------------------------------------------------


class ThreadItem(BaseModel):
    """Represents one item in a Threads reply chain.

    Attributes:
        post: Parsed post for this thread item.
    """

    post: Post | None = Field(default=None)


class ThreadData(BaseModel):
    """Represents Threads reply-chain data extracted from HTML JSON.

    Attributes:
        thread_items: Ordered thread items from the embedded JSON.
    """

    thread_items: list[ThreadItem] = Field(default_factory=list)

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


# ---------------------------------------------------------------------------
# Output model (public API — fields unchanged)
# ---------------------------------------------------------------------------


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

    text: str = Field(default="")
    url: str = Field(default="")
    image_urls: list[str] = Field(default=[])
    video_urls: list[str] = Field(default_factory=list)
    video_paths: list[Path] = Field(default=[])
    author_name: str = Field(default="")
    author_icon_url: str = Field(default="")
    like_count: int = Field(default=0)
    reply_count: int = Field(default=0)
    repost_count: int = Field(default=0)
    quote_count: int = Field(default=0)
    reshare_count: int = Field(default=0)
    taken_at: datetime | None = Field(default=None, description="Post creation time")

    def unlink(self) -> None:
        """Deletes downloaded video files for this post."""
        for path in self.video_paths:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

_SJS_PATTERN = re.compile(
    r'<script type="application/json"[^>]*data-sjs>(.*?)</script>', re.DOTALL
)


class ThreadsDownloader(BaseModel):
    """A downloader for extracting text and media from Threads.net posts.

    Attributes:
        output_folder: Directory where downloaded media files are written.
    """

    output_folder: str = Field(default="./data/threads")

    # -- HTTP -----------------------------------------------------------------

    def _fetch_html(self, url: str) -> str:
        """Fetches the HTML content of the given URL."""
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch HTML from {url}: {e}") from e

    # -- HTML parsing ---------------------------------------------------------

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
    def _collect_threads(data: dict | list) -> list[ThreadData]:
        """Collects every ``thread_items`` list found in a parsed SJS payload."""
        threads: list[ThreadData] = []
        for raw_items in ThreadsDownloader._find_keys(obj=data, key="thread_items"):
            if isinstance(raw_items, list):
                threads.append(ThreadData(thread_items=raw_items))
        return threads

    @staticmethod
    def _locate_target(threads: list[ThreadData], post_code: str) -> tuple[Post | None, int]:
        """Returns the target post and the index of the thread that contains it."""
        for i, thread in enumerate(threads):
            post, _ = thread.find_post_with_parents(post_code=post_code)
            if post:
                return post, i
        return None, -1

    @staticmethod
    def _collect_direct_replies(
        threads: list[ThreadData], target_idx: int, target_author: str
    ) -> list[Post]:
        """Returns the first post of each non-target branch whose reply is to ``target_author``."""
        replies: list[Post] = []
        for i, thread in enumerate(threads):
            if i == target_idx:
                continue
            for item in thread.thread_items:
                if item.post is None:
                    continue
                if item.post.reply_to_username == target_author:
                    replies.append(item.post)
                break
        return replies

    def _parse_post_with_replies_from_html(
        self, html: str, post_code: str
    ) -> tuple[Post | None, list[Post]]:
        """Parses the target post and its direct replies from the SJS script tags.

        Threads renders a single SJS block per post page that contains every
        relevant ``thread_items`` list: one for the chain ending at the target,
        and one per branch of replies. A branch's first post is a reply whose
        ``reply_to_author`` identifies what it is replying to. We rely on this
        to filter out siblings (which point to the target's parent) when the
        target is itself a reply.

        Args:
            html: The full HTML of the Threads page.
            post_code: The short code of the target post.

        Returns:
            A tuple containing:
                - The target Post instance if found, else None.
                - A list of direct reply Post instances, in page order.
        """
        for match in _SJS_PATTERN.finditer(string=html):
            text = match.group(1)
            if "thread_items" not in text:
                continue
            try:
                data = json.loads(s=text)
            except (json.JSONDecodeError, ValueError):
                continue

            threads = self._collect_threads(data=data)
            target_post, target_idx = self._locate_target(threads=threads, post_code=post_code)
            if target_post is None:
                continue
            replies = self._collect_direct_replies(
                threads=threads, target_idx=target_idx, target_author=target_post.author_name
            )
            return target_post, replies

        return None, []

    # -- Media download -------------------------------------------------------

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
            response = requests.get(url, headers=headers, stream=True, timeout=15)
            response.raise_for_status()

            filepath = Path(self.output_folder) / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with Path.open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return filepath
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download media from {url}: {e}") from e

    # -- Public API -----------------------------------------------------------

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

        The yielded list is ordered ``[root, ..., direct_parent, target]``. The
        target post (last element) has its videos downloaded into
        ``output_folder``; ancestor posts do not. Downloaded video files are
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

    def list_all(self, url: str) -> list[ThreadsOutput]:
        """Returns the target post followed by every direct reply to it.

        The first element is the target; subsequent elements are direct replies
        in the order Threads renders them on the page (newest first). Nested
        replies (replies to replies) are deliberately omitted. No videos are
        downloaded by this method; callers that need local video files should
        invoke ``parse`` on the specific reply URL instead.

        Args:
            url: The Threads post URL.

        Returns:
            Ordered list ``[target, reply_1, reply_2, ...]``. Empty when the
            target post cannot be found.
        """
        threads_url = ThreadsURL(raw_url=url)
        html = self._fetch_html(url=threads_url.clean_url)
        target_post, reply_posts = self._parse_post_with_replies_from_html(
            html=html, post_code=threads_url.post_code
        )
        results: list[ThreadsOutput] = []
        if target_post is None:
            return results
        results.append(self._build_output(post=target_post, url=url, download=False))
        for reply in reply_posts:
            results.append(
                self._build_output(post=reply, url=self._post_url(post=reply), download=False)
            )
        return results


if __name__ == "__main__":
    test_url = "https://www.threads.com/@cyj308/post/DVn6dqzjzQf?hl=zh-tw"
    downloader = ThreadsDownloader()
    with downloader.parse(url=test_url) as results:
        console.print(results)
    console.rule(title="list_all")
    console.print(downloader.list_all(url=test_url))
