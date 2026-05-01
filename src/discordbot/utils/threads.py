import re
import json
import types
from pathlib import Path
from datetime import UTC, datetime
from functools import cached_property
from urllib.parse import urlparse

from rich import get_console
from pydantic import Field, BaseModel, computed_field
import requests

console = get_console()


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


class ThreadsURL(BaseModel):
    """Parses and normalises a Threads post URL."""

    raw_url: str

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        parsed = urlparse(self.raw_url)
        netloc = parsed.netloc
        if netloc in ("www.threads.com", "threads.com"):
            netloc = "www.threads.net"
        return f"{parsed.scheme}://{netloc}{parsed.path}"

    @computed_field
    @cached_property
    def post_code(self) -> str:
        parsed = urlparse(self.raw_url)
        path_parts = parsed.path.strip("/").split("/")
        return path_parts[-1] if path_parts else ""


# ---------------------------------------------------------------------------
# API data models
# ---------------------------------------------------------------------------


class User(BaseModel):
    username: str = Field(default="", description="Username handle")
    profile_pic_url: str = Field(default="", description="Profile picture URL")


class Caption(BaseModel):
    text: str = Field(default="", description="Caption text content")


class VideoVersion(BaseModel):
    url: str = Field(default="", description="Video file URL")


class ImageCandidate(BaseModel):
    url: str = Field(default="", description="Image URL")


class ImageVersions2(BaseModel):
    candidates: list[ImageCandidate] = Field(
        default_factory=list, description="Available image resolutions"
    )


class CarouselMedia(BaseModel):
    video_versions: list[VideoVersion] | None = Field(
        default=None, description="Available video versions"
    )
    image_versions2: ImageVersions2 | None = Field(
        default=None, description="Available image versions"
    )


class MediaContainer(BaseModel):
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
    plaintext: str = Field(default="", description="Plain text content of the fragment")


class TextFragments(BaseModel):
    fragments: list[Fragment] = Field(
        default_factory=list, description="Ordered list of text fragments"
    )


class LinkPreviewAttachment(BaseModel):
    title: str = Field(default="", description="Title shown in the link preview")
    image_url: str = Field(default="", description="Image shown in the link preview")
    url: str = Field(default="", description="Original link preview URL")


class LinkedInlineMedia(MediaContainer):
    code: str = Field(default="", description="Linked media short code")
    caption: Caption | None = Field(default=None, description="Linked media caption")


class TextPostAppInfo(BaseModel):
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


class Post(MediaContainer):
    """Represents a single Threads post parsed from the API JSON."""

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
        return self.user.username if self.user else ""

    @property
    def author_icon_url(self) -> str:
        return self.user.profile_pic_url if self.user else ""

    @property
    def reply_count(self) -> int:
        return (self.text_post_app_info.direct_reply_count or 0) if self.text_post_app_info else 0

    @property
    def repost_count(self) -> int:
        return (self.text_post_app_info.repost_count or 0) if self.text_post_app_info else 0

    @property
    def quote_count(self) -> int:
        return (self.text_post_app_info.quote_count or 0) if self.text_post_app_info else 0

    @property
    def reshare_count(self) -> int:
        return (self.text_post_app_info.reshare_count or 0) if self.text_post_app_info else 0

    @property
    def media_urls(self) -> list[str]:
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
    post: Post | None = Field(default=None)


class ThreadData(BaseModel):
    thread_items: list[ThreadItem] = Field(default_factory=list)

    def find_post_with_parents(self, post_code: str) -> tuple[Post | None, list[Post]]:
        """Return the matching post and the chronologically-ordered ancestors before it.

        Threads stores an entire reply chain (root → direct parent → target) in a single
        ``thread_items`` list, oldest first. Everything appearing before the target item is
        therefore an ancestor of it.
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
    """Output model for Threads downloader."""

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
    parents: list["ThreadsOutput"] = Field(
        default_factory=list,
        description="Ancestor posts in the reply chain, ordered oldest (root) → newest (direct parent)",
    )

    def unlink(self) -> None:
        for path in self.video_paths:
            path.unlink(missing_ok=True)
        for parent in self.parents:
            parent.unlink()

    def __enter__(self):  # noqa: D105
        return self

    def __exit__(  # noqa: D105
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ):
        self.unlink()


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

_SJS_PATTERN = re.compile(
    r'<script type="application/json"[^>]*data-sjs>(.*?)</script>', re.DOTALL
)


class ThreadsDownloader(BaseModel):
    """A downloader for extracting text and media from Threads.net posts."""

    output_folder: str = Field(default="./data/threads")

    # -- HTTP -----------------------------------------------------------------

    def _fetch_html(self, url: str) -> str:
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

    # -- Media download -------------------------------------------------------

    @staticmethod
    def _determine_extension(media_url: str) -> str:
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
        threads_url = ThreadsURL(raw_url=url)
        html = self._fetch_html(url=threads_url.clean_url)
        return self._parse_post_from_html(html=html, post_code=threads_url.post_code)

    @staticmethod
    def _post_url(post: Post) -> str:
        """Reconstruct a canonical Threads URL from a post's author handle and code."""
        username = post.author_name
        code = post.code
        if username and code:
            return f"https://www.threads.com/@{username}/post/{code}"
        return ""

    def _build_output(
        self, post: Post, url: str, *, download: bool, parents: list[ThreadsOutput] | None = None
    ) -> ThreadsOutput:
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
            parents=parents or [],
        )

    def parse(self, url: str) -> ThreadsOutput:
        post, parent_posts = self.extract_post_data(url=url)
        if not post:
            return ThreadsOutput()

        parents = [
            self._build_output(post=parent, url=self._post_url(post=parent), download=False)
            for parent in parent_posts
        ]
        return self._build_output(post=post, url=url, download=True, parents=parents)


if __name__ == "__main__":
    test_url = "https://www.threads.com/@lift4life_nickson/post/DXy_VeVmSGK"
    # test_url = "https://www.threads.com/@cyj308/post/DVn6dqzjzQf?hl=zh-tw"
    downloader = ThreadsDownloader()
    with downloader.parse(url=test_url) as result:
        console.print(result)
