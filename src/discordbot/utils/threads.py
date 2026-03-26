import re
import json
import types
from pathlib import Path
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
    id: str = Field(default="")
    pk: str = Field(default="")
    username: str = Field(default="")
    full_name: str = Field(default="")
    profile_pic_url: str = Field(default="")
    is_verified: bool = Field(default=False)
    text_post_app_is_private: bool = Field(default=False)


class Caption(BaseModel):
    text: str = Field(default="")
    pk: str = Field(default="")
    has_translation: bool = Field(default=False)


class VideoVersion(BaseModel):
    url: str = Field(default="")


class ImageCandidate(BaseModel):
    url: str = Field(default="")
    height: int = Field(default=0)
    width: int = Field(default=0)


class ImageVersions2(BaseModel):
    candidates: list[ImageCandidate] = Field(default_factory=list)


class CarouselMedia(BaseModel):
    video_versions: list[VideoVersion] | None = Field(default=None)
    image_versions2: ImageVersions2 | None = Field(default=None)
    accessibility_caption: str | None = Field(default=None)
    media_type: int | None = Field(default=None)
    has_audio: bool | None = Field(default=None)


class Fragment(BaseModel):
    fragment_type: str = Field(default="plaintext")
    plaintext: str = Field(default="")


class TextFragments(BaseModel):
    fragments: list[Fragment] = Field(default_factory=list)


class ShareInfo(BaseModel):
    quoted_post: dict | None = Field(default=None)
    reposted_post: dict | None = Field(default=None)


class PinnedPostInfo(BaseModel):
    is_pinned_to_parent_post: bool = Field(default=False)
    is_pinned_to_profile: bool = Field(default=False)


class TextPostAppInfo(BaseModel):
    direct_reply_count: int = Field(default=0)
    repost_count: int = Field(default=0)
    quote_count: int = Field(default=0)
    reshare_count: int = Field(default=0)
    is_spoiler_media: bool = Field(default=False)
    is_reply: bool = Field(default=False)
    is_post_unavailable: bool = Field(default=False)
    reply_control: str = Field(default="everyone")
    text_fragments: TextFragments | None = Field(default=None)
    share_info: ShareInfo | None = Field(default=None)
    pinned_post_info: PinnedPostInfo | None = Field(default=None)
    self_thread_count: int = Field(default=0)


class Post(BaseModel):
    """Represents a single Threads post parsed from the API JSON."""

    pk: str = Field(default="")
    code: str = Field(default="")
    caption: Caption | None = Field(default=None)
    user: User | None = Field(default=None)
    like_count: int = Field(default=0)
    carousel_media: list[CarouselMedia] | None = Field(default=None)
    video_versions: list[VideoVersion] | None = Field(default=None)
    image_versions2: ImageVersions2 | None = Field(default=None)
    text_post_app_info: TextPostAppInfo | None = Field(default=None)
    media_type: int | None = Field(default=None)
    original_height: int | None = Field(default=None)
    original_width: int | None = Field(default=None)
    taken_at: int | None = Field(default=None)
    caption_is_edited: bool = Field(default=False)
    like_and_view_counts_disabled: bool = Field(default=False)
    accessibility_caption: str | None = Field(default=None)

    # -- derived properties ---------------------------------------------------

    @property
    def caption_text(self) -> str:
        if self.text_post_app_info and self.text_post_app_info.text_fragments:
            fragments_text = "".join(
                f.plaintext for f in self.text_post_app_info.text_fragments.fragments
            )
            if fragments_text:
                return fragments_text
        return self.caption.text if self.caption else ""

    @property
    def author_name(self) -> str:
        return self.user.username if self.user else ""

    @property
    def author_icon_url(self) -> str:
        return self.user.profile_pic_url if self.user else ""

    @property
    def reply_count(self) -> int:
        return self.text_post_app_info.direct_reply_count if self.text_post_app_info else 0

    @property
    def repost_count(self) -> int:
        return self.text_post_app_info.repost_count if self.text_post_app_info else 0

    @property
    def quote_count(self) -> int:
        return self.text_post_app_info.quote_count if self.text_post_app_info else 0

    @property
    def reshare_count(self) -> int:
        return self.text_post_app_info.reshare_count if self.text_post_app_info else 0

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


# ---------------------------------------------------------------------------
# HTML → Post extraction models
# ---------------------------------------------------------------------------


class ThreadItem(BaseModel):
    post: Post | None = Field(default=None)


class ThreadData(BaseModel):
    thread_items: list[ThreadItem] = Field(default_factory=list)

    def find_post(self, post_code: str) -> Post | None:
        for item in self.thread_items:
            if item.post and item.post.code == post_code:
                return item.post
        return None


# ---------------------------------------------------------------------------
# Output model (public API — fields unchanged)
# ---------------------------------------------------------------------------


class ThreadsOutput(BaseModel):
    """Output model for Threads downloader."""

    text: str = Field(default="找不到貼文")
    url: str = Field(default="")
    image_urls: list[str] = Field(default=[])
    video_paths: list[Path] = Field(default=[])
    author_name: str = Field(default="")
    author_icon_url: str = Field(default="")
    like_count: int = Field(default=0)
    reply_count: int = Field(default=0)
    repost_count: int = Field(default=0)
    quote_count: int = Field(default=0)
    reshare_count: int = Field(default=0)

    def unlink(self) -> None:
        for path in self.video_paths:
            path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(
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
                    results.extend(ThreadsDownloader._find_keys(v, key))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(ThreadsDownloader._find_keys(item, key))
        return results

    def _parse_post_from_html(self, html: str, post_code: str) -> Post | None:
        for match in _SJS_PATTERN.finditer(html):
            text = match.group(1)
            if "thread_items" not in text:
                continue

            try:
                data = json.loads(text)
                raw_lists = self._find_keys(data, "thread_items")
                for raw_items in raw_lists:
                    if not isinstance(raw_items, list):
                        continue
                    thread_data = ThreadData(thread_items=raw_items)
                    post = thread_data.find_post(post_code)
                    if post:
                        return post
            except (json.JSONDecodeError, ValueError):
                continue

        return None

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
            response = requests.get(url, stream=True, timeout=15)
            response.raise_for_status()

            filepath = Path(self.output_folder) / filename
            with Path.open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return filepath
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download media from {url}: {e}") from e

    # -- Public API -----------------------------------------------------------

    def extract_post_data(self, url: str) -> Post | None:
        threads_url = ThreadsURL(raw_url=url)
        html = self._fetch_html(threads_url.clean_url)
        return self._parse_post_from_html(html, threads_url.post_code)

    def parse(self, url: str) -> ThreadsOutput:
        post = self.extract_post_data(url)
        if not post:
            return ThreadsOutput()

        post_code = post.code or "unknown"
        image_urls: list[str] = []
        video_paths: list[Path] = []

        for i, media_url in enumerate(post.media_urls):
            ext = self._determine_extension(media_url)
            if ext == "mp4":
                filename = f"threads_{post_code}_{i}.{ext}"
                filepath = self.download_media(media_url, filename)
                if filepath:
                    video_paths.append(filepath)
            else:
                image_urls.append(media_url)

        return ThreadsOutput(
            text=post.caption_text,
            url=url,
            image_urls=image_urls,
            video_paths=video_paths,
            author_name=post.author_name,
            author_icon_url=post.author_icon_url,
            like_count=post.like_count,
            reply_count=post.reply_count,
            repost_count=post.repost_count,
            quote_count=post.quote_count,
            reshare_count=post.reshare_count,
        )


if __name__ == "__main__":
    test_url = "https://www.threads.com/@c32971/post/DVnt6dciSRc?xmt=AQF0GAejzXClnOrILy2_aqEN7a0IhvY6Nq4iAsUbI0K_Yw"
    # test_url = "https://www.threads.com/@cyj308/post/DVn6dqzjzQf?hl=zh-tw"
    downloader = ThreadsDownloader()
    with downloader.parse(test_url) as result:
        console.print(result)
