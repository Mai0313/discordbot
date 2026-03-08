import re
import json
import types
from pathlib import Path
from urllib.parse import urlparse

from rich import get_console
from pydantic import Field, BaseModel
import requests

console = get_console()


class ThreadsOutput(BaseModel):
    """Output model for Threads downloader."""

    text: str = Field(default="找不到貼文", description="The text content of the post")
    url: str = Field(default="", description="The original post URL")
    image_urls: list[str] = Field(default=[], description="List of image URLs")
    video_paths: list[Path] = Field(default=[], description="List of downloaded video file paths")
    author_name: str = Field(default="", description="The username of the post author")
    author_icon_url: str = Field(default="", description="The profile icon URL of the post author")
    like_count: int = Field(default=0, description="The number of likes the post received")
    reply_count: int = Field(default=0, description="The number of replies the post received")
    repost_count: int = Field(default=0, description="The number of reposts the post received")
    quote_count: int = Field(default=0, description="The number of quotes the post received")
    reshare_count: int = Field(default=0, description="The number of reshares the post received")

    def unlink(self) -> None:
        """Unlink (delete) the downloaded video files."""
        for path in self.video_paths:
            path.unlink(missing_ok=True)

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ):
        """Exit the context manager and cleanup."""
        self.unlink()


class Caption(BaseModel):
    text: str = Field(default="", description="The caption text of the post")


class User(BaseModel):
    username: str = Field(default="", description="The username of the post author")
    full_name: str = Field(default="", description="The full name of the post author")
    profile_pic_url: str = Field(
        default="", description="The profile picture URL of the post author"
    )


class VideoVersion(BaseModel):
    url: str = Field(default="")


class ImageCandidate(BaseModel):
    url: str = Field(default="")


class ImageVersions2(BaseModel):
    candidates: list[ImageCandidate] = Field(default_factory=list)


class CarouselMedia(BaseModel):
    video_versions: list[VideoVersion] | None = Field(default=None)
    image_versions2: ImageVersions2 | None = Field(default=None)


class Fragment(BaseModel):
    plaintext: str = Field(default="")


class TextFragments(BaseModel):
    fragments: list[Fragment] = Field(default_factory=list)


class TextPostAppInfo(BaseModel):
    direct_reply_count: int = Field(default=0)
    repost_count: int = Field(default=0)
    quote_count: int = Field(default=0)
    reshare_count: int = Field(default=0)
    text_fragments: TextFragments | None = Field(default=None)


class Post(BaseModel):
    code: str = Field(default="")
    caption: Caption | None = Field(default=None)
    user: User | None = Field(default=None)
    like_count: int = Field(default=0, description="The number of likes the post received")
    carousel_media: list[CarouselMedia] | None = Field(default=None)
    video_versions: list[VideoVersion] | None = Field(default=None)
    image_versions2: ImageVersions2 | None = Field(default=None)
    text_post_app_info: TextPostAppInfo | None = Field(default=None)


class ThreadsDownloader(BaseModel):
    """A downloader for extracting text and media from Threads.net posts."""

    output_folder: str = Field(default="./data/threads", description="Download folder")

    def _default_http_headers(self) -> dict[str, str]:
        return {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}

    def _find_keys(self, obj: dict | list | str | float | None, key: str) -> list[dict]:
        results: list[dict] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key and isinstance(v, list | dict):
                    results.append(v)  # type: ignore[arg-type]
                else:
                    results.extend(self._find_keys(v, key))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(self._find_keys(item, key))
        return results

    def _fetch_html(self, clean_url: str) -> str | None:
        headers = self._default_http_headers()
        try:
            response = requests.get(clean_url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch HTML from {clean_url}: {e}") from e

    def _parse_post_from_html(self, html: str, post_code: str) -> Post | None:
        for match in re.finditer(
            r'<script type="application/json"[^>]*data-sjs>(.*?)</script>', html, re.DOTALL
        ):
            text = match.group(1)
            if "thread_items" not in text:
                continue

            try:
                data = json.loads(text)
                items = self._find_keys(data, "thread_items")
                for item_list in items:
                    if not isinstance(item_list, list):
                        continue
                    for item in item_list:
                        if not isinstance(item, dict):
                            continue
                        post_data = item.get("post")
                        if not post_data or not isinstance(post_data, dict):
                            continue
                        if post_data.get("code") == post_code:
                            return Post.model_validate(post_data)
            except (json.JSONDecodeError, ValueError):
                continue

        return None

    def extract_post_data(self, url: str) -> Post | None:
        """Extracts the post JSON data from the given Threads URL."""
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc

        if netloc in ("www.threads.com", "threads.com"):
            netloc = "www.threads.net"

        clean_url = f"{parsed_url.scheme}://{netloc}{parsed_url.path}"
        path_parts = parsed_url.path.strip("/").split("/")
        post_code = path_parts[-1] if len(path_parts) > 0 else ""

        html = self._fetch_html(clean_url)
        if not html:
            return None

        return self._parse_post_from_html(html, post_code)

    def download_media(self, url: str, filename: str) -> Path | None:
        """Downloads a media file from the given URL and saves it to the output folder."""
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

    def _extract_media_urls(self, post: Post) -> list[str]:
        media_urls: list[str] = []

        if post.carousel_media:
            for c in post.carousel_media:
                if c.video_versions and len(c.video_versions) > 0:
                    media_urls.append(c.video_versions[0].url)
                    continue

                if (
                    c.image_versions2
                    and c.image_versions2.candidates
                    and len(c.image_versions2.candidates) > 0
                ):
                    media_urls.append(c.image_versions2.candidates[0].url)
        elif post.video_versions and len(post.video_versions) > 0:
            media_urls.append(post.video_versions[0].url)
        elif (
            post.image_versions2
            and post.image_versions2.candidates
            and len(post.image_versions2.candidates) > 0
        ):
            media_urls.append(post.image_versions2.candidates[0].url)

        return [url for url in media_urls if url]

    def _determine_extension(self, media_url: str) -> str:
        parsed_media_url = urlparse(media_url)
        path_lower = parsed_media_url.path.lower()

        ext = "mp4"
        if ".jpg" in path_lower or ".jpeg" in path_lower:
            ext = "jpg"
        elif ".webp" in path_lower:
            ext = "webp"
        elif ".png" in path_lower:
            ext = "png"
        elif ".mp4" in path_lower:
            ext = "mp4"
        elif "video" not in media_url and "mp4" not in media_url:
            ext = "jpg"
        return ext

    def parse(self, url: str) -> ThreadsOutput:
        """Main method to extract text and download all media from a Threads URL."""
        post = self.extract_post_data(url)
        if not post:
            return ThreadsOutput()

        caption_text = post.caption.text if post.caption else ""
        if post.text_post_app_info and post.text_post_app_info.text_fragments:
            fragments_text = "".join(
                f.plaintext for f in post.text_post_app_info.text_fragments.fragments
            )
            if fragments_text:
                caption_text = fragments_text

        post_code = post.code or "unknown"
        author_name = post.user.username if post.user else ""
        author_icon_url = post.user.profile_pic_url if post.user else ""
        like_count = post.like_count

        reply_count = 0
        repost_count = 0
        quote_count = 0
        reshare_count = 0
        if post.text_post_app_info:
            reply_count = post.text_post_app_info.direct_reply_count
            repost_count = post.text_post_app_info.repost_count
            quote_count = post.text_post_app_info.quote_count
            reshare_count = post.text_post_app_info.reshare_count

        media_urls = self._extract_media_urls(post)

        image_urls: list[str] = []
        video_paths: list[Path] = []
        for i, media_url in enumerate(media_urls):
            ext = self._determine_extension(media_url)
            if ext == "mp4":
                filename = f"threads_{post_code}_{i}.{ext}"
                filepath = self.download_media(media_url, filename)
                if filepath:
                    video_paths.append(filepath)
            else:
                image_urls.append(media_url)

        return ThreadsOutput(
            text=caption_text,
            url=url,
            image_urls=image_urls,
            video_paths=video_paths,
            author_name=author_name,
            author_icon_url=author_icon_url,
            like_count=like_count,
            reply_count=reply_count,
            repost_count=repost_count,
            quote_count=quote_count,
            reshare_count=reshare_count,
        )


if __name__ == "__main__":
    test_url = "https://www.threads.com/@c32971/post/DVnt6dciSRc?xmt=AQF0GAejzXClnOrILy2_aqEN7a0IhvY6Nq4iAsUbI0K_Yw"
    # test_url = "https://www.threads.com/@cyj308/post/DVn6dqzjzQf?hl=zh-tw"
    downloader = ThreadsDownloader()
    with downloader.parse(test_url) as result:
        console.print(result)
