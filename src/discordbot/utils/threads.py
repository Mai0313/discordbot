import re
import json
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, BaseModel
import requests


class ThreadsOutput(BaseModel):
    """Output model for Threads downloader."""

    text: str = Field(default="找不到貼文", description="The text content of the post")
    media_urls: list[str] = Field(default=[], description="List of media URLs")
    media_paths: list[Path] = Field(default=[], description="List of downloaded media file paths")


class Caption(BaseModel):
    text: str = Field(default="", description="The caption text of the post")


class ThreadsDownloader(BaseModel):
    """A downloader for extracting text and media from Threads.net posts."""

    output_folder: str = Field(default="./data/threads", description="Download folder")
    headers: dict[str, str] = Field(default={"User-Agent": "Mozilla/5.0", "Accept": "text/html"})
    timeout: int = Field(default=15, description="Timeout for requests")

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
        try:
            response = requests.get(clean_url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch HTML from {clean_url}: {e}") from e

    def _parse_post_from_html(self, html: str, post_code: str) -> dict | None:
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
                        post = item.get("post", {})
                        if isinstance(post, dict) and post.get("code") == post_code:
                            return post
            except json.JSONDecodeError:
                continue

        return None

    def extract_post_data(self, url: str) -> dict | None:
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
            response = requests.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()

            filepath = Path(self.output_folder) / filename
            with Path.open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return filepath
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download media from {url}: {e}") from e

    def _extract_media_urls(self, post: dict) -> list[str]:
        media_urls: list[str] = []
        carousel_media = post.get("carousel_media")

        if carousel_media and isinstance(carousel_media, list):
            for c in carousel_media:
                if not isinstance(c, dict):
                    continue
                c_video_versions = c.get("video_versions")
                if c_video_versions and isinstance(c_video_versions, list):
                    media_urls.append(c_video_versions[0].get("url", ""))
                    continue

                c_image_versions2 = c.get("image_versions2")
                if c_image_versions2 and isinstance(c_image_versions2, dict):
                    c_candidates = c_image_versions2.get("candidates", [])
                    if c_candidates and isinstance(c_candidates, list):
                        media_urls.append(c_candidates[0].get("url", ""))
        else:
            video_versions = post.get("video_versions")
            if video_versions and isinstance(video_versions, list):
                media_urls.append(video_versions[0].get("url", ""))
            else:
                image_versions2 = post.get("image_versions2")
                if image_versions2 and isinstance(image_versions2, dict):
                    candidates = image_versions2.get("candidates", [])
                    if candidates and isinstance(candidates, list):
                        media_urls.append(candidates[0].get("url", ""))

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

    def process_url(self, url: str) -> ThreadsOutput:
        """Main method to extract text and download all media from a Threads URL."""
        post = self.extract_post_data(url)
        if not post:
            return ThreadsOutput()

        caption: Caption = Caption(**post.get("caption", {}))

        post_code = post.get("code", "unknown")
        media_urls = self._extract_media_urls(post)

        media_paths: list[Path] = []
        for i, media_url in enumerate(media_urls):
            ext = self._determine_extension(media_url)
            filename = f"threads_{post_code}_{i}.{ext}"
            filepath = self.download_media(media_url, filename)
            if filepath:
                media_paths.append(filepath)

        return ThreadsOutput(text=caption.text, media_urls=media_urls, media_paths=media_paths)


def download_threads_post(url: str) -> ThreadsOutput:
    """Helper function to download Threads post text and media."""
    downloader = ThreadsDownloader()
    return downloader.process_url(url)


if __name__ == "__main__":
    test_url = "https://www.threads.com/@myun.60761/post/DVnP0ATET7d?xmt=AQF0GAejzXClnOrILy2_aqEN7a0IhvY6Nq4iAsUbI0K_Yw"
    result = download_threads_post(test_url)
    print(result)  # noqa: T201
