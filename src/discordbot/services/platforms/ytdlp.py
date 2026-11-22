"""Synchronous yt-dlp video downloader/metadata probe and its asyncio stop-signal wrapper."""

import re
import shutil
from typing import Any, ClassVar
import asyncio
from pathlib import Path
import threading
import contextlib
from urllib.parse import parse_qs, urlparse
from collections.abc import Callable

from yt_dlp import YoutubeDL
import logfire
from pydantic import Field, BaseModel
from requests import Session
from requests.exceptions import RequestException

from discordbot.utils.urls import normalized_host, host_matches_domain
from discordbot.typings.video import VideoQuality
from discordbot.typings.timeouts import (
    YTDLP_RETRIES,
    DOWNLOAD_STOP_JOIN_SECONDS,
    YTDLP_SOCKET_TIMEOUT_SECONDS,
    SHARE_RESOLVE_TIMEOUT_SECONDS,
)
from discordbot.utils.file_downloads import DownloadTooLargeError
from discordbot.services.platforms.base import PlatformDownloader
from discordbot.services.platforms.facebook import (
    _LOGIN_WALL_PATHS,
    FacebookDownloader,
    is_facebook_post_url,
)
from discordbot.services.platforms.page_json import BROWSER_HEADERS


class DownloadStoppedError(Exception):
    """Raised inside yt-dlp when the caller's stop signal is set mid-download."""


class DownloadResult(BaseModel):
    """Represents one or more downloaded video files."""

    filenames: list[Path] = Field(..., description="Local paths of the downloaded files.")
    total_items: int = Field(
        default=0,
        description=(
            "Total entries reported by a playlist or carousel before applying `max_items`; "
            "0 when the source is a single video or does not report a total."
        ),
    )

    @property
    def omitted_items(self) -> int:
        """Returns the number of playlist items present in the source but not downloaded."""
        return max(0, self.total_items - len(self.filenames))


class VideoMetadata(BaseModel):
    """Metadata for a video, read by yt-dlp without downloading any media."""

    title: str = Field(default="", description="Video title reported by yt-dlp.")
    uploader: str = Field(default="", description="Uploader / channel display name.")
    description: str = Field(
        default="", description="Full video description; callers trim it to their own budget."
    )
    duration_seconds: float = Field(
        default=0.0, description="Duration in seconds; 0.0 when the site does not report one."
    )
    webpage_url: str = Field(
        default="", description="Canonical page URL after redirects, so a short link resolves."
    )
    is_live: bool = Field(
        default=False,
        description="Whether the URL points at a live stream rather than a finished video.",
    )
    from_playlist: bool = Field(
        default=False,
        description="Whether the fields describe the first entry of a playlist-shaped page (a space, a collection, a season) rather than the page itself.",
    )


def _positive_number(value: object) -> float | None:
    """Returns `value` as a positive float when numeric (excluding `bool`), else `None`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _non_negative_int(value: object) -> int | None:
    """Returns `value` as a non-negative int when numeric (excluding `bool`), else `None`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value >= 0 else None


def _is_direct_http_format(fmt: dict[str, Any]) -> bool:
    """Returns whether `fmt` is a direct HTTP(S) stream whose HEAD reports media byte size."""
    stream_url = fmt.get("url")
    if not isinstance(stream_url, str) or not stream_url:
        return False
    protocol = fmt.get("protocol")
    if isinstance(protocol, str) and protocol:
        return protocol in {"http", "https"}
    parsed = urlparse(url=stream_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    return not parsed.path.lower().endswith((".m3u8", ".mpd"))


def _head_content_length(session: Session, url: str, headers: dict[str, str]) -> int | None:
    """Returns positive `Content-Length` from a HEAD probe, or `None` when unavailable."""
    try:
        response = session.head(
            url=url, headers=headers, allow_redirects=True, timeout=YTDLP_SOCKET_TIMEOUT_SECONDS
        )
    except RequestException as error:
        logfire.debug(
            "HEAD size probe failed; falling back to bitrate estimate", url=url, _exc_info=error
        )
        return None
    try:
        if not response.ok:
            return None
        declared = response.headers.get("Content-Length")
        if declared is not None and declared.isdecimal():
            size = int(declared)
            return size if size > 0 else None
        return None
    finally:
        response.close()


def _estimate_format_bytes(
    fmt: dict[str, Any],
    duration: float | None,
    default_headers: dict[str, str],
    session: Session,
    stop_signal: threading.Event | None,
) -> int | None:
    """Estimates bytes for one format via metadata, HEAD `Content-Length`, or `duration * tbr`."""
    for key in ("filesize", "filesize_approx"):
        known = _positive_number(value=fmt.get(key))
        if known is not None:
            return int(known)

    if _is_direct_http_format(fmt=fmt):
        if stop_signal is not None and stop_signal.is_set():
            raise DownloadStoppedError("download stopped during size probe")
        fmt_headers = fmt.get("http_headers")
        headers = fmt_headers if isinstance(fmt_headers, dict) else default_headers
        head_size = _head_content_length(session=session, url=fmt["url"], headers=headers)
        if head_size is not None:
            return head_size

    fmt_duration = _positive_number(value=fmt.get("duration")) or duration
    tbr = (
        _positive_number(value=fmt.get("tbr"))
        or _positive_number(value=fmt.get("vbr"))
        or _positive_number(value=fmt.get("abr"))
    )
    if fmt_duration is not None and tbr is not None:
        return int(fmt_duration * tbr * 1000 / 8)
    return None


def _estimate_entry_bytes(
    entry: dict[str, Any],
    default_headers: dict[str, str],
    session: Session,
    stop_signal: threading.Event | None,
    max_bytes: int | None = None,
) -> int | None:
    """Estimates total byte size for one video entry, or `None` when no signal is available."""
    for key in ("filesize", "filesize_approx"):
        known = _positive_number(value=entry.get(key))
        if known is not None:
            return int(known)

    duration = _positive_number(value=entry.get("duration"))
    requested_formats = entry.get("requested_formats")
    if isinstance(requested_formats, list) and requested_formats:
        total = 0
        any_known = False
        for fmt in requested_formats:
            if not isinstance(fmt, dict):
                continue
            estimated = _estimate_format_bytes(
                fmt=fmt,
                duration=duration,
                default_headers=default_headers,
                session=session,
                stop_signal=stop_signal,
            )
            if estimated is None:
                continue
            total += estimated
            any_known = True
            if max_bytes is not None and total > max_bytes:
                return total
        if any_known:
            return total

    return _estimate_format_bytes(
        fmt=entry,
        duration=duration,
        default_headers=default_headers,
        session=session,
        stop_signal=stop_signal,
    )


def _build_byte_cap_hook(max_bytes: int, url: str) -> Callable[[dict[str, Any]], None]:
    """Returns a yt-dlp progress hook raising `DownloadTooLargeError` once cumulative bytes pass `max_bytes`."""
    finalized_bytes = 0
    active_streams: dict[str, int] = {}

    def _enforce_byte_cap(progress: dict[str, Any]) -> None:
        nonlocal finalized_bytes
        status = progress.get("status")
        stream_key = str(progress.get("filename") or progress.get("tmpfilename") or "")
        downloaded = _non_negative_int(value=progress.get("downloaded_bytes"))
        total_bytes = _positive_number(value=progress.get("total_bytes"))

        if status == "finished":
            prev = active_streams.pop(stream_key, 0)
            stream_final = max(
                prev,
                downloaded if downloaded is not None else 0,
                int(total_bytes) if total_bytes is not None else 0,
            )
            finalized_bytes += stream_final
            if finalized_bytes + sum(active_streams.values()) > max_bytes:
                raise DownloadTooLargeError(
                    f"Media at {url} exceeds {max_bytes} bytes during download"
                )
            return

        if downloaded is not None:
            prev = active_streams.get(stream_key, 0)
            # A drop on the same stream key signals a reset across streams or fragments.
            if downloaded < prev:
                finalized_bytes += prev
            active_streams[stream_key] = downloaded

        cumulative = finalized_bytes + sum(active_streams.values())
        current = active_streams.get(stream_key, 0)
        projected = (
            cumulative - current + int(total_bytes)
            if total_bytes is not None and int(total_bytes) > current
            else cumulative
        )
        if projected > max_bytes:
            raise DownloadTooLargeError(
                f"Media at {url} exceeds {max_bytes} bytes during download"
            )

    return _enforce_byte_cap


def _extract_entry_paths(ydl: YoutubeDL, entry: dict[str, Any]) -> list[Path]:
    """Returns existing downloaded file paths recorded on one yt-dlp entry dict."""
    requested_downloads = entry.get("requested_downloads")
    paths = (
        list(
            dict.fromkeys(
                Path(item["filepath"])
                for item in requested_downloads
                if isinstance(item, dict) and isinstance(item.get("filepath"), str)
            )
        )
        if isinstance(requested_downloads, list)
        else []
    )
    if not paths:
        paths = [Path(ydl.prepare_filename(entry))]
    return [path for path in paths if path.is_file()]


def _collect_downloaded_paths(ydl: YoutubeDL, info: dict[str, Any]) -> list[Path]:
    """Returns all existing downloaded file paths from a single-video or playlist info dict."""
    entries = info.get("entries")
    if entries is None:
        return _extract_entry_paths(ydl=ydl, entry=info)
    return list(
        dict.fromkeys(
            path
            for entry in entries
            if isinstance(entry, dict)
            for path in _extract_entry_paths(ydl=ydl, entry=entry)
        )
    )


def _playlist_total(info: dict[str, Any]) -> int:
    """Returns the full playlist entry count reported by yt-dlp, or 0 when absent."""
    raw_total = info.get("playlist_count")
    if isinstance(raw_total, int) and not isinstance(raw_total, bool) and raw_total > 0:
        return raw_total
    return 0


def _cleanup_new_entries(output_path: Path, existing_entries: set[Path]) -> None:
    """Removes files and directories created inside `output_path` during a rejected download."""
    if not output_path.is_dir():
        return
    for entry in output_path.iterdir():
        if entry in existing_entries:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(path=entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def _estimate_total_bytes(
    info: dict[str, Any],
    max_items: int,
    max_bytes: int,
    default_headers: dict[str, str],
    stop_signal: threading.Event | None,
) -> int:
    """Returns estimated bytes across entries up to `max_items`, short-circuiting past `max_bytes`."""
    raw_entries = info.get("entries")
    if raw_entries is not None:
        if not isinstance(raw_entries, list):
            raw_entries = list(raw_entries)
            info["entries"] = raw_entries
        targets = [entry for entry in raw_entries if isinstance(entry, dict)][:max_items]
    else:
        targets = [info]

    total_estimated = 0
    with Session() as session:
        for entry in targets:
            estimated = _estimate_entry_bytes(
                entry=entry,
                default_headers=default_headers,
                session=session,
                stop_signal=stop_signal,
                max_bytes=max_bytes - total_estimated,
            )
            if estimated is None:
                continue
            total_estimated += estimated
            if total_estimated > max_bytes:
                return total_estimated
    return total_estimated


def _build_progress_hooks(
    url: str, stop_signal: threading.Event | None, max_bytes: int | None
) -> list[Callable[[dict[str, Any]], None]]:
    """Builds the yt-dlp progress hooks needed for stop signals and streaming byte caps."""
    hooks: list[Callable[[dict[str, Any]], None]] = []
    if stop_signal is not None:

        def _abort_if_stopped(_progress: dict[str, Any]) -> None:
            if stop_signal.is_set():
                raise DownloadStoppedError(f"download stopped for {url}")

        hooks.append(_abort_if_stopped)
    if max_bytes is not None:
        hooks.append(_build_byte_cap_hook(max_bytes=max_bytes, url=url))
    return hooks


class VideoDownloader(PlatformDownloader):
    """Downloads videos with yt-dlp and local project defaults."""

    output_folder: str = Field(..., description="Directory where downloaded files are written.")

    # Prefers separate video+audio with fallbacks to muxed or video-only streams; trailing
    # `/best/bestvideo*` lets height-less MP4s (e.g. Twitter GIFs) match quality presets.
    quality_formats: ClassVar[dict[VideoQuality, str]] = {
        "best": "bestvideo*+bestaudio/best/bestvideo*",
        "high": "bestvideo[height<=1080][fps<=60]+bestaudio/best[height<=1080][fps<=60]/best[height<=1080]/best/bestvideo*",
        "medium": "bestvideo[height<=720][fps<=60]+bestaudio/best[height<=720][fps<=60]/best[height<=720]/best/bestvideo*",
        "low": "bestvideo[height<=480]+bestaudio/best[height<=480]/best[height<=480]/best/bestvideo*",
    }

    def _default_http_headers(self) -> dict[str, str]:
        """Returns default HTTP headers for requests."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
        }

    def _resolve_facebook_share_url(self, url: str) -> str:
        """Follows redirects for `facebook.com/share/...` links to return the final target URL."""
        with Session() as session:
            for method_name in ("head", "get"):
                request_method = getattr(session, method_name)
                try:
                    # Stream GET so redirects resolve without downloading the final page body.
                    response = request_method(
                        url,
                        allow_redirects=True,
                        headers=BROWSER_HEADERS,
                        timeout=SHARE_RESOLVE_TIMEOUT_SECONDS,
                        stream=True,
                    )
                except RequestException:
                    continue

                final_url = response.url
                response.close()
                if (
                    final_url
                    and final_url != url
                    and not urlparse(url=final_url).path.lower().startswith(_LOGIN_WALL_PATHS)
                ):
                    return final_url

        return url

    def _convert_facebook_url(self, url: str, _resolve_share: bool = True) -> str:
        """Returns `url` rewritten for yt-dlp from `/watch`, `/share/...`, or post URLs."""
        parsed = urlparse(url=url)

        if not host_matches_domain(host=normalized_host(url=url), domain="facebook.com"):
            return url

        if _resolve_share and parsed.path.startswith("/share/"):
            resolved_url = self._resolve_facebook_share_url(url=url)
            if resolved_url != url:
                return self._convert_facebook_url(url=resolved_url, _resolve_share=False)

        if parsed.path == "/watch":
            query_params = parse_qs(qs=parsed.query)
            video_id = query_params.get("v", [None])[0]

            if video_id:
                return f"https://www.facebook.com/reel/{video_id}"

        # Skip direct `/videos/` permalinks: yt-dlp reads them directly and `FacebookDownloader`
        # carries no logged-out payload for them.
        if not re.match(pattern=r"^/[^/]+/videos/", string=parsed.path) and is_facebook_post_url(
            url=url
        ):
            try:
                conversation = FacebookDownloader().parse_metadata(url=url)
            except Exception as error:
                # Broad on purpose: post-page resolution is best-effort; any parser or network
                # failure falls back to handing the original URL to yt-dlp.
                logfire.warn(
                    "Facebook post video resolution failed; falling back to original URL",
                    url=url,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )
                return url
            target = conversation.target
            if target is not None and target.video_urls:
                return target.video_urls[0]

        return url

    def _run_download(
        self,
        params: dict[str, Any],
        url: str,
        max_items: int,
        max_bytes: int | None,
        stop_signal: threading.Event | None,
    ) -> DownloadResult:
        """Executes yt-dlp (with optional pre-download probe) and validates the resulting files."""
        with YoutubeDL(params=params) as ydl:
            probed_total_items = 0
            if max_bytes is not None:
                probe_info = ydl.extract_info(url=url, download=False)
                if probe_info is None:
                    msg = f"yt-dlp returned no metadata for {url}"
                    raise RuntimeError(msg)
                probed_total_items = _playlist_total(info=probe_info)
                estimated = _estimate_total_bytes(
                    info=probe_info,
                    max_items=max_items,
                    max_bytes=max_bytes,
                    default_headers=params["http_headers"],
                    stop_signal=stop_signal,
                )
                if estimated > max_bytes:
                    raise DownloadTooLargeError(
                        f"Media at {url} estimates {estimated} bytes, over the {max_bytes} byte cap"
                    )
                info = ydl.process_ie_result(ie_result=probe_info, download=True)
            else:
                info = ydl.extract_info(url=url, download=True)

            if info is None:
                msg = f"yt-dlp returned no metadata for {url}"
                raise RuntimeError(msg)

            filenames = _collect_downloaded_paths(ydl=ydl, info=info)
            if not filenames:
                msg = f"yt-dlp produced no downloaded files for {url}"
                raise RuntimeError(msg)

            if max_bytes is not None:
                actual_bytes = sum(path.stat().st_size for path in filenames)
                if actual_bytes > max_bytes:
                    raise DownloadTooLargeError(
                        f"Media at {url} is {actual_bytes} bytes, over the {max_bytes} byte cap"
                    )

            total_items = _playlist_total(info=info) or probed_total_items
            return DownloadResult(filenames=filenames, total_items=total_items)

    def get_params(self, quality: VideoQuality, url: str) -> dict[str, Any]:
        """Returns the yt-dlp configuration parameters for `quality` and `url`.

        Args:
            quality (VideoQuality): The requested quality preset.
            url (str): The URL to be read, which decides site-specific headers (e.g., bilibili).
        """
        output_path = Path(self.output_folder)

        # Match the parsed host, never a substring, before sending a Bilibili Referer.
        http_headers = self._default_http_headers()
        if host_matches_domain(host=normalized_host(url=url), domain="bilibili.com"):
            http_headers["Referer"] = "https://www.bilibili.com"

        return {
            "format": self.quality_formats[quality],
            "outtmpl": f"{output_path.as_posix()}/%(id)s.%(ext)s",
            "quiet": True,
            "no_warnings": False,
            "continuedl": True,
            "noplaylist": True,
            "restrictfilenames": True,
            "writeinfojson": False,
            "writedescription": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ignoreerrors": False,
            "retries": YTDLP_RETRIES,
            "fragment_retries": YTDLP_RETRIES,
            "merge_output_format": "mp4",
            "http_headers": http_headers,
            "socket_timeout": YTDLP_SOCKET_TIMEOUT_SECONDS,
            "extractor_retries": YTDLP_RETRIES,
            "geo_bypass": True,
        }

    def download(
        self,
        url: str,
        quality: VideoQuality = "best",
        stop_signal: threading.Event | None = None,
        max_items: int = 1,
        max_bytes: int | None = None,
    ) -> DownloadResult:
        """Downloads one or more videos from `url` and returns their local file paths.

        Args:
            url (str): The URL of the video or playlist/carousel to download.
            quality (VideoQuality): The requested quality preset.
            stop_signal (threading.Event | None): Optional event checked at every yt-dlp progress
                tick to abort the blocking worker thread.
            max_items (int): Maximum number of playlist or carousel entries to download.
            max_bytes (int | None): Optional cumulative byte cap enforced via pre-download
                estimation, streaming progress hooks, and final disk size.

        Raises:
            ValueError: When `max_items` is less than 1.
            DownloadStoppedError: When `stop_signal` is set mid-download.
            DownloadTooLargeError: When estimated, streamed, or final bytes exceed `max_bytes`.
            RuntimeError: When yt-dlp returns no metadata or produces no files on disk.
        """
        if max_items < 1:
            msg = f"max_items must be at least 1, got {max_items}"
            raise ValueError(msg)

        url = self._convert_facebook_url(url=url)
        params = self.get_params(quality=quality, url=url)
        params["playlist_items"] = f"1-{max_items}"
        progress_hooks = _build_progress_hooks(
            url=url, stop_signal=stop_signal, max_bytes=max_bytes
        )
        if progress_hooks:
            params["progress_hooks"] = progress_hooks

        output_path = Path(self.output_folder)
        existing_entries = set(output_path.iterdir()) if output_path.is_dir() else set()
        try:
            return self._run_download(
                params=params,
                url=url,
                max_items=max_items,
                max_bytes=max_bytes,
                stop_signal=stop_signal,
            )
        except DownloadTooLargeError:
            _cleanup_new_entries(output_path=output_path, existing_entries=existing_entries)
            raise

    def parse_metadata(self, url: str) -> VideoMetadata:
        """Returns a video's metadata via yt-dlp without downloading media.

        Args:
            url (str): The URL of the video to inspect.

        Raises:
            RuntimeError: When yt-dlp returns no metadata for the URL.
        """
        url = self._convert_facebook_url(url=url)
        params = self.get_params(quality="best", url=url)
        # Keep playlist-shaped pages to one request instead of probing every entry.
        params.update({"simulate": True, "skip_download": True, "extract_flat": "in_playlist"})
        with YoutubeDL(params=params) as ydl:
            info = ydl.extract_info(url=url, download=False)
        if info is None:
            msg = f"yt-dlp returned no metadata for {url}"
            raise RuntimeError(msg)
        # Preserve the page's own URL before unwrapping a playlist entry.
        page_url = str(info.get("webpage_url") or "")
        # Unwrap multi-part/playlist pages to their first entry and flag `from_playlist`.
        from_playlist = False
        entries = info.get("entries")
        if entries:
            from_playlist = True
            info = next((entry for entry in entries if entry), info)
        return VideoMetadata(
            title=str(info.get("title") or ""),
            uploader=str(info.get("uploader") or ""),
            description=str(info.get("description") or ""),
            duration_seconds=float(info.get("duration") or 0.0),
            webpage_url=page_url or str(info.get("webpage_url") or ""),
            is_live=bool(info.get("is_live") or False),
            from_playlist=from_playlist,
        )


def _retrieve_quietly(task: "asyncio.Task[DownloadResult]") -> None:
    """Retrieves an abandoned task's outcome so asyncio never logs it as never-retrieved."""
    if not task.cancelled():
        task.exception()


async def download_with_stop_signal(
    downloader: VideoDownloader,
    url: str,
    quality: VideoQuality,
    max_items: int = 1,
    max_bytes: int | None = None,
) -> DownloadResult:
    """Returns the finished download, delivering a stop signal and bounded join on cancellation.

    Signals `stop_signal` on interruption and waits up to `DOWNLOAD_STOP_JOIN_SECONDS` for the
    worker thread to exit so scratch-directory cleanup does not race a live yt-dlp write.

    Args:
        downloader (VideoDownloader): The downloader to run, already pointed at its scratch directory.
        url (str): The URL to download.
        quality (VideoQuality): The requested quality preset.
        max_items (int): Maximum number of playlist or carousel entries to download.
        max_bytes (int | None): Optional cumulative byte cap enforced during download.
    """
    stop_signal = threading.Event()
    download_task = asyncio.create_task(
        coro=asyncio.to_thread(
            downloader.download,
            url=url,
            quality=quality,
            stop_signal=stop_signal,
            max_items=max_items,
            max_bytes=max_bytes,
        )
    )
    download_task.add_done_callback(_retrieve_quietly)
    try:
        return await asyncio.shield(download_task)
    except BaseException:
        stop_signal.set()
        done, _pending = await asyncio.wait({download_task}, timeout=DOWNLOAD_STOP_JOIN_SECONDS)
        if done:
            with contextlib.suppress(BaseException):
                download_task.result()
        else:
            logfire.warn(
                "Download worker ignored the stop signal within the join window",
                url=url,
                join_seconds=DOWNLOAD_STOP_JOIN_SECONDS,
            )
        raise
