"""One place that decides how a piece of media reaches Discord: attach, host, or drop.

Feature code stays pure: a generator/downloader produces media bytes (or a file on disk)
plus a filename, wraps it in a `MediaItem`, and hands a list of them to
`MediaDeliveryPlanner.plan`. The planner returns a `MediaPlan` splitting them into native
Discord attachments, externally-hosted public URLs (for anything too big to upload), and
items that could not be delivered, so every call site shares one attach-vs-host-vs-drop
decision and differs only in how it sends the result.

This module also owns the two low-level concerns the decision needs: the destination's real
upload ceiling (`upload_limit_for`) and the external static host that turns oversized media
into a public URL (`MediaHostingService`, env-backed via `MediaHostingConfig`). The host is
best-effort: every method returns None (never raises) when hosting is disabled, unconfigured,
handed a non-allowlisted suffix, or fails to write, so a `MEDIA_HOSTING_ENABLED=false` (or
unconfigured) deployment degrades to its prior, host-free behavior at every call site.
"""

from io import BytesIO
import shutil
from typing import TYPE_CHECKING
import asyncio
from pathlib import Path
import secrets

import dotenv
import logfire
from nextcord import File
from pydantic import Field, BaseModel, ConfigDict, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from nextcord import Guild

dotenv.load_dotenv()

# Discord lowered the non-Nitro base upload limit to 10 MiB in 2024; a guild-less context
# (DM) has no boost-tier table to consult, so it falls back to this base.
DEFAULT_NON_NITRO_UPLOAD_LIMIT = 10 * 1024 * 1024

# Discord caps one message at 10 attachments; the combined media edit is clamped to this.
DISCORD_ATTACHMENT_LIMIT = 10
# Discord measures the full multipart request body, not just the file bytes, so a combined
# attach (or one riding with embeds) is held this far under the limit; without it a set whose
# per-file sizes each pass can still 400 the final send.
MEDIA_ENVELOPE_MARGIN = 1024 * 1024

# Mirrors the nginx static host's extension allowlist (media.mai0313.com). A file whose suffix
# is not here is refused rather than written, because the host returns 404 for it. Keep this in
# sync with the server block's `location ~* \.(...)$` allowlist.
_ALLOWED_SUFFIXES = frozenset({
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".ico",
    ".avif",
    ".mp4",
    ".m4v",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
    ".ogv",
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".weba",
    ".txt",
    ".md",
    ".csv",
    ".log",
    ".json",
    ".yaml",
    ".ini",
})


def upload_limit_for(guild: "Guild | None") -> int:
    """Returns the destination's real attachment upload ceiling in bytes.

    A boosted guild's 50/100 MiB is honored via nextcord's `filesize_limit` (its boost-tier
    table lookup keyed on `premium_tier`); a DM has no guild to query, so it falls back to
    Discord's non-Nitro base of 10 MiB.

    Args:
        guild: The destination guild, or None for a DM.

    Returns:
        The maximum attachment size in bytes for that destination.
    """
    return guild.filesize_limit if guild is not None else DEFAULT_NON_NITRO_UPLOAD_LIMIT


def _normalize_suffix(suffix: str) -> str | None:
    """Lowercases a file suffix and returns it only if the host would serve it, else None.

    `.JPG` normalizes to `.jpg`; a non-allowlisted suffix (e.g. `.aiff`, which the music
    renderer can emit) returns None so the caller never produces a URL that 404s.
    """
    normalized = suffix.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized if normalized in _ALLOWED_SUFFIXES else None


class MediaHostingConfig(BaseSettings):
    """Configuration for the external media host, read from environment variables.

    Attributes:
        enabled: Kill-switch; when false the fallback is inert and oversized media degrades
            exactly as before.
        base_url: Public base URL the host serves from (e.g. https://media.mai0313.com).
        serve_dir: In-container directory (bind-mounted from the host) files are written into;
            nginx serves the same files from the host path.
    """

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(
        default=True,
        description="Whether oversized media may be hosted externally and linked.",
        validation_alias=AliasChoices("MEDIA_HOSTING_ENABLED"),
    )
    base_url: str = Field(
        default="",
        description="Public base URL the external static host serves from.",
        examples=["https://media.mai0313.com"],
        validation_alias=AliasChoices("MEDIA_HOSTING_BASE_URL"),
    )
    serve_dir: str = Field(
        default="",
        description="In-container directory (bind-mounted from the host) hosted files are written into.",
        examples=["/mnt/share/media"],
        validation_alias=AliasChoices("MEDIA_HOSTING_SERVE_DIR"),
    )

    @property
    def available(self) -> bool:
        """Whether hosting can actually run: enabled AND both base URL and serve dir are set."""
        return self.enabled and bool(self.base_url.strip()) and bool(self.serve_dir.strip())


class MediaHostingService(BaseModel):
    """Writes oversized media into the served directory and returns its public URL.

    Attributes:
        config: The media-hosting configuration backing this service.
    """

    config: MediaHostingConfig = Field(
        ..., description="The media-hosting configuration backing this service."
    )

    def model_post_init(self, _context: object, /) -> None:
        """Logs the resolved serve dir once at construction so a misconfig fails loud.

        `available` only checks for non-empty config; a typo'd or unmounted serve dir would
        write files nowhere nginx can see and every URL would 404 silently. Surfacing the
        resolved path (and warning when hosting is on but the dir is missing) makes that loud.
        """
        if not self.config.available:
            return
        serve = Path(self.config.serve_dir)
        if serve.is_dir():
            logfire.info(
                "Media hosting enabled", base_url=self.config.base_url, serve_dir=str(serve)
            )
        else:
            logfire.warn(
                "Media hosting serve dir does not exist; hosted URLs will 404",
                serve_dir=str(serve),
            )

    def _public_url(self, name: str) -> str:
        """Joins the configured base URL with a served filename."""
        return f"{self.config.base_url.rstrip('/')}/{name}"

    def _destination(self, ext: str) -> Path:
        """Builds an unguessable destination path inside the serve dir, creating it if needed."""
        serve = Path(self.config.serve_dir)
        serve.mkdir(parents=True, exist_ok=True)
        return serve / f"{secrets.token_urlsafe(16)}{ext}"

    def publish_bytes(self, data: bytes, suffix: str) -> str | None:
        """Writes bytes into the served dir under an unguessable name; returns the URL or None.

        Args:
            data: The media bytes to host.
            suffix: The intended file extension (e.g. ".wav"); refused if not allowlisted.

        Returns:
            The public URL, or None when hosting is unavailable / the suffix is refused / the
            write fails.
        """
        if not self.config.available:
            return None
        ext = _normalize_suffix(suffix=suffix)
        if ext is None:
            logfire.debug("Media hosting refused a non-allowlisted suffix", suffix=suffix)
            return None
        try:
            destination = self._destination(ext=ext)
            destination.write_bytes(data)
        except Exception:
            logfire.warn("Failed to host media bytes", _exc_info=True)
            return None
        return self._public_url(name=destination.name)

    def publish_path(self, file_path: Path) -> str | None:
        """Moves an existing file into the served dir; returns the URL or None.

        On a non-allowlisted suffix or any failure the file is left in place for the caller's
        own cleanup. The serve dir is a bind-mount, so it may be a different filesystem than
        the temp file; `shutil.move` falls back to copy+unlink across devices where
        `Path.rename` would raise EXDEV.

        Args:
            file_path: The existing file to move into the served dir.

        Returns:
            The public URL, or None when hosting is unavailable / the suffix is refused / the
            move fails.
        """
        if not self.config.available:
            return None
        ext = _normalize_suffix(suffix=file_path.suffix)
        if ext is None:
            logfire.debug(
                "Media hosting refused a non-allowlisted suffix", suffix=file_path.suffix
            )
            return None
        try:
            destination = self._destination(ext=ext)
            shutil.move(str(file_path), str(destination))
        except Exception:
            logfire.warn("Failed to host media file", _exc_info=True)
            return None
        return self._public_url(name=destination.name)


class MediaItem(BaseModel):
    """One built media payload awaiting the attach-vs-host-vs-drop decision.

    The source is either in-memory bytes (a synthesized voice/music clip, a generated or
    inline image/video, a research report) or an on-disk file (a yt-dlp download, a Threads
    video) that is hosted by move rather than re-read into memory. `isinstance(source, bytes)`
    selects the right primitive throughout, so there is no "neither/both" state to validate.

    Attributes:
        source: The media bytes, or the path to the on-disk file.
        filename: Attachment filename carrying the allowlisted suffix; drives both the Discord
            attachment name and the hosted-suffix allowlist check.
    """

    model_config = ConfigDict(frozen=True)

    source: bytes | Path = Field(
        ..., description="The media bytes, or the path to the on-disk file."
    )
    filename: str = Field(..., description="Attachment filename carrying the allowlisted suffix.")

    @property
    def size(self) -> int:
        """Byte size: len for in-memory bytes, st_size for a path (never reads the file)."""
        if isinstance(self.source, bytes):
            return len(self.source)
        return self.source.stat().st_size

    def to_file(self) -> File:
        """Builds a fresh single-use nextcord File from the source bytes or on-disk path."""
        if isinstance(self.source, bytes):
            return File(fp=BytesIO(self.source), filename=self.filename)
        return File(fp=str(self.source), filename=self.filename)

    def host_with(self, *, service: MediaHostingService) -> str | None:
        """Hosts this item via the right primitive (publish_bytes vs publish_path); blocking.

        The suffix for in-memory bytes comes from the filename so the host allowlist (and thus
        inline playability of the link) is honored.
        """
        if isinstance(self.source, bytes):
            return service.publish_bytes(data=self.source, suffix=Path(self.filename).suffix)
        return service.publish_path(file_path=self.source)


class MediaPlan(BaseModel):
    """The attach-vs-host-vs-drop outcome for one set of media items.

    Attributes:
        native: Items small enough to ride as native attachments, count-clamped to
            DISCORD_ATTACHMENT_LIMIT, kept in input order (callers lead with voice/music).
        hosted_urls: Public URLs for items hosted externally because they were individually
            oversize or peeled to fit the combined multipart body.
        dropped_items: Items lost to a hosting failure (off / unavailable / non-allowlisted /
            write-fail) or the attachment-count clamp; the caller applies its own policy
            (drop + hint, raise, a failure message, or a refusal) so this stays faithful to
            each site's pre-hosting behavior when hosting is off.
    """

    native: list[MediaItem] = Field(
        default_factory=list, description="Items to attach natively, count-clamped, in order."
    )
    hosted_urls: list[str] = Field(
        default_factory=list, description="Public URLs for hosted (oversize / peeled) items."
    )
    dropped_items: list[MediaItem] = Field(
        default_factory=list, description="Items lost to a hosting failure or the count clamp."
    )


class MediaDeliveryPlanner(BaseModel):
    """Decides which media attach natively, which host to a URL, and which drop.

    Generalizes one decision (originally the QA streamer's media partition) to bytes-backed
    and path-backed items and to the single-item case, so every call site shares it and
    differs only in how it sends the result. The host writes run off the event loop.

    Attributes:
        media_hosting: External host for oversize media; its `available` gate makes the URL
            fallback inert (every oversize item drops) when hosting is unconfigured / off.
    """

    media_hosting: MediaHostingService = Field(
        ..., description="External host for oversize media."
    )

    async def _host(self, *, item: MediaItem) -> str | None:
        """Runs one blocking host write off the event loop; None when unavailable/refused/failed."""
        return await asyncio.to_thread(item.host_with, service=self.media_hosting)

    async def plan(
        self,
        *,
        items: list[MediaItem],
        upload_limit: int,
        envelope_margin: int = 0,
        attachment_limit: int = DISCORD_ATTACHMENT_LIMIT,
    ) -> MediaPlan:
        """Splits items into native attachments, hosted URLs, and dropped items.

        (a) Each individually-oversize item is hosted (or dropped when hosting is off / fails),
        concurrently. (b) The largest remaining are then peeled to hosted URLs until the
        combined body clears `upload_limit - envelope_margin`. (c) The native list is finally
        clamped to `attachment_limit`, the overflow dropped. Input order is preserved in
        `native`, so a caller leading with voice/music keeps a trailing image as the drop.

        Args:
            items: The built media items to deliver, in caller-preferred order.
            upload_limit: The destination's attachment ceiling (see `upload_limit_for`).
            envelope_margin: Headroom held back for the multipart body / embeds JSON.
            attachment_limit: Max native attachments Discord allows in one message.

        Returns:
            A `MediaPlan` partitioning the items.
        """
        sizes = {id(item): item.size for item in items}  # stat each path exactly once
        hosted_urls: list[str] = []
        dropped: list[MediaItem] = []

        # (a) Per-item: host every individually-oversize item concurrently (independent writes),
        # keeping the rest as native-attach candidates.
        fitting = [item for item in items if sizes[id(item)] <= upload_limit]
        oversize = [item for item in items if sizes[id(item)] > upload_limit]
        if oversize:
            urls = await asyncio.gather(*(self._host(item=item) for item in oversize))
            for item, url in zip(oversize, urls, strict=True):
                if url is not None:
                    hosted_urls.append(url)
                else:
                    dropped.append(item)

        # (b) Combined total: peel the largest remaining to a URL until the multipart body fits.
        total = sum(sizes[id(item)] for item in fitting)
        while fitting and total + envelope_margin > upload_limit:
            largest = max(fitting, key=lambda item: sizes[id(item)])
            fitting.remove(largest)
            total -= sizes[id(largest)]
            url = await self._host(item=largest)
            if url is not None:
                hosted_urls.append(url)
            else:
                dropped.append(largest)

        # (c) Discord's per-message attachment cap; the overflow (trailing, in order) drops.
        if len(fitting) > attachment_limit:
            dropped.extend(fitting[attachment_limit:])
            fitting = fitting[:attachment_limit]
        return MediaPlan(native=fitting, hosted_urls=hosted_urls, dropped_items=dropped)


__all__ = [
    "DEFAULT_NON_NITRO_UPLOAD_LIMIT",
    "DISCORD_ATTACHMENT_LIMIT",
    "MEDIA_ENVELOPE_MARGIN",
    "MediaDeliveryPlanner",
    "MediaHostingConfig",
    "MediaHostingService",
    "MediaItem",
    "MediaPlan",
    "upload_limit_for",
]
