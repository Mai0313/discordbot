"""Attach-vs-host-vs-drop planning, Discord upload-limit lookup, and external static media hosting.

Callers wrap outputs in `MediaItem` and pass them to `MediaDeliveryPlanner.plan` to partition
into native attachments, hosted public URLs, dropped items, and clamped items.

`MediaHostingService` is best-effort and returns None when disabled, unconfigured, given a
non-allowlisted suffix, or on write failure. Every call site must stay host-free under
`MEDIA_HOSTING_ENABLED=false` or an unconfigured host.
"""

from io import BytesIO
import os
import re
import time
import shutil
from typing import TYPE_CHECKING, Any, Unpack, TypedDict
import asyncio
import hashlib
from pathlib import Path
import secrets
import threading
import contextlib

import logfire
from nextcord import File, Client, Interaction
from pydantic import Field, BaseModel, ConfigDict, AliasChoices
from pydantic_settings import BaseSettings

from discordbot.utils.discord_embeds import DISCORD_ATTACHMENT_LIMIT

if TYPE_CHECKING:
    from nextcord import Guild
    from nextcord.state import ConnectionState
    from nextcord.types.interactions import Interaction as InteractionPayload

# Discord's non-Nitro base upload limit (20 MiB as of 2026-08-13); fallback for guild-less DMs.
DEFAULT_NON_NITRO_UPLOAD_LIMIT = 20 * 1024 * 1024

# Multipart/embed body headroom subtracted from the upload limit on combined sends.
MEDIA_ENVELOPE_MARGIN = 1024 * 1024

# Must stay in sync with media.mai0313.com nginx allowlist; unlisted suffixes 404.
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

# Content-addressed stem: first 32 hex chars (128 bits) of SHA-256.
_HASH_HEX_LEN = 32
_HASH_CHUNK_BYTES = 1024 * 1024
# Bot-scoped temp prefix for atomic os.replace writes; stale-temp sweep reaps only matching names.
_TEMP_PREFIX = ".mediahost-tmp-"
_STALE_TEMP_SECONDS = 300.0
# Grace window protecting newly published files from concurrent size-cap eviction.
_EVICTION_GRACE_SECONDS = 300.0

# Matches only files written by this service (32-hex stem + allowlisted suffix), excluding others.
_HOSTED_NAME_RE = re.compile(
    f"[0-9a-f]{{{_HASH_HEX_LEN}}}(?:"
    + "|".join(re.escape(s) for s in sorted(_ALLOWED_SUFFIXES))
    + ")"
)

# Matches only this service's temp files so foreign `.tmp-*` files are never reaped.
_TEMP_NAME_RE = re.compile(re.escape(_TEMP_PREFIX) + r"[A-Za-z0-9_-]+")

# Serializes serve-dir scan/mutate sections across `asyncio.to_thread` workers; writes stay free.
_SERVE_DIR_LOCK = threading.Lock()


class AttachmentLimitInteraction(Interaction[Client]):
    """An interaction that keeps the `attachment_size_limit` nextcord drops while parsing.

    Discord computes it per interaction as the larger of the invoking user's Nitro allowance and
    the guild's boost tier, and it holds only for what is sent through this interaction.
    `DiscordBot.get_interaction` builds every interaction as this class.
    """

    __slots__ = ("attachment_size_limit",)

    def __init__(self, data: "InteractionPayload", state: "ConnectionState") -> None:
        """Parses the payload as nextcord does, then keeps its attachment size limit.

        Args:
            data (InteractionPayload): Raw Discord interaction payload.
            state (ConnectionState): Gateway connection state.
        """
        super().__init__(data=data, state=state)
        self.attachment_size_limit: int | None = data.get("attachment_size_limit")


def upload_limit_for(guild: "Guild | None", interaction: Interaction[Any] | None = None) -> int:
    """Returns the destination's attachment upload ceiling in bytes.

    Uses `interaction.attachment_size_limit` when available (reflects Nitro allowance), else
    `guild.filesize_limit`, else `DEFAULT_NON_NITRO_UPLOAD_LIMIT` for DMs.

    Args:
        guild (Guild | None): Destination guild, or None for a DM.
        interaction (Interaction[Any] | None): Interaction carrying the reply, or None for channel
            sends.
    """
    if (
        isinstance(interaction, AttachmentLimitInteraction)
        and interaction.attachment_size_limit is not None
    ):
        return interaction.attachment_size_limit
    # Trust nextcord's static table (even where tier 0/1 reports 25 MiB vs 20 MiB) so upgrades win.
    return guild.filesize_limit if guild is not None else DEFAULT_NON_NITRO_UPLOAD_LIMIT


def _normalize_suffix(suffix: str) -> str | None:
    """Returns lowercased `suffix` with leading dot if allowlisted by the host, else None."""
    normalized = suffix.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized if normalized in _ALLOWED_SUFFIXES else None


def _hash_bytes(data: bytes) -> str:
    """Returns the 128-bit SHA-256 hex stem for `data`."""
    return hashlib.sha256(data).hexdigest()[:_HASH_HEX_LEN]


def _hash_file(path: Path) -> str:
    """Returns the 128-bit SHA-256 hex stem for `path`, streamed in chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:_HASH_HEX_LEN]


class MediaHostingConfig(BaseSettings):
    """Configuration for the external media host, read from environment variables."""

    enabled: bool = Field(
        default=True,
        description=(
            "Kill-switch for hosting oversized media externally and linking it; when false the "
            "URL fallback is inert and every oversize item takes the call site's host-free path."
        ),
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
        description=(
            "In-container directory (bind-mounted from the host) hosted files are written into; "
            "nginx serves those same files from the host path."
        ),
        examples=["/mnt/share/media"],
        validation_alias=AliasChoices("MEDIA_HOSTING_SERVE_DIR"),
    )
    max_bytes: int = Field(
        default=8 * 1024**3,
        description="Soft cap on total hosted bytes; oldest files are evicted past it (<=0 disables).",
        validation_alias=AliasChoices("MEDIA_HOSTING_MAX_BYTES"),
    )
    retention_hours: float = Field(
        default=168.0,
        description="Hosted files older than this many hours are reaped, cap or not (<=0 disables).",
        validation_alias=AliasChoices("MEDIA_HOSTING_RETENTION_HOURS"),
    )

    @property
    def available(self) -> bool:
        """Returns whether hosting can run: enabled and both base URL and serve dir are set."""
        return self.enabled and bool(self.base_url.strip()) and bool(self.serve_dir.strip())

    @property
    def cleanup_enabled(self) -> bool:
        """Returns whether the cleanup loop should run: hosting available and at least one cap is set."""
        return self.available and (self.max_bytes > 0 or self.retention_hours > 0)


class _HostedFile(BaseModel):
    """One file the hosting service itself wrote, as the serve-dir sweeps see it."""

    model_config = ConfigDict(frozen=True)

    mtime: float = Field(
        ..., description="Last-modified time in epoch seconds; the eviction and retention order."
    )
    size: int = Field(..., description="Size in bytes, counted against the hosting size cap.")
    path: str = Field(..., description="Path of the file inside the serve directory.")


class _HostedSource(TypedDict, total=False):
    """Typed logfire kwargs (`size_bytes` or `path`) for media-hosting refusal logs."""

    size_bytes: int
    path: str


class MediaHostingService(BaseModel):
    """Writes oversized media into the served directory and returns its public URL."""

    config: MediaHostingConfig = Field(
        ..., description="The media-hosting configuration backing this service."
    )

    def model_post_init(self, _context: object, /) -> None:
        """Logs the resolved serve directory at construction, warning if missing.

        Args:
            _context (object): Unused Pydantic validation context.
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
                "Media hosting serve dir is missing; media will fall back to host-free delivery",
                serve_dir=str(serve),
            )

    def _public_url(self, name: str) -> str:
        """Joins the configured base URL with a served filename."""
        return f"{self.config.base_url.rstrip('/')}/{name}"

    def _serve_dir(self) -> Path | None:
        """Returns the serve directory if it exists, else None; never creates it."""
        serve = Path(self.config.serve_dir)
        if serve.is_dir():
            return serve
        logfire.warn(
            "Media hosting serve dir is missing; falling back to host-free delivery",
            serve_dir=str(serve),
        )
        return None

    def _dedup_hit(self, serve: Path, name: str) -> str | None:
        """Refreshes mtime under `_SERVE_DIR_LOCK` and returns the URL if `name` exists."""
        final = serve / name
        with _SERVE_DIR_LOCK:
            if not final.exists():
                return None
            with contextlib.suppress(FileNotFoundError):
                os.utime(final)
            return self._public_url(name=name)

    def _finalize(self, serve: Path, name: str, tmp: Path) -> str:
        """Atomically replaces `tmp` onto `serve / name`, stamps mtime, and returns the URL."""
        final = serve / name
        os.replace(tmp, final)
        os.utime(final)
        return self._public_url(name=name)

    def _destination(
        self, suffix: str, **source: Unpack[_HostedSource]
    ) -> tuple[Path, str] | None:
        """Returns `(serve_dir, ext)` if hosting is active and `suffix` is allowed, else None."""
        if not self.config.available:
            return None
        ext = _normalize_suffix(suffix=suffix)
        if ext is None:
            logfire.warn(
                "Media hosting refused a non-allowlisted suffix; the item will be dropped",
                suffix=suffix,
                **source,
            )
            return None
        serve = self._serve_dir()
        if serve is None:
            return None
        return serve, ext

    def publish_bytes(self, data: bytes, suffix: str) -> str | None:
        """Returns the hosted public URL for `data`, or None on refusal or failure.

        Identical bytes deduplicate to the same name (refreshing mtime without rewriting); misses
        write via a sibling temp, atomically replace onto the final name, and enforce the size cap.

        Args:
            data (bytes): Media bytes to publish.
            suffix (str): File suffix determining the hosted extension.
        """
        destination = self._destination(suffix=suffix, size_bytes=len(data))
        if destination is None:
            return None
        serve, ext = destination
        name = f"{_hash_bytes(data=data)}{ext}"
        hit = self._dedup_hit(serve=serve, name=name)
        if hit is not None:
            return hit
        tmp = serve / f"{_TEMP_PREFIX}{secrets.token_urlsafe(8)}"
        try:
            tmp.write_bytes(data)
            url = self._finalize(serve=serve, name=name, tmp=tmp)
        # Broad on purpose: any write / filesystem error must degrade to a drop, never raise into
        # a reply pipeline.
        except Exception as exc:
            logfire.warn(
                "Failed to host media bytes",
                name=name,
                size_bytes=len(data),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            return None
        self.enforce_cap(now=time.time())
        return url

    def publish_path(self, file_path: Path) -> str | None:
        """Returns the hosted public URL for `file_path`, or None on refusal or failure.

        Hashes via streaming reads. On a dedup hit, leaves `file_path` in place; on a miss, copies
        to a serve-dir temp, atomically replaces onto the final name, unlinks `file_path`, and
        enforces the size cap.

        Args:
            file_path (Path): On-disk media file to publish.
        """
        destination = self._destination(suffix=file_path.suffix, path=str(file_path))
        if destination is None:
            return None
        serve, ext = destination
        try:
            name = f"{_hash_file(path=file_path)}{ext}"
        except OSError as exc:
            logfire.warn(
                "Failed to hash media file",
                path=str(file_path),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        hit = self._dedup_hit(serve=serve, name=name)
        if hit is not None:
            return hit
        tmp = serve / f"{_TEMP_PREFIX}{secrets.token_urlsafe(8)}"
        try:
            shutil.copy2(str(file_path), str(tmp))
        # Broad on purpose: any copy / filesystem error must degrade to a drop, never raise
        # into a reply pipeline.
        except Exception as exc:
            logfire.warn(
                "Failed to copy media into the serve dir",
                path=str(file_path),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            return None
        try:
            url = self._finalize(serve=serve, name=name, tmp=tmp)
        # Broad on purpose: same boundary; a half-finalized host must not surface a URL.
        except Exception as exc:
            logfire.warn(
                "Failed to finalize hosted media",
                name=name,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            return None
        # Published from here on: a failed source cleanup must not report the item as dropped.
        try:
            file_path.unlink(missing_ok=True)
        except OSError as exc:
            logfire.debug(
                "Hosted media published but the source file could not be removed",
                path=str(file_path),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        self.enforce_cap(now=time.time())
        return url

    def _scan_hosted(self, serve: Path) -> list[_HostedFile]:
        """Returns regular non-symlink files in `serve` matching `_HOSTED_NAME_RE`."""
        hosted: list[_HostedFile] = []
        with os.scandir(serve) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if _HOSTED_NAME_RE.fullmatch(entry.name) is None:
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                hosted.append(_HostedFile(mtime=stat.st_mtime, size=stat.st_size, path=entry.path))
        return hosted

    def enforce_cap(self, now: float) -> int:
        """Returns bytes freed after evicting oldest hosted files down to `max_bytes`.

        Counts and evicts only service-written files older than `_EVICTION_GRACE_SECONDS`.

        Args:
            now (float): Current epoch timestamp in seconds.
        """
        cap = self.config.max_bytes
        if cap <= 0:
            return 0
        serve = self._serve_dir()
        if serve is None:
            return 0
        freed = 0
        with _SERVE_DIR_LOCK:
            files = self._scan_hosted(serve=serve)
            total = sum(hosted.size for hosted in files)
            if total <= cap:
                return 0
            cutoff = now - _EVICTION_GRACE_SECONDS
            evictable = sorted(
                (hosted for hosted in files if hosted.mtime < cutoff), key=lambda f: f.mtime
            )
            for hosted in evictable:
                if total - freed <= cap:
                    break
                try:
                    os.unlink(hosted.path)
                    freed += hosted.size
                except FileNotFoundError:
                    freed += hosted.size
                except OSError:
                    logfire.warn("Failed to evict hosted media", path=hosted.path, _exc_info=True)
        if freed:
            logfire.info("Evicted hosted media over the size cap", freed_bytes=freed)
        return freed

    def cleanup_expired(self, now: float) -> int:
        """Returns the count of hosted files older than `retention_hours` that were deleted.

        Args:
            now (float): Current epoch timestamp in seconds.
        """
        retention = self.config.retention_hours
        if retention <= 0:
            return 0
        serve = self._serve_dir()
        if serve is None:
            return 0
        cutoff = now - retention * 3600.0
        deleted = 0
        with _SERVE_DIR_LOCK:
            for hosted in self._scan_hosted(serve=serve):
                if hosted.mtime >= cutoff:
                    continue
                try:
                    os.unlink(hosted.path)
                    deleted += 1
                except FileNotFoundError:
                    deleted += 1
                except OSError:
                    logfire.warn("Failed to reap expired media", path=hosted.path, _exc_info=True)
        if deleted:
            logfire.info("Reaped expired hosted media", deleted_count=deleted)
        return deleted

    def sweep_stale_temps(self, now: float) -> None:
        """Unlinks crash-left bot temp files matching `_TEMP_NAME_RE` past `_STALE_TEMP_SECONDS`.

        Args:
            now (float): Current epoch timestamp in seconds.
        """
        serve = self._serve_dir()
        if serve is None:
            return
        cutoff = now - _STALE_TEMP_SECONDS
        with _SERVE_DIR_LOCK, os.scandir(serve) as entries:
            for entry in entries:
                if _TEMP_NAME_RE.fullmatch(entry.name) is None:
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                try:
                    if entry.stat().st_mtime < cutoff:
                        os.unlink(entry.path)
                except OSError:
                    continue

    def run_maintenance(self, now: float) -> tuple[int, int]:
        """Returns `(deleted_count, freed_bytes)` after sweeping temps, expired files, and cap.

        Runs age expiration before size-cap eviction.

        Args:
            now (float): Current epoch timestamp in seconds.
        """
        self.sweep_stale_temps(now=now)
        deleted = self.cleanup_expired(now=now)
        freed = self.enforce_cap(now=now)
        return deleted, freed


class MediaItem(BaseModel):
    """One media payload (in-memory bytes or on-disk path) awaiting delivery planning."""

    model_config = ConfigDict(frozen=True)

    source: bytes | Path = Field(
        ..., description="The media bytes, or the path to the on-disk file."
    )
    filename: str = Field(
        ...,
        description=(
            "Attachment filename carrying the allowlisted suffix; it drives both the Discord "
            "attachment name and the hosted-suffix allowlist check."
        ),
    )

    @property
    def size(self) -> int:
        """Byte size: len for in-memory bytes, st_size for a path (never reads the file)."""
        if isinstance(self.source, bytes):
            return len(self.source)
        return self.source.stat().st_size

    def to_file(self) -> File:
        """Returns a fresh single-use nextcord File from the source bytes or on-disk path."""
        if isinstance(self.source, bytes):
            return File(fp=BytesIO(self.source), filename=self.filename)
        return File(fp=str(self.source), filename=self.filename)

    def host_with(self, service: MediaHostingService) -> str | None:
        """Returns the hosted public URL for this item via `service`, or None on failure.

        The suffix for in-memory bytes comes from `filename` so the host allowlist is honored.

        Args:
            service (MediaHostingService): Hosting service to publish through.
        """
        if isinstance(self.source, bytes):
            return service.publish_bytes(data=self.source, suffix=Path(self.filename).suffix)
        return service.publish_path(file_path=self.source)


class MediaPlan(BaseModel):
    """The attach-vs-host-vs-drop outcome for one set of media items."""

    native: list[MediaItem] = Field(
        default_factory=list,
        description="Items to attach natively, count-clamped, in the caller's input order.",
    )
    hosted_urls: list[str] = Field(
        default_factory=list,
        description=(
            "Public URLs for items hosted externally, either individually oversize or peeled to "
            "fit the combined multipart body."
        ),
    )
    dropped_items: list[MediaItem] = Field(
        default_factory=list,
        description=(
            "Items lost to a hosting failure; what happens to them is the caller's own policy, "
            "which with hosting off must be that site's host-free behavior."
        ),
    )
    clamped_items: list[MediaItem] = Field(
        default_factory=list,
        description=(
            "Items that fit on their own but fell past Discord's per-message attachment cap; "
            "never offered to the host, and what happens to them is the caller's own policy."
        ),
    )


class MediaDeliveryPlanner(BaseModel):
    """Partitions media items into native attachments, hosted URLs, dropped, and clamped items."""

    media_hosting: MediaHostingService = Field(
        ...,
        description=(
            "External host for oversize media; its `available` gate makes the URL fallback inert, "
            "dropping every oversize item, when hosting is off or unconfigured."
        ),
    )

    async def _host(self, item: MediaItem) -> str | None:
        """Runs one blocking host write off the event loop; None when unavailable/refused/failed."""
        return await asyncio.to_thread(item.host_with, service=self.media_hosting)

    async def plan(
        self,
        items: list[MediaItem],
        upload_limit: int,
        envelope_margin: int = 0,
        attachment_limit: int = DISCORD_ATTACHMENT_LIMIT,
    ) -> MediaPlan:
        """Returns a `MediaPlan` splitting `items` into native, hosted, dropped, and clamped.

        Hosts individually-oversize items concurrently (dropping on failure), clamps the remaining
        list to `attachment_limit` (setting trailing overflow aside unhosted), then peels the
        largest remaining items to hosted URLs until the sum fits within
        `upload_limit - envelope_margin`. Preserves input order in `native`.

        Args:
            items (list[MediaItem]): Built media items to deliver, in caller-preferred order.
            upload_limit (int): Destination attachment ceiling (see `upload_limit_for`).
            envelope_margin (int): Headroom held back for the multipart body / embeds JSON.
            attachment_limit (int): Remaining attachment slots on the message.
        """
        sizes = {id(item): item.size for item in items}  # stat each path exactly once
        hosted_urls: list[str] = []
        dropped: list[MediaItem] = []

        effective_limit = max(upload_limit - envelope_margin, 0)
        fitting = [item for item in items if sizes[id(item)] <= effective_limit]
        oversize = [item for item in items if sizes[id(item)] > effective_limit]
        if oversize:
            urls = await asyncio.gather(*(self._host(item=item) for item in oversize))
            for item, url in zip(oversize, urls, strict=True):
                if url is not None:
                    hosted_urls.append(url)
                else:
                    dropped.append(item)

        attachment_limit = max(attachment_limit, 0)
        # Clamp count before byte-peeling so trailing items drop before prioritized leading clips.
        clamped = fitting[attachment_limit:]
        fitting = fitting[:attachment_limit]

        total = sum(sizes[id(item)] for item in fitting)
        while fitting and total > effective_limit:
            largest = max(fitting, key=lambda item: sizes[id(item)])
            fitting.remove(largest)
            total -= sizes[id(largest)]
            url = await self._host(item=largest)
            if url is not None:
                hosted_urls.append(url)
            else:
                dropped.append(largest)
        return MediaPlan(
            native=fitting, hosted_urls=hosted_urls, dropped_items=dropped, clamped_items=clamped
        )


def build_media_delivery_planner() -> MediaDeliveryPlanner:
    """Returns the default MediaDeliveryPlanner wired to the env-configured external media host.

    `MediaHostingConfig` self-disables when the host is unconfigured, so a planner built here
    stays on the host-free path until hosting is set up.
    """
    return MediaDeliveryPlanner(media_hosting=MediaHostingService(config=MediaHostingConfig()))
