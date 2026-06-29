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
import os
import re
import time
import shutil
from typing import TYPE_CHECKING
import asyncio
import hashlib
from pathlib import Path
import secrets
import threading
import contextlib

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

# Hosted files are content-addressed: `<sha256(content)[:32]>.<allowlisted-ext>`. 128 bits makes a
# collision (~1e-23 at 100M files) a dedup false-positive at worst, never a security break.
_HASH_HEX_LEN = 32
_HASH_CHUNK_BYTES = 1024 * 1024
# In-progress writes land on a sibling temp then `os.replace` onto the final name (atomic on the
# serve filesystem), so the content-addressed name only ever appears with complete content. A crash
# leaves a stale temp the 32-hex reaper can't match, so the sweep reaps temps older than this. The
# prefix is bot-specific (and the sweep verifies the full name shape) so a foreign `.tmp-*` parked in
# the shared serve dir is never reaped.
_TEMP_PREFIX = ".mediahost-tmp-"
_STALE_TEMP_SECONDS = 300.0
# A freshly-hosted file (and every in-flight concurrent publish) is protected from size-cap eviction
# for this long, so one publisher never reaps another's just-returned-and-posted URL.
_EVICTION_GRACE_SECONDS = 300.0
# How often the media_cleanup cog runs the age+size+temp sweep (a backstop; each publish enforces the
# size cap eagerly). A module constant, not env: an operational cadence, and @tasks.loop wants it static.
MEDIA_CLEANUP_INTERVAL_HOURS = 6.0

# The cleanup reaper only ever deletes files the service itself wrote: a 32-hex stem plus an
# allowlisted suffix. Built from the single `_ALLOWED_SUFFIXES` source so the writer and reaper
# cannot drift; a foreign `access.log` / human-named `movie.mp4` never matches.
_HOSTED_NAME_RE = re.compile(
    f"[0-9a-f]{{{_HASH_HEX_LEN}}}(?:"
    + "|".join(re.escape(s) for s in sorted(_ALLOWED_SUFFIXES))
    + ")"
)

# The bot's own in-progress temp names (`_TEMP_PREFIX` + a token_urlsafe stem). The stale-temp sweep
# verifies this full shape so it only ever reaps temps the service itself wrote, never a foreign one.
_TEMP_NAME_RE = re.compile(re.escape(_TEMP_PREFIX) + r"[A-Za-z0-9_-]+")

# All directory scan+mutate critical sections take this module-level lock so the ~5 service
# instances (one per media cog) that share one serve dir never race each other; the multi-GB byte
# writes go to unique temp names OUTSIDE the lock and stay fully concurrent. It is a threading.Lock
# (not asyncio.Lock) because publish/cleanup run in `asyncio.to_thread` worker threads.
_SERVE_DIR_LOCK = threading.Lock()


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


def _hash_bytes(data: bytes) -> str:
    """The 128-bit hex content stem for in-memory bytes."""
    return hashlib.sha256(data).hexdigest()[:_HASH_HEX_LEN]


def _hash_file(path: Path) -> str:
    """Streams a file through sha256 in chunks (never loads it whole) and returns its hex stem.

    The path branch exists for the large-file case (multi-GB downloads), so hashing must never
    `read_bytes()` the whole file into memory.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:_HASH_HEX_LEN]


class MediaHostingConfig(BaseSettings):
    """Configuration for the external media host, read from environment variables.

    Attributes:
        enabled: Kill-switch; when false the fallback is inert and oversized media degrades
            exactly as before.
        base_url: Public base URL the host serves from (e.g. https://media.mai0313.com).
        serve_dir: In-container directory (bind-mounted from the host) files are written into;
            nginx serves the same files from the host path.
        max_bytes: Soft cap on total hosted bytes; the oldest files are evicted past it (<=0 disables).
        retention_hours: Hosted files older than this are reaped even under the cap (<=0 disables).
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
        """Whether hosting can actually run: enabled AND both base URL and serve dir are set."""
        return self.enabled and bool(self.base_url.strip()) and bool(self.serve_dir.strip())

    @property
    def cleanup_enabled(self) -> bool:
        """Whether the cleanup loop should run: hosting available AND at least one cap is set."""
        return self.available and (self.max_bytes > 0 or self.retention_hours > 0)


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
                "Media hosting serve dir is missing; media will fall back to host-free delivery",
                serve_dir=str(serve),
            )

    def _public_url(self, name: str) -> str:
        """Joins the configured base URL with a served filename."""
        return f"{self.config.base_url.rstrip('/')}/{name}"

    def _serve_dir(self) -> Path | None:
        """The serve dir if it exists, else None (the bot never creates it; a missing dir falls back).

        The serve dir is a host-provided bind mount; a container-local dir nginx cannot see would
        only 404, so a missing dir means hosting is inactive and the caller degrades to host-free.
        """
        serve = Path(self.config.serve_dir)
        if serve.is_dir():
            return serve
        logfire.warn(
            "Media hosting serve dir is missing; falling back to host-free delivery",
            serve_dir=str(serve),
        )
        return None

    def _dedup_hit(self, *, serve: Path, name: str) -> str | None:
        """If the content-addressed file already exists, refresh its mtime and return its URL.

        Holds the dir lock so the refresh (which keeps a re-hosted clip alive under both caps) never
        races the sweep deleting that exact file.
        """
        final = serve / name
        with _SERVE_DIR_LOCK:
            if not final.exists():
                return None
            with contextlib.suppress(FileNotFoundError):
                os.utime(final)
            return self._public_url(name=name)

    def _finalize(self, *, serve: Path, name: str, tmp: Path) -> str:
        """Atomically moves a written temp onto its content-addressed name and returns the URL.

        `os.replace` is atomic within the serve filesystem, so the final name only ever appears with
        complete content (a crash leaves a `.tmp-*` the sweep reaps, never a poison cache entry that
        dedup would serve forever). mtime is stamped to now so it means "last hosted" for both caps.
        """
        final = serve / name
        os.replace(tmp, final)
        os.utime(final)
        return self._public_url(name=name)

    def publish_bytes(self, data: bytes, suffix: str) -> str | None:
        """Hosts bytes under a content-addressed name (dedup); returns the URL or None.

        Identical bytes map to the same name, so a re-host refreshes the existing file and returns
        the same URL without rewriting; a miss writes to a sibling temp then `os.replace`s it onto
        the final name (atomic) and enforces the size cap.

        Returns:
            The public URL, or None when hosting is unavailable / the serve dir is missing / the
            suffix is refused / the write fails.
        """
        if not self.config.available:
            return None
        ext = _normalize_suffix(suffix=suffix)
        if ext is None:
            logfire.debug("Media hosting refused a non-allowlisted suffix", suffix=suffix)
            return None
        serve = self._serve_dir()
        if serve is None:
            return None
        name = f"{_hash_bytes(data)}{ext}"
        hit = self._dedup_hit(serve=serve, name=name)
        if hit is not None:
            return hit
        tmp = serve / f"{_TEMP_PREFIX}{secrets.token_urlsafe(8)}"
        try:
            tmp.write_bytes(data)
            url = self._finalize(serve=serve, name=name, tmp=tmp)
        except Exception:
            logfire.warn("Failed to host media bytes", _exc_info=True)
            with contextlib.suppress(Exception):
                tmp.unlink(missing_ok=True)
            return None
        self.enforce_cap(now=time.time())
        return url

    def publish_path(self, file_path: Path) -> str | None:  # noqa: PLR0911 -- best-effort short-circuit guards
        """Hosts an on-disk file under a content-addressed name (dedup); returns the URL or None.

        The file is hashed by a streaming read (never loaded whole into memory), so a multi-GB clip
        stays flat. On a dedup HIT the source is left in place for the caller's own cleanup; on a
        miss it is copied into a serve-dir temp, `os.replace`d onto the final name, and the source is
        unlinked. The size cap is enforced after a successful host.

        Returns:
            The public URL, or None when hosting is unavailable / the serve dir is missing / the
            suffix is refused / the move fails.
        """
        if not self.config.available:
            return None
        ext = _normalize_suffix(suffix=file_path.suffix)
        if ext is None:
            logfire.debug(
                "Media hosting refused a non-allowlisted suffix", suffix=file_path.suffix
            )
            return None
        serve = self._serve_dir()
        if serve is None:
            return None
        try:
            name = f"{_hash_file(file_path)}{ext}"
        except OSError:
            logfire.warn("Failed to hash media file", _exc_info=True)
            return None
        hit = self._dedup_hit(serve=serve, name=name)
        if hit is not None:
            return hit  # the source is left in place for the caller's own cleanup
        tmp = serve / f"{_TEMP_PREFIX}{secrets.token_urlsafe(8)}"
        try:
            shutil.copy2(str(file_path), str(tmp))
            url = self._finalize(serve=serve, name=name, tmp=tmp)
            file_path.unlink(missing_ok=True)
        except Exception:
            logfire.warn("Failed to host media file", _exc_info=True)
            with contextlib.suppress(Exception):
                tmp.unlink(missing_ok=True)
            return None
        self.enforce_cap(now=time.time())
        return url

    def _scan_hosted(self, *, serve: Path) -> list[tuple[float, int, str]]:
        """(mtime, size, path) for every file the service itself wrote (the reaper guard).

        Only a 32-hex stem + allowlisted suffix, regular files (not symlinks/dirs), non-recursive,
        so a foreign file in the serve dir (an nginx log, a parked clip) is never a candidate.
        """
        hosted: list[tuple[float, int, str]] = []
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
                hosted.append((stat.st_mtime, stat.st_size, entry.path))
        return hosted

    def enforce_cap(self, *, now: float) -> int:
        """Evicts oldest hosted files until total bytes <= max_bytes; returns the bytes freed.

        Only the service's own files count and are evictable. A file hosted within the grace window
        is protected (so a concurrent publisher's just-returned URL is never reaped), and a single
        delivered file larger than the cap is kept: the loop stops when no evictable candidate
        remains rather than reaping good recent files, leaving disk temporarily over cap.
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
            total = sum(size for _, size, _ in files)
            if total <= cap:
                return 0
            cutoff = now - _EVICTION_GRACE_SECONDS
            evictable = sorted((f for f in files if f[0] < cutoff), key=lambda f: f[0])
            for _mtime, size, path in evictable:
                if total - freed <= cap:
                    break
                try:
                    os.unlink(path)
                    freed += size
                except FileNotFoundError:
                    freed += size
                except OSError:
                    logfire.warn("Failed to evict hosted media", path=path, _exc_info=True)
        if freed:
            logfire.info("Evicted hosted media over the size cap", freed_bytes=freed)
        return freed

    def cleanup_expired(self, *, now: float) -> int:
        """Deletes hosted files older than retention_hours; returns the count deleted."""
        retention = self.config.retention_hours
        if retention <= 0:
            return 0
        serve = self._serve_dir()
        if serve is None:
            return 0
        cutoff = now - retention * 3600.0
        deleted = 0
        with _SERVE_DIR_LOCK:
            for mtime, _size, path in self._scan_hosted(serve=serve):
                if mtime >= cutoff:
                    continue
                try:
                    os.unlink(path)
                    deleted += 1
                except FileNotFoundError:
                    deleted += 1
                except OSError:
                    logfire.warn("Failed to reap expired media", path=path, _exc_info=True)
        if deleted:
            logfire.info("Reaped expired hosted media", deleted_count=deleted)
        return deleted

    def sweep_stale_temps(self, *, now: float) -> None:
        """Unlinks crash-left bot temps older than the stale-temp window (best-effort).

        Gated on the bot's own temp-name shape (like the reaper's 32-hex guard), so a foreign
        `.tmp-*` parked in the shared serve dir is never reaped.
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

    def run_maintenance(self, *, now: float) -> tuple[int, int]:
        """One sweep for the cleanup loop: reap expired, enforce the cap, clear stale temps.

        Returns (deleted_count, freed_bytes). Age runs before size so the cap acts on what remains.
        """
        self.sweep_stale_temps(now=now)
        deleted = self.cleanup_expired(now=now)
        freed = self.enforce_cap(now=now)
        return deleted, freed


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
        concurrently. (b) The native list is clamped to `attachment_limit` first, the trailing
        overflow dropped, so a marginal combined overflow then sheds a low-priority trailing image
        rather than a prioritized voice/music clip. (c) The largest remaining are peeled to hosted
        URLs until the combined body clears `upload_limit - envelope_margin`. Input order is
        preserved in `native`, so a caller leading with voice/music keeps a trailing image as the drop.

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

        # (b) Discord's per-message attachment cap, applied BEFORE byte peeling: items past the cap
        # cannot ride the edit anyway, so dropping the trailing overflow first means a marginal
        # combined overflow sheds a low-priority trailing image instead of peeling the prioritized
        # voice/music clip (callers lead with those).
        if len(fitting) > attachment_limit:
            dropped.extend(fitting[attachment_limit:])
            fitting = fitting[:attachment_limit]

        # (c) Combined total: peel the largest remaining to a URL until the multipart body fits.
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
        return MediaPlan(native=fitting, hosted_urls=hosted_urls, dropped_items=dropped)


def build_media_delivery_planner() -> MediaDeliveryPlanner:
    """Builds the default MediaDeliveryPlanner wired to the env-configured external media host.

    The shared wiring used by every cog that delivers media (the streamer builds its own
    test-friendly disabled planner instead). `MediaHostingConfig` self-disables when the host is
    unconfigured, so this stays the byte-for-byte host-free path until hosting is set up.
    """
    return MediaDeliveryPlanner(media_hosting=MediaHostingService(config=MediaHostingConfig()))


__all__ = [
    "DEFAULT_NON_NITRO_UPLOAD_LIMIT",
    "DISCORD_ATTACHMENT_LIMIT",
    "MEDIA_ENVELOPE_MARGIN",
    "MediaDeliveryPlanner",
    "MediaHostingConfig",
    "MediaHostingService",
    "MediaItem",
    "MediaPlan",
    "build_media_delivery_planner",
    "upload_limit_for",
]
