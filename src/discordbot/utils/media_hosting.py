"""Hosts oversized media on an external static server and returns a public URL.

When generated or downloaded media exceeds Discord's per-message upload limit, the bot writes
it into a directory served by an external nginx static host (bind-mounted from the Docker host)
and posts the public URL instead of failing or silently dropping the file. The host serves a
strict extension allowlist, so this helper refuses any suffix the server would 404 on. Every
method is best-effort: it returns None (never raises) when hosting is disabled, unconfigured,
handed a non-allowlisted suffix, or fails to write, so callers degrade back to their previous
behavior with no extra handling.
"""

import shutil
from pathlib import Path
import secrets

import dotenv
import logfire
from pydantic import Field, BaseModel, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()

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


__all__ = ["MediaHostingConfig", "MediaHostingService"]
