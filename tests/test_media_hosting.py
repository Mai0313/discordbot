"""Unit tests for the external media-hosting fallback helper."""

from pathlib import Path

from discordbot.utils.media_hosting import MediaHostingConfig, MediaHostingService


def _service(
    *, serve_dir: Path, enabled: bool = True, base_url: str = "https://media.test"
) -> MediaHostingService:
    """Builds a service whose config points at a temp serve dir (via the env aliases)."""
    return MediaHostingService(
        config=MediaHostingConfig(
            MEDIA_HOSTING_ENABLED=enabled,
            MEDIA_HOSTING_BASE_URL=base_url,
            MEDIA_HOSTING_SERVE_DIR=str(serve_dir),
        )
    )


def test_publish_bytes_writes_allowlisted_suffix(tmp_path: Path) -> None:
    """An allowlisted suffix is written under an unguessable name and a public URL returned."""
    service = _service(serve_dir=tmp_path)

    url = service.publish_bytes(data=b"fake-wav", suffix=".wav")

    assert url is not None
    name = url.removeprefix("https://media.test/")
    assert name.endswith(".wav")
    assert name != ".wav"  # a token stem precedes the suffix
    written = tmp_path / name
    assert written.read_bytes() == b"fake-wav"


def test_publish_bytes_rejects_non_allowlisted_suffix(tmp_path: Path) -> None:
    """A suffix the host would 404 (e.g. .aiff from the music renderer) is refused, nothing written."""
    service = _service(serve_dir=tmp_path)

    url = service.publish_bytes(data=b"x", suffix=".aiff")

    assert url is None
    assert list(tmp_path.iterdir()) == []


def test_publish_bytes_normalizes_uppercase_suffix(tmp_path: Path) -> None:
    """An uppercase suffix is lowercased to its allowlisted form."""
    service = _service(serve_dir=tmp_path)

    url = service.publish_bytes(data=b"x", suffix=".JPG")

    assert url is not None
    assert url.endswith(".jpg")


def test_publish_path_moves_file_across_dirs(tmp_path: Path) -> None:
    """publish_path moves the source into the serve dir (source gone, dest present)."""
    source_dir = tmp_path / "src"
    serve_dir = tmp_path / "serve"
    source_dir.mkdir()
    source = source_dir / "clip.mp4"
    source.write_bytes(b"movie")
    service = _service(serve_dir=serve_dir)

    url = service.publish_path(file_path=source)

    assert url is not None
    assert not source.exists()
    name = url.removeprefix("https://media.test/")
    assert (serve_dir / name).read_bytes() == b"movie"


def test_publish_path_rejects_non_allowlisted_and_keeps_file(tmp_path: Path) -> None:
    """A non-allowlisted file is not moved; it stays in place for the caller's own cleanup."""
    serve_dir = tmp_path / "serve"
    source = tmp_path / "archive.zip"
    source.write_bytes(b"data")
    service = _service(serve_dir=serve_dir)

    url = service.publish_path(file_path=source)

    assert url is None
    assert source.exists()


def test_disabled_returns_none(tmp_path: Path) -> None:
    """An explicit kill-switch off disables the fallback even when fully configured."""
    service = _service(serve_dir=tmp_path, enabled=False)
    assert service.publish_bytes(data=b"x", suffix=".png") is None


def test_empty_base_url_returns_none(tmp_path: Path) -> None:
    """An empty base URL leaves the fallback inert (keeps tests / unconfigured deploys green)."""
    service = _service(serve_dir=tmp_path, base_url="")
    assert service.publish_bytes(data=b"x", suffix=".png") is None


def test_empty_serve_dir_returns_none() -> None:
    """An empty serve dir leaves the fallback inert."""
    service = MediaHostingService(
        config=MediaHostingConfig(
            MEDIA_HOSTING_ENABLED=True,
            MEDIA_HOSTING_BASE_URL="https://media.test",
            MEDIA_HOSTING_SERVE_DIR="",
        )
    )
    assert service.publish_bytes(data=b"x", suffix=".png") is None


def test_write_failure_returns_none_without_raising(tmp_path: Path) -> None:
    """A serve dir that cannot be created (it is a regular file) degrades to None, never raises."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    service = _service(serve_dir=blocker)

    assert service.publish_bytes(data=b"x", suffix=".png") is None


def test_empty_config_is_unavailable() -> None:
    """Empty base_url / serve_dir make the service unavailable (the test-green guard)."""
    config = MediaHostingConfig(MEDIA_HOSTING_BASE_URL="", MEDIA_HOSTING_SERVE_DIR="")
    assert config.available is False
