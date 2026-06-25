"""Unit tests for the unified media-delivery module: the host writer and the planner."""

import os
import errno
from pathlib import Path

import pytest

from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaHostingConfig,
    MediaHostingService,
    MediaDeliveryPlanner,
)


def _service(
    *, serve_dir: Path, enabled: bool = True, base_url: str = "https://media.test"
) -> MediaHostingService:
    """Builds a host writer whose config points at a temp serve dir (via the env aliases)."""
    return MediaHostingService(
        config=MediaHostingConfig(
            MEDIA_HOSTING_ENABLED=enabled,
            MEDIA_HOSTING_BASE_URL=base_url,
            MEDIA_HOSTING_SERVE_DIR=str(serve_dir),
        )
    )


def _planner(
    *, serve_dir: Path, enabled: bool = True, base_url: str = "https://media.test"
) -> MediaDeliveryPlanner:
    """Builds a delivery planner over a host writer pointed at a temp serve dir."""
    return MediaDeliveryPlanner(
        media_hosting=_service(serve_dir=serve_dir, enabled=enabled, base_url=base_url)
    )


# --- host writer (publish_bytes / publish_path) ---------------------------------------------


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


def test_publish_path_falls_back_on_cross_device_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When os.rename raises EXDEV (serve dir is a bind-mount), shutil.move's copy fallback runs."""
    source_dir = tmp_path / "src"
    serve_dir = tmp_path / "serve"
    source_dir.mkdir()
    source = source_dir / "clip.mp4"
    source.write_bytes(b"movie")
    service = _service(serve_dir=serve_dir)

    def _exdev_rename(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "rename", _exdev_rename)

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


# --- MediaItem ------------------------------------------------------------------------------


def test_media_item_size_reads_bytes_and_path(tmp_path: Path) -> None:
    """Size is len for bytes and st_size for a path (the path is read, not the file)."""
    on_disk = tmp_path / "clip.mp4"
    on_disk.write_bytes(b"abc")
    assert MediaItem(source=b"abcd", filename="a.png").size == 4
    assert MediaItem(source=on_disk, filename="clip.mp4").size == 3


def test_media_item_to_file_carries_filename(tmp_path: Path) -> None:
    """to_file builds a fresh nextcord File from bytes or a path, keeping the filename."""
    on_disk = tmp_path / "clip.mp4"
    on_disk.write_bytes(b"abc")
    assert MediaItem(source=b"abcd", filename="a.png").to_file().filename == "a.png"
    assert MediaItem(source=on_disk, filename="clip.mp4").to_file().filename == "clip.mp4"


# --- planner --------------------------------------------------------------------------------


async def test_plan_single_item_fits_is_native(tmp_path: Path) -> None:
    """A lone item under the limit attaches natively; nothing hosted or dropped."""
    planner = _planner(serve_dir=tmp_path, enabled=False)
    plan = await planner.plan(
        items=[MediaItem(source=b"x" * 10, filename="a.png")], upload_limit=100
    )
    assert [item.filename for item in plan.native] == ["a.png"]
    assert plan.hosted_urls == []
    assert plan.dropped_items == []


async def test_plan_hosts_individually_oversize_bytes_item(tmp_path: Path) -> None:
    """A bytes item over the limit is hosted to a URL, none attached, none dropped."""
    planner = _planner(serve_dir=tmp_path)
    plan = await planner.plan(
        items=[MediaItem(source=b"y" * 200, filename="big.wav")], upload_limit=100
    )
    assert plan.native == []
    assert len(plan.hosted_urls) == 1
    assert plan.hosted_urls[0].endswith(".wav")
    assert plan.dropped_items == []


async def test_plan_hosts_oversize_path_item_by_move(tmp_path: Path) -> None:
    """A path item over the limit is moved into the serve dir and linked (source gone)."""
    serve_dir = tmp_path / "serve"
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"m" * 200)
    planner = _planner(serve_dir=serve_dir)

    plan = await planner.plan(
        items=[MediaItem(source=source, filename=source.name)], upload_limit=100
    )

    assert plan.native == []
    assert len(plan.hosted_urls) == 1
    assert plan.hosted_urls[0].endswith(".mp4")
    assert not source.exists()


async def test_plan_drops_oversize_when_hosting_disabled(tmp_path: Path) -> None:
    """With hosting off, an oversize item drops while the fitting one still attaches."""
    planner = _planner(serve_dir=tmp_path, enabled=False)
    items = [
        MediaItem(source=b"x" * 50, filename="small.png"),
        MediaItem(source=b"y" * 200, filename="big.wav"),
    ]

    plan = await planner.plan(items=items, upload_limit=100)

    assert [item.filename for item in plan.native] == ["small.png"]
    assert plan.hosted_urls == []
    assert [item.filename for item in plan.dropped_items] == ["big.wav"]


async def test_plan_peels_largest_on_combined_overflow(tmp_path: Path) -> None:
    """Items each fitting individually but summing past the limit peel the largest to a URL."""
    planner = _planner(serve_dir=tmp_path)
    # All three fit under the limit individually, but their sum + the 1 MiB envelope margin does
    # not; only the largest is peeled to a hosted URL, leaving the other two as native attachments.
    limit = 1024 * 1024 + 500
    items = [
        MediaItem(source=b"a" * 400, filename="reply.wav"),
        MediaItem(source=b"b" * 300, filename="music.mp3"),
        MediaItem(source=b"c" * 200, filename="generated.png"),
    ]

    plan = await planner.plan(
        items=items, upload_limit=limit, envelope_margin=MEDIA_ENVELOPE_MARGIN
    )

    assert {item.filename for item in plan.native} == {"music.mp3", "generated.png"}
    assert len(plan.hosted_urls) == 1
    assert plan.hosted_urls[0].endswith(".wav")
    assert plan.dropped_items == []


async def test_plan_drops_largest_on_combined_overflow_when_hosting_disabled(
    tmp_path: Path,
) -> None:
    """Host-off combined-overflow: the largest is peeled into dropped_items, the rest stay native."""
    planner = _planner(serve_dir=tmp_path, enabled=False)
    # Each fits individually, but sum + the 1 MiB margin overflows; with hosting off the largest
    # cannot be hosted, so it drops (the streamer's drop + ⚠️ path) while the rest stay native in order.
    limit = 1024 * 1024 + 500
    items = [
        MediaItem(source=b"a" * 400, filename="reply.wav"),
        MediaItem(source=b"b" * 300, filename="music.mp3"),
        MediaItem(source=b"c" * 200, filename="generated.png"),
    ]

    plan = await planner.plan(
        items=items, upload_limit=limit, envelope_margin=MEDIA_ENVELOPE_MARGIN
    )

    assert [item.filename for item in plan.native] == ["music.mp3", "generated.png"]
    assert plan.hosted_urls == []
    assert [item.filename for item in plan.dropped_items] == ["reply.wav"]


async def test_plan_clamps_to_attachment_limit(tmp_path: Path) -> None:
    """Eleven items all fitting (voice + music + 9 images) clamp to 10, dropping the trailing one."""
    planner = _planner(serve_dir=tmp_path, enabled=False)
    # Limit is well above the combined size + envelope margin, so nothing is hosted; only the
    # 10-attachment count cap applies, dropping the trailing item while native keeps input order.
    limit = 1024 * 1024 + 1000
    items = [MediaItem(source=b"x" * 10, filename=f"f{i}.png") for i in range(11)]

    plan = await planner.plan(
        items=items, upload_limit=limit, envelope_margin=MEDIA_ENVELOPE_MARGIN
    )

    assert [item.filename for item in plan.native] == [f"f{i}.png" for i in range(10)]
    assert plan.hosted_urls == []
    assert [item.filename for item in plan.dropped_items] == ["f10.png"]


async def test_plan_count_clamp_precedes_peel_so_marginal_overflow_keeps_voice(
    tmp_path: Path,
) -> None:
    """An 11th trailing image causing a marginal overflow is dropped first, sparing the voice clip.

    Eleven items each fit individually but their sum overflows; dropping the trailing 11th (count
    cap) brings the rest under the limit, so the prioritized (largest, leading) voice clip is never
    peeled. With hosting off, peeling-before-clamping would have dropped the voice clip instead.
    """
    planner = _planner(serve_dir=tmp_path, enabled=False)
    items = [
        MediaItem(source=b"v" * 200, filename="reply.wav"),  # largest + leads: must survive
        MediaItem(source=b"m" * 10, filename="music.mp3"),
        *(MediaItem(source=b"i" * 10, filename=f"generated_{i}.png") for i in range(1, 10)),
    ]
    # 290 (10 kept items) <= 295 < 300 (all 11): clamping the trailing image avoids any peel.
    plan = await planner.plan(items=items, upload_limit=295, envelope_margin=0)

    native_names = [item.filename for item in plan.native]
    assert "reply.wav" in native_names  # the voice clip survived (not peeled for size)
    assert len(plan.native) == 10
    assert plan.hosted_urls == []
    assert [item.filename for item in plan.dropped_items] == ["generated_9.png"]
