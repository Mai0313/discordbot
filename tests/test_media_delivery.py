"""Unit tests for the unified media-delivery module: the host writer and the planner."""

import os
import re
import time
from pathlib import Path

import pytest

from discordbot.utils.media_delivery import (
    _TEMP_PREFIX,
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaHostingConfig,
    MediaHostingService,
    MediaDeliveryPlanner,
)


def _service(
    *,
    serve_dir: Path,
    enabled: bool = True,
    base_url: str = "https://media.test",
    max_bytes: int = 8 * 1024**3,
    retention_hours: float = 168.0,
) -> MediaHostingService:
    """Builds a host writer whose config points at a temp serve dir (via the env aliases)."""
    return MediaHostingService(
        config=MediaHostingConfig(
            MEDIA_HOSTING_ENABLED=enabled,
            MEDIA_HOSTING_BASE_URL=base_url,
            MEDIA_HOSTING_SERVE_DIR=str(serve_dir),
            MEDIA_HOSTING_MAX_BYTES=max_bytes,
            MEDIA_HOSTING_RETENTION_HOURS=retention_hours,
        )
    )


def _hosted_files(serve_dir: Path) -> list[str]:
    """The final hosted filenames in a serve dir (excluding in-flight `.tmp-*` temps)."""
    return [p.name for p in serve_dir.iterdir() if not p.name.startswith(".tmp-")]


def _host(service: MediaHostingService, *, data: bytes, suffix: str = ".png") -> str:
    """Hosts bytes and returns the resulting filename (asserts the publish succeeded)."""
    url = service.publish_bytes(data=data, suffix=suffix)
    assert url is not None
    return url.removeprefix("https://media.test/")


def _age(path: Path, *, seconds: float) -> None:
    """Backdates a file's mtime by `seconds` (so it is past the eviction grace / age cutoff)."""
    when = time.time() - seconds
    os.utime(path, (when, when))


def _planner(
    *, serve_dir: Path, enabled: bool = True, base_url: str = "https://media.test"
) -> MediaDeliveryPlanner:
    """Builds a delivery planner over a host writer pointed at a temp serve dir."""
    return MediaDeliveryPlanner(
        media_hosting=_service(serve_dir=serve_dir, enabled=enabled, base_url=base_url)
    )


# --- host writer (publish_bytes / publish_path) ---------------------------------------------


def test_publish_bytes_writes_content_addressed_name(tmp_path: Path) -> None:
    """Bytes are written under a 32-hex content-addressed name; the temp is os.replace'd away."""
    service = _service(serve_dir=tmp_path)

    url = service.publish_bytes(data=b"fake-wav", suffix=".wav")

    assert url is not None
    name = url.removeprefix("https://media.test/")
    assert re.fullmatch(r"[0-9a-f]{32}\.wav", name)  # content hash + allowlisted suffix
    assert (tmp_path / name).read_bytes() == b"fake-wav"
    assert not any(p.name.startswith(".tmp-") for p in tmp_path.iterdir())  # no leftover temp


def test_publish_bytes_dedups_identical_content(tmp_path: Path) -> None:
    """Hosting identical bytes twice yields one file and the same URL, refreshing the mtime."""
    service = _service(serve_dir=tmp_path)

    url1 = _host(service, data=b"A" * 64)
    _age(tmp_path / url1, seconds=100)  # age the file so the refresh is observable
    old_mtime = (tmp_path / url1).stat().st_mtime
    url2 = service.publish_bytes(data=b"A" * 64, suffix=".png")

    assert url2 == f"https://media.test/{url1}"
    assert _hosted_files(tmp_path) == [url1]  # exactly one copy
    assert (tmp_path / url1).stat().st_mtime > old_mtime  # the re-host refreshed it (LRU/age)


def test_publish_bytes_different_content_two_files(tmp_path: Path) -> None:
    """Different bytes hash to different names: two files, two URLs."""
    service = _service(serve_dir=tmp_path)

    name_a = _host(service, data=b"A" * 10)
    name_b = _host(service, data=b"B" * 10)

    assert name_a != name_b
    assert len(_hosted_files(tmp_path)) == 2


def test_publish_bytes_same_content_different_suffix_two_files(tmp_path: Path) -> None:
    """The same bytes under different suffixes stay distinct (the suffix rides the name)."""
    service = _service(serve_dir=tmp_path)

    name_png = _host(service, data=b"A" * 10, suffix=".png")
    name_jpg = _host(service, data=b"A" * 10, suffix=".jpg")

    assert name_png != name_jpg
    assert len(_hosted_files(tmp_path)) == 2


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


def test_publish_bytes_failure_leaves_no_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the atomic os.replace fails, no content-named file ever appears (and the temp is cleaned)."""
    service = _service(serve_dir=tmp_path)

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", _boom)

    assert service.publish_bytes(data=b"A" * 10, suffix=".png") is None
    assert _hosted_files(tmp_path) == []  # only a (cleaned) temp ever existed, never a final name


def test_publish_path_hosts_and_consumes_source(tmp_path: Path) -> None:
    """publish_path hosts the source under a content name and unlinks the source on a miss."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()  # the serve dir is a pre-existing host mount; the bot never creates it
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"movie")
    service = _service(serve_dir=serve_dir)

    url = service.publish_path(file_path=source)

    assert url is not None
    assert not source.exists()  # consumed on a fresh host
    name = url.removeprefix("https://media.test/")
    assert re.fullmatch(r"[0-9a-f]{32}\.mp4", name)
    assert (serve_dir / name).read_bytes() == b"movie"


def test_publish_path_dedup_hit_leaves_source(tmp_path: Path) -> None:
    """On a dedup hit publish_path returns the URL but leaves the source for the caller to clean."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    service = _service(serve_dir=serve_dir)
    first = tmp_path / "a.mp4"
    first.write_bytes(b"movie")
    url1 = service.publish_path(file_path=first)  # miss -> hosted, source consumed
    second = tmp_path / "b.mp4"
    second.write_bytes(b"movie")  # byte-identical

    url2 = service.publish_path(file_path=second)  # hit

    assert url2 == url1
    assert second.exists()  # the source is LEFT for the caller's own cleanup
    assert len(_hosted_files(serve_dir)) == 1


def test_publish_path_streams_without_reading_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """publish_path hashes by streaming; it never read_bytes() the (possibly multi-GB) source."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x" * 4096)
    service = _service(serve_dir=serve_dir)

    def _no_read_bytes(self: Path) -> bytes:
        raise AssertionError("publish_path must stream the hash, not read the whole file")

    monkeypatch.setattr(Path, "read_bytes", _no_read_bytes)

    assert service.publish_path(file_path=source) is not None


def test_publish_path_rejects_non_allowlisted_and_keeps_file(tmp_path: Path) -> None:
    """A non-allowlisted file is not hosted; it stays in place for the caller's own cleanup."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
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


def test_serve_dir_that_is_a_regular_file_returns_none(tmp_path: Path) -> None:
    """A serve dir that is a regular file (not a directory) degrades to None, never raises."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    service = _service(serve_dir=blocker)

    assert service.publish_bytes(data=b"x", suffix=".png") is None


def test_missing_serve_dir_falls_back_without_creating_it(tmp_path: Path) -> None:
    """A configured-but-absent serve dir falls back to None and is never created by the bot."""
    serve_dir = tmp_path / "not_mounted"
    service = _service(serve_dir=serve_dir)

    assert service.publish_bytes(data=b"x", suffix=".png") is None
    assert not serve_dir.exists()  # the bot must not create the (unmounted) serve dir


# --- cleanup: size cap, age cap, reaper guard -----------------------------------------------


def test_size_cap_evicts_oldest_keeps_recent(tmp_path: Path) -> None:
    """Past the size cap, the oldest hosted files are evicted (eagerly, at publish time)."""
    service = _service(serve_dir=tmp_path, max_bytes=120, retention_hours=0)
    n1 = _host(service, data=b"A" * 50)
    _age(tmp_path / n1, seconds=1000)  # past the grace window
    n2 = _host(service, data=b"B" * 50)
    _age(tmp_path / n2, seconds=500)
    n3 = _host(service, data=b"C" * 50)  # fresh; total 150 > 120 -> evict the oldest aged file

    remaining = _hosted_files(tmp_path)
    assert n1 not in remaining  # oldest evicted
    assert n2 in remaining
    assert n3 in remaining
    assert sum((tmp_path / f).stat().st_size for f in remaining) <= 120


def test_size_cap_protects_files_within_grace(tmp_path: Path) -> None:
    """A just-hosted file (and every concurrent publish) is grace-protected from eviction."""
    service = _service(serve_dir=tmp_path, max_bytes=80, retention_hours=0)
    n1 = _host(service, data=b"A" * 50)
    n2 = _host(service, data=b"B" * 50)  # total 100 > 80, but both within grace -> nothing evicted

    assert set(_hosted_files(tmp_path)) == {n1, n2}  # disk sits temporarily over cap


def test_size_cap_keeps_single_file_larger_than_cap(tmp_path: Path) -> None:
    """A delivered file alone exceeding the cap is kept; the loop terminates without thrashing."""
    service = _service(serve_dir=tmp_path, max_bytes=30, retention_hours=0)
    n1 = _host(service, data=b"A" * 20)
    _age(tmp_path / n1, seconds=1000)
    n2 = _host(service, data=b"B" * 100)  # alone over cap; total 120 -> evict n1, then stop

    remaining = _hosted_files(tmp_path)
    assert n2 in remaining  # the delivered file survives
    assert n1 not in remaining  # the only evictable candidate was reaped


def test_age_cap_reaps_old_keeps_recent(tmp_path: Path) -> None:
    """cleanup_expired deletes files older than retention_hours and keeps recent ones."""
    service = _service(serve_dir=tmp_path, max_bytes=0, retention_hours=1)
    old = _host(service, data=b"A" * 10)
    _age(tmp_path / old, seconds=7200)  # 2h, past the 1h retention
    recent = _host(service, data=b"B" * 10)

    deleted = service.cleanup_expired(now=time.time())

    assert deleted == 1
    remaining = _hosted_files(tmp_path)
    assert old not in remaining
    assert recent in remaining


def test_age_cap_keeps_file_at_exact_cutoff(tmp_path: Path) -> None:
    """A file whose mtime equals the cutoff is kept; only strictly-older files are reaped."""
    service = _service(serve_dir=tmp_path, max_bytes=0, retention_hours=1)
    name = _host(service, data=b"A" * 10)
    now = 1_000_000.0
    os.utime(tmp_path / name, (now - 3600.0, now - 3600.0))  # mtime == now - retention

    assert service.cleanup_expired(now=now) == 0
    assert _hosted_files(tmp_path) == [name]


def test_cleanup_never_touches_foreign_files(tmp_path: Path) -> None:
    """The reaper only ever deletes the bot's own 32-hex files, never a foreign file in the dir."""
    service = _service(serve_dir=tmp_path, max_bytes=1, retention_hours=0.0001)
    (tmp_path / "access.log").write_text("log")  # foreign, allowlisted suffix
    (tmp_path / "report.json").write_text("{}")  # foreign, allowlisted suffix
    (tmp_path / "movie.mp4").write_bytes(b"film")  # foreign, human stem
    (tmp_path / ("0" * 32 + ".zip")).write_bytes(b"zip")  # 32-hex stem but NON-allowlisted ext
    (tmp_path / "subdir").mkdir()
    bot_file = _host(service, data=b"Z" * 99)  # a real bot file that should be reaped
    for entry in tmp_path.iterdir():
        if entry.is_file():
            _age(entry, seconds=99999)

    service.run_maintenance(now=time.time())

    survivors = {p.name for p in tmp_path.iterdir()}
    assert {"access.log", "report.json", "movie.mp4", "0" * 32 + ".zip", "subdir"} <= survivors
    assert bot_file not in survivors  # only the bot's own file was reaped


def test_cleanup_skips_symlinks(tmp_path: Path) -> None:
    """A hex-named symlink pointing at a foreign file is skipped, so the target is never deleted."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    foreign = tmp_path / "foreign.mp4"
    foreign.write_bytes(b"x" * 999)
    link = serve_dir / ("0" * 32 + ".mp4")
    link.symlink_to(foreign)
    service = _service(serve_dir=serve_dir, max_bytes=1, retention_hours=0.0001)

    service.run_maintenance(now=time.time())

    assert (
        foreign.exists()
    )  # a symlink is not a regular file, so it (and its target) are untouched


def test_enforce_cap_disabled_when_max_bytes_zero(tmp_path: Path) -> None:
    """max_bytes <= 0 disables size eviction independently of the cleanup gate."""
    service = _service(serve_dir=tmp_path, max_bytes=0, retention_hours=0)
    name = _host(service, data=b"A" * 999)
    _age(tmp_path / name, seconds=99999)

    assert service.enforce_cap(now=time.time()) == 0
    assert name in _hosted_files(tmp_path)


def test_cleanup_expired_disabled_when_retention_zero(tmp_path: Path) -> None:
    """retention_hours <= 0 disables age reaping independently of the cleanup gate."""
    service = _service(serve_dir=tmp_path, max_bytes=0, retention_hours=0)
    name = _host(service, data=b"A" * 10)
    _age(tmp_path / name, seconds=99999)

    assert service.cleanup_expired(now=time.time()) == 0
    assert name in _hosted_files(tmp_path)


def test_sweep_removes_stale_bot_temps_only(tmp_path: Path) -> None:
    """A crash-left bot temp past the window is reaped; a fresh one and any FOREIGN temp are kept."""
    service = _service(serve_dir=tmp_path)
    stale_bot = tmp_path / f"{_TEMP_PREFIX}staletoken"
    stale_bot.write_bytes(b"partial")
    _age(stale_bot, seconds=9999)
    fresh_bot = tmp_path / f"{_TEMP_PREFIX}freshtoken"
    fresh_bot.write_bytes(b"partial")
    foreign = tmp_path / ".tmp-someoneelse"  # a foreign temp parked in the shared dir
    foreign.write_bytes(b"theirs")
    _age(foreign, seconds=9999)

    service.sweep_stale_temps(now=time.time())

    assert not stale_bot.exists()  # the bot's own stale temp is reaped
    assert fresh_bot.exists()  # a recent (in-flight) bot temp is kept
    assert foreign.exists()  # a foreign .tmp-* is never reaped


def test_cleanup_no_op_on_missing_serve_dir(tmp_path: Path) -> None:
    """All cleanup methods no-op (and never create the dir) when the serve dir is absent."""
    serve_dir = tmp_path / "absent"
    service = _service(serve_dir=serve_dir)

    assert service.enforce_cap(now=time.time()) == 0
    assert service.cleanup_expired(now=time.time()) == 0
    service.sweep_stale_temps(now=time.time())
    assert not serve_dir.exists()


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
    serve_dir.mkdir()  # the serve dir is a pre-existing host mount; the bot never creates it
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
