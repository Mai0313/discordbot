"""Pins the unified media-delivery decision: what attaches, what becomes a URL, and what is lost.

`utils/media_delivery.py` is the one place five media-delivering cogs share the
attach-vs-host-vs-drop decision, so a regression here is never local to a single feature. Three
properties carry the weight, and none of them is visible in a diff.

The reaper guard is the first. The serve dir is a shared bind mount that can hold files this bot
never wrote (an nginx `access.log`, a human-named clip, a foreign temp), so the cleanup tests
assert deletion is confined to the exact name shape the writer produces — 32 hex characters plus
an allowlisted suffix, regular files only, non-recursive — and that a hex-named symlink is
skipped rather than followed onto its target. Widening that match deletes someone else's data.

The host writer's degradation is the second. A kill-switch off, an empty base URL, an empty or
absent or not-a-directory serve dir, a suffix the static host would 404, and a failed `os.replace`
each have to answer None without writing anything, without creating the serve dir, and without
raising into a reply pipeline, because `MEDIA_HOSTING_ENABLED=false` is required to be
byte-for-byte the old host-free path at every call site. Content addressing sits beside it:
identical bytes dedup to one file and one URL while refreshing its mtime (which is what keeps a
re-hosted clip alive under both caps), a path source is consumed on a fresh host but left for the
caller on a dedup hit, and the hash is streamed rather than read whole, since that branch exists
for multi-GB downloads.

The planner's ordering is the third. Items are clamped to Discord's 10-attachment cap BEFORE the
largest are peeled to fit the multipart body, so a marginal combined overflow sheds a trailing
generated image instead of the voice clip the caller led with; the reverse order looks just as
reasonable and quietly costs the user the artifact they asked for.

Everything drives a real `MediaHostingService` over a `tmp_path` serve dir built through
`make_media_hosting_config`, so no test can reach a deployment's live serve dir, and the grace,
retention and stale-temp windows are exercised by backdating mtimes rather than by waiting.
"""

import os
import re
import time
from pathlib import Path

import pytest

from discordbot.utils.media_delivery import (
    _TEMP_PREFIX,
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaHostingService,
    MediaDeliveryPlanner,
)

from tests.helpers.casting import make_media_hosting_config


def _service(
    *,
    serve_dir: Path,
    enabled: bool = True,
    base_url: str = "https://media.test",
    max_bytes: int = 8 * 1024**3,
    retention_hours: float = 168.0,
) -> MediaHostingService:
    """Builds a host writer whose config points at a temp serve dir (via the env aliases).

    Returns:
        A service over `serve_dir` with no `.env` or process environment mixed in, so the
        surrounding deployment's `MEDIA_HOSTING_*` cannot reach it.
    """
    return MediaHostingService(
        config=make_media_hosting_config(
            enabled=enabled,
            base_url=base_url,
            serve_dir=str(serve_dir),
            max_bytes=max_bytes,
            retention_hours=retention_hours,
        )
    )


def _hosted_files(serve_dir: Path) -> list[str]:
    """The serve dir's entry names, minus anything called `.tmp-*`.

    Returns:
        Every remaining entry name, in `iterdir` order. The service's own in-flight temps are
        prefixed `_TEMP_PREFIX`, not a bare `.tmp-`, so a leftover one still shows up here and
        an `== []` assertion below does pin that a publish left no scratch file behind.
    """
    return [p.name for p in serve_dir.iterdir() if not p.name.startswith(".tmp-")]


def _host(service: MediaHostingService, *, data: bytes, suffix: str = ".png") -> str:
    """Hosts bytes, failing the calling test if the publish did not produce a URL.

    Returns:
        The published filename, with the test base URL stripped off, so it can be joined onto
        the serve dir for an mtime or content assertion.
    """
    url = service.publish_bytes(data=data, suffix=suffix)
    assert url is not None
    return url.removeprefix("https://media.test/")


def _age(path: Path, *, seconds: float) -> None:
    """Backdates a file's mtime so a window can be crossed without the test waiting for it."""
    when = time.time() - seconds
    os.utime(path, (when, when))


def _planner(
    *, serve_dir: Path, enabled: bool = True, base_url: str = "https://media.test"
) -> MediaDeliveryPlanner:
    """Builds a delivery planner over a host writer pointed at a temp serve dir.

    Returns:
        A planner that really hosts into `serve_dir`, or, with `enabled` false, one whose
        hosting is inert so every oversize item lands in `dropped_items` instead.
    """
    return MediaDeliveryPlanner(
        media_hosting=_service(serve_dir=serve_dir, enabled=enabled, base_url=base_url)
    )


# --- host writer (publish_bytes / publish_path) ---------------------------------------------


def test_publish_bytes_writes_content_addressed_name(tmp_path: Path) -> None:
    """Bytes are published under a 32-hex content-addressed name carrying the asked-for suffix."""
    service = _service(serve_dir=tmp_path)

    url = service.publish_bytes(data=b"fake-wav", suffix=".wav")

    assert url is not None
    name = url.removeprefix("https://media.test/")
    assert re.fullmatch(r"[0-9a-f]{32}\.wav", name)
    assert (tmp_path / name).read_bytes() == b"fake-wav"
    assert not any(p.name.startswith(".tmp-") for p in tmp_path.iterdir())


def test_publish_bytes_dedups_identical_content(tmp_path: Path) -> None:
    """Hosting identical bytes twice yields one file and the same URL, refreshing the mtime."""
    service = _service(serve_dir=tmp_path)

    url1 = _host(service, data=b"A" * 64)
    _age(tmp_path / url1, seconds=100)  # so the refresh is observable against a coarse mtime
    old_mtime = (tmp_path / url1).stat().st_mtime
    url2 = service.publish_bytes(data=b"A" * 64, suffix=".png")

    assert url2 == f"https://media.test/{url1}"
    assert _hosted_files(tmp_path) == [url1]
    assert (tmp_path / url1).stat().st_mtime > old_mtime  # the refresh is what defers both caps


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
    """A suffix the host would 404 (`.aiff`, which the music renderer emits) writes nothing."""
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
    """A failed atomic replace leaves the serve dir empty: no content name, no scratch file."""
    service = _service(serve_dir=tmp_path)

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", _boom)

    assert service.publish_bytes(data=b"A" * 10, suffix=".png") is None
    # A surviving final name would be a poison cache entry dedup then serves forever.
    assert _hosted_files(tmp_path) == []


def test_publish_path_hosts_and_consumes_source(tmp_path: Path) -> None:
    """A fresh host copies the on-disk source under a content name and consumes the original."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()  # the serve dir is a pre-existing host mount; the bot never creates it
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"movie")
    service = _service(serve_dir=serve_dir)

    url = service.publish_path(file_path=source)

    assert url is not None
    assert not source.exists()
    name = url.removeprefix("https://media.test/")
    assert re.fullmatch(r"[0-9a-f]{32}\.mp4", name)
    assert (serve_dir / name).read_bytes() == b"movie"


def test_publish_path_dedup_hit_leaves_source(tmp_path: Path) -> None:
    """A dedup hit returns the existing URL and leaves the source for the caller to clean up."""
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
    """Hashing an on-disk source streams it, so a multi-GB clip is never loaded whole."""
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
    """A file the host would 404 is refused and left in place for the caller's own cleanup."""
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
    """An empty base URL leaves the fallback inert, so an unconfigured deployment hosts nothing."""
    service = _service(serve_dir=tmp_path, base_url="")
    assert service.publish_bytes(data=b"x", suffix=".png") is None


def test_empty_serve_dir_returns_none() -> None:
    """An empty serve dir leaves the fallback inert."""
    service = MediaHostingService(
        config=make_media_hosting_config(enabled=True, base_url="https://media.test", serve_dir="")
    )
    assert service.publish_bytes(data=b"x", suffix=".png") is None


def test_serve_dir_that_is_a_regular_file_returns_none(tmp_path: Path) -> None:
    """A serve dir that is a regular file degrades to None rather than raising at the caller."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    service = _service(serve_dir=blocker)

    assert service.publish_bytes(data=b"x", suffix=".png") is None


def test_missing_serve_dir_falls_back_without_creating_it(tmp_path: Path) -> None:
    """A configured-but-absent serve dir falls back to None and is never created by the bot."""
    serve_dir = tmp_path / "not_mounted"
    service = _service(serve_dir=serve_dir)

    assert service.publish_bytes(data=b"x", suffix=".png") is None
    # A container-local dir nginx cannot see would only 404, so creating it is worse than a drop.
    assert not serve_dir.exists()


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
    assert n1 not in remaining
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
    """A file past the retention window is reaped by the age sweep while a recent one stays."""
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
    """The reaper deletes only the bot's own 32-hex names, never a foreign file in the dir."""
    service = _service(serve_dir=tmp_path, max_bytes=1, retention_hours=0.0001)
    (tmp_path / "access.log").write_text("log")  # foreign, allowlisted suffix
    (tmp_path / "report.json").write_text("{}")  # foreign, allowlisted suffix
    (tmp_path / "movie.mp4").write_bytes(b"film")  # foreign, human stem
    (tmp_path / ("0" * 32 + ".zip")).write_bytes(b"zip")  # 32-hex stem but NON-allowlisted ext
    (tmp_path / "subdir").mkdir()
    bot_file = _host(service, data=b"Z" * 99)  # the positive control: this one must be reaped
    for entry in tmp_path.iterdir():
        if entry.is_file():
            _age(entry, seconds=99999)

    service.run_maintenance(now=time.time())

    survivors = {p.name for p in tmp_path.iterdir()}
    assert {"access.log", "report.json", "movie.mp4", "0" * 32 + ".zip", "subdir"} <= survivors
    assert bot_file not in survivors


def test_cleanup_skips_symlinks(tmp_path: Path) -> None:
    """A hex-named symlink is skipped by the reaper, so the file it points at survives."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    foreign = tmp_path / "foreign.mp4"
    foreign.write_bytes(b"x" * 999)
    link = serve_dir / ("0" * 32 + ".mp4")
    link.symlink_to(foreign)
    service = _service(serve_dir=serve_dir, max_bytes=1, retention_hours=0.0001)

    service.run_maintenance(now=time.time())

    # The name matches the reaper's shape exactly; only the regular-file check saves the target.
    assert foreign.exists()


def test_enforce_cap_disabled_when_max_bytes_zero(tmp_path: Path) -> None:
    """A non-positive max_bytes disables size eviction even when the sweep is called directly."""
    service = _service(serve_dir=tmp_path, max_bytes=0, retention_hours=0)
    name = _host(service, data=b"A" * 999)
    _age(tmp_path / name, seconds=99999)

    assert service.enforce_cap(now=time.time()) == 0
    assert name in _hosted_files(tmp_path)


def test_cleanup_expired_disabled_when_retention_zero(tmp_path: Path) -> None:
    """A non-positive retention_hours disables age reaping even on a direct sweep call."""
    service = _service(serve_dir=tmp_path, max_bytes=0, retention_hours=0)
    name = _host(service, data=b"A" * 10)
    _age(tmp_path / name, seconds=99999)

    assert service.cleanup_expired(now=time.time()) == 0
    assert name in _hosted_files(tmp_path)


def test_sweep_removes_stale_bot_temps_only(tmp_path: Path) -> None:
    """A crash-left bot temp is reaped while a fresh bot temp and a foreign temp both survive."""
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

    assert not stale_bot.exists()
    assert fresh_bot.exists()  # still plausibly an in-flight write
    assert foreign.exists()  # the shape gate, not the age, is what spares it


def test_cleanup_no_op_on_missing_serve_dir(tmp_path: Path) -> None:
    """All cleanup methods no-op (and never create the dir) when the serve dir is absent."""
    serve_dir = tmp_path / "absent"
    service = _service(serve_dir=serve_dir)

    assert service.enforce_cap(now=time.time()) == 0
    assert service.cleanup_expired(now=time.time()) == 0
    service.sweep_stale_temps(now=time.time())
    assert not serve_dir.exists()


def test_empty_config_is_unavailable() -> None:
    """An empty base_url and serve_dir leave `available` false however the kill-switch is set."""
    config = make_media_hosting_config(enabled=True, base_url="", serve_dir="")
    assert config.available is False


# --- MediaItem ------------------------------------------------------------------------------


def test_media_item_size_reads_bytes_and_path(tmp_path: Path) -> None:
    """Size is len() for in-memory bytes and st_size for a path, which is stat'd, not read."""
    on_disk = tmp_path / "clip.mp4"
    on_disk.write_bytes(b"abc")
    assert MediaItem(source=b"abcd", filename="a.png").size == 4
    assert MediaItem(source=on_disk, filename="clip.mp4").size == 3


def test_media_item_to_file_carries_filename(tmp_path: Path) -> None:
    """A nextcord File built from either source shape keeps the item's attachment filename."""
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
    """A bytes item over the upload limit becomes a hosted URL instead of an attachment."""
    planner = _planner(serve_dir=tmp_path)
    plan = await planner.plan(
        items=[MediaItem(source=b"y" * 200, filename="big.wav")], upload_limit=100
    )
    assert plan.native == []
    assert len(plan.hosted_urls) == 1
    assert plan.hosted_urls[0].endswith(".wav")
    assert plan.dropped_items == []


async def test_plan_hosts_oversize_path_item_by_move(tmp_path: Path) -> None:
    """A path item over the limit is moved into the serve dir and linked, source consumed."""
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
    # The margin is what overflows here: the three payloads are tiny, so a planner that measured
    # only the file bytes would attach all three and let Discord 400 the multipart body.
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
    """On a combined overflow with hosting off, the peeled item drops instead of becoming a URL."""
    planner = _planner(serve_dir=tmp_path, enabled=False)
    # Dropping is the streamer's own pre-hosting behavior, which hosting-off owes byte for byte.
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
    """Eleven items that all fit clamp to Discord's ten, dropping the trailing one."""
    planner = _planner(serve_dir=tmp_path, enabled=False)
    # Well above the combined size + margin, so the count cap is the only thing acting here.
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
    """A marginal overflow sheds the trailing 11th image rather than peeling the voice clip."""
    planner = _planner(serve_dir=tmp_path, enabled=False)
    items = [
        # The voice clip is both the largest and the caller's lead, so peeling before clamping
        # would pick exactly it, and with hosting off that means the user loses it outright.
        MediaItem(source=b"v" * 200, filename="reply.wav"),
        MediaItem(source=b"m" * 10, filename="music.mp3"),
        *(MediaItem(source=b"i" * 10, filename=f"generated_{i}.png") for i in range(1, 10)),
    ]
    # 290 (10 kept items) <= 295 < 300 (all 11): clamping the trailing image avoids any peel.
    plan = await planner.plan(items=items, upload_limit=295, envelope_margin=0)

    native_names = [item.filename for item in plan.native]
    assert "reply.wav" in native_names
    assert len(plan.native) == 10
    assert plan.hosted_urls == []
    assert [item.filename for item in plan.dropped_items] == ["generated_9.png"]
