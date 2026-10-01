"""Shared pytest fixtures.

Each `*_isolated_db` fixture points the owning module's module-level engine at a fresh
`tmp_path` SQLite file for one test, all through `_isolate_engine`; the economy and
games-history ones are autouse, the others are requested by the tests that need them.
`memory_isolated_dir` covers more than a directory: the store dir, the `memory_job` engine,
the process-local caches, counters and task registries the store and pipeline hold, and the
git committer. The autouse fixtures are the other half of that isolation, keeping a real
deployment's `.env` and `data/` out of every test whether or not it asked for them.
"""

import os
from pathlib import Path
from itertools import count

import pytest
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine


def _isolate_engine(*, monkeypatch: pytest.MonkeyPatch, target: str, db_path: Path) -> None:
    """Points one module's `_engine` at a throwaway SQLite file for the test.

    NullPool closes each connection on return, so there is no pool to dispose and every fixture
    built on this stays sync. The schema bootstraps on the module's first session, because
    readiness is keyed on the engine and this one is new.
    """
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}", poolclass=NullPool)
    monkeypatch.setattr(target, engine)


@pytest.fixture(autouse=True)
def economy_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points the economy ledger at a throwaway `economy.db`.

    Autouse because the module engine is the deployed ledger, and a test that forgets the
    swap moves real balances rather than leaving a stray row: with no transaction table
    behind `total_earned - total_spent == balance`, such a write cannot be reconstructed.
    Patching every ledger function a command reaches is no substitute, since nothing checks
    that a test patched them all. The leaderboard caches are keyed on the query alone, so each
    test starts them empty too.
    """
    _isolate_engine(
        monkeypatch=monkeypatch,
        target="discordbot.services.economy.database._engine",
        db_path=tmp_path / "economy.db",
    )
    monkeypatch.setattr("discordbot.services.economy.database._top_n_cache", {})
    monkeypatch.setattr("discordbot.services.economy.database._top_losers_cache", {})


@pytest.fixture
def research_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-test SQLite file for the research table (reply.db)."""
    _isolate_engine(
        monkeypatch=monkeypatch,
        target="discordbot.cogs.research.database._engine",
        db_path=tmp_path / "reply.db",
    )


@pytest.fixture
def ask_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-test SQLite file for the `/ask` conversation table (reply.db)."""
    _isolate_engine(
        monkeypatch=monkeypatch,
        target="discordbot.cogs.gen_reply.ask_store._engine",
        db_path=tmp_path / "reply.db",
    )


@pytest.fixture
def messages_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-test SQLite file for the message log (messages.db)."""
    _isolate_engine(
        monkeypatch=monkeypatch,
        target="discordbot.cogs.log_msg.cog._engine",
        db_path=tmp_path / "messages.db",
    )


@pytest.fixture
def memory_isolated_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test memory dir + isolated memory_job DB with reset process-local state."""
    memories_dir = tmp_path / "memories"
    monkeypatch.setattr("discordbot.services.memory.store._MEMORY_DIR", memories_dir)
    monkeypatch.setattr("discordbot.services.memory.store._cleared_at", {})
    # The render cache is keyed on a per-scope write counter, and both live for the
    # process; without the reset a scope id reused across tests would serve the previous
    # test's document from a tmp_path that no longer exists.
    monkeypatch.setattr("discordbot.services.memory.store._write_generation", {})
    monkeypatch.setattr("discordbot.services.memory.store._render_cache", {})
    # No test may ever run git against the real store, so the committer stays off and
    # its queue stays unbound; `memory_git.start()` is exercised on its own.
    monkeypatch.setattr("discordbot.services.memory.git_history.memory_git.enabled", False)
    monkeypatch.setattr("discordbot.services.memory.git_history.memory_git._queue", None)
    monkeypatch.setattr("discordbot.services.memory.consolidation._last_consolidation", {})
    monkeypatch.setattr("discordbot.services.memory.regeneration._last_regeneration", {})
    monkeypatch.setattr("discordbot.services.memory.consolidation._consecutive_rejections", {})
    monkeypatch.setattr("discordbot.services.memory.inflight._db_tasks", set())
    # Point the memory_job engine at a throwaway reply.db so no test ever writes the
    # real file: every schedule_memory_update now persists, and those writes are
    # swallowed best-effort, so a missing swap would pass green while polluting the
    # real DB.
    _isolate_engine(
        monkeypatch=monkeypatch,
        target="discordbot.services.memory.database._engine",
        db_path=tmp_path / "memory_reply.db",
    )
    monkeypatch.setattr("discordbot.services.memory.database._token_sequence", count(start=1))
    monkeypatch.setattr("discordbot.services.memory.database._token_block_bases", {})
    # _scope_locks, staging_locks, _inflight_tasks, _pending_updates, _regeneration_tasks
    # and the memory semaphore are loop-local helpers that rebuild on the per-test event
    # loop, so they need no manual reset. An `asyncio.Task` left in a registry is unusable
    # on the next loop anyway, so the rebuild is the correctness rule rather than a test
    # convenience.
    return memories_dir


@pytest.fixture(autouse=True)
def usage_log_isolated_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points every usage recorder built during a test at a throwaway directory.

    Autouse because `UsageRecorder`'s default config reads the environment, so any cog a
    test constructs would otherwise append to the live `data/usage` file — the one store
    in this repo that nothing ever prunes. The kill-switch is pinned on for the same
    reason the directory is: `.env` is loaded at import, so a deployment that set
    `USAGE_LOG_ENABLED=false` would otherwise turn the recording assertions red on that
    checkout alone.
    """
    usage_dir = tmp_path / "usage"
    monkeypatch.setenv(name="USAGE_LOG_DIR", value=str(usage_dir))
    monkeypatch.setenv(name="USAGE_LOG_ENABLED", value="true")
    return usage_dir


@pytest.fixture(autouse=True)
def expansion_store_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points the pending-expansion table at a throwaway `reply.db`.

    Autouse for the reason `usage_log_isolated_dir` is: every expansion cog records its
    placeholder, the write is swallowed best-effort, so a test missing the swap would pass
    green while inserting rows into the live `reply.db` — where the next real restart would
    find them and try to expand a link nobody posted.
    """
    _isolate_engine(
        monkeypatch=monkeypatch,
        target="discordbot.utils.expansion_placeholder._engine",
        db_path=tmp_path / "expansion_reply.db",
    )


@pytest.fixture(autouse=True)
def cleanup_store_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points the pending public-message cleanup table at a throwaway `games.db`.

    Autouse for the reason `expansion_store_isolated` is: every expiring public message is
    recorded for deletion, the write is swallowed best-effort, so a test missing the swap
    would pass green while inserting rows into the live `games.db` — where the next real
    start would try to delete a message named by a test's fake ids.
    """
    _isolate_engine(
        monkeypatch=monkeypatch,
        target="discordbot.utils.message_cleanup._engine",
        db_path=tmp_path / "game_cleanup.db",
    )


@pytest.fixture(autouse=True)
def games_history_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points the Blackjack round history at a throwaway `games.db`.

    Autouse for the reason `expansion_store_isolated` is: every settled round records its
    history in a background task whose failure is swallowed, so a test that settles a table
    without the swap would pass green while writing rows into the live `games.db`.
    """
    _isolate_engine(
        monkeypatch=monkeypatch,
        target="discordbot.cogs.games.database._engine",
        db_path=tmp_path / "games_history.db",
    )


@pytest.fixture(autouse=True)
def file_api_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the Files API kill-switch on for every test.

    Autouse for the reason `usage_log_isolated_dir` pins its own switch: `LLMConfig` reads the
    environment and `.env` is loaded at import, so a deployment that set `FILE_API_ENABLED=false`
    to ride out a provider outage would otherwise turn every upload and Gemini-renderer
    assertion red on that checkout alone. A test about the switched-off path sets it back to
    false itself, which wins because `monkeypatch` applies in fixture-then-test order.
    """
    monkeypatch.setenv(name="FILE_API_ENABLED", value="true")


@pytest.fixture(autouse=True)
def media_hosting_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turns media hosting off for every planner a cog builds from the environment.

    Autouse because such a planner reads `.env`, and on a deployment's checkout that names the
    live serve directory: an oversize item a test delivered through it would be published there
    without the test noticing. A test about hosting builds its own config through
    `make_media_hosting_config`, which reads nothing from the environment.
    """
    monkeypatch.setenv(name="MEDIA_HOSTING_ENABLED", value="false")


@pytest.fixture(autouse=True)
def model_price_mirror_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the LiteLLM price-table mirror at a throwaway file.

    Autouse because any test reaching `load_model_info` mirrors the fetched table, and the
    live `data/` is no place for a 1.6MB file a test wrote (the same reason
    `usage_log_isolated_dir` exists). The table the loader holds is deliberately NOT reset
    here, since resetting per test would make every test that reaches it pay the fetch again;
    `tests/test_model_pricing.py` swaps its own in through `monkeypatch`, which puts the
    worker's back afterwards. Which table a worker ends up holding is therefore not
    deterministic — pinning that for the whole suite is #450.
    """
    mirror_path = tmp_path / "model_prices_and_context_window.json"
    monkeypatch.setattr("discordbot.utils.model_pricing.MODEL_INFO_CACHE_PATH", mirror_path)
    return mirror_path


@pytest.fixture(autouse=True)
def gemini_key_set_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leaves every test a keyless deployment whatever the checkout has configured.

    Autouse because an unconfigured deployment is what the reply tests are built for — they
    set the key explicitly when they are about a Gemini-only path — and leaving the real one
    in place made a test's outcome depend on whether the developer's `.env` happened to be
    visible, which in a git worktree is the parent checkout's file. `model_validate` does not
    save you either: it skips the settings sources only for the keys it is handed, so any
    field a test does not name still comes from the process environment.
    """
    monkeypatch.delenv(name="GEMINI_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def git_environment_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps every git a test runs inside the repository the test names.

    Autouse because git exports `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE` and friends to a hook
    or a `git rebase --exec` command run from a linked worktree, and a `git -C <tmp_path>` under
    those still acts on the enclosing repository, writing a test identity and `core.bare` into
    the checkout's own `.git/config`.
    """
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name=name, raising=False)
