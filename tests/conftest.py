"""Shared pytest fixtures: the isolation the suite runs every stateful test behind.

Each subsystem here keeps its storage handle as a module-level global read per call — an
`AsyncEngine` for the SQLite-backed ones, a `Path` for the file-backed memory store — which is
what makes `monkeypatch.setattr` on that global the supported isolation seam (the engine is
module-level for a pool-leak reason of its own; `services/economy/database.py` has the why). A
fixture swaps the global onto `tmp_path` and, for a SQLite one, creates the schema eagerly so a
read-only test still finds its tables and disposes the engine on teardown. Nothing in a test run
should ever open the checked-in `data/database/*.db`, `data/memories/` or `data/usage/`.

`economy_isolated_db` lives here instead of being copy-pasted into every test module that
exercises the ledger, and `fishing_isolated_db` layers on it because a cast settles across both
files. `research_isolated_db` and `memory_isolated_dir` both stand in for `reply.db`, which two
independent modules share in production, so a test needing research rows and memory job rows takes
both. `memory_isolated_dir` is the widest of them: the memory store is file-backed and its pipeline
keeps caches, cooldowns and registries for the life of the process, so redirecting the directory
alone would leak one test's state into the next. `usage_log_isolated_dir` is autouse because a
recorder is built from the environment by any cog a test constructs.

`tests/conftest.py` and `tests/helpers/` are the two places under `tests/` that owe a full typed
`Args:` block, being what a test file calls into rather than receives fixtures from;
`tests/test_docstrings.py` pins that split.
"""

from pathlib import Path
from itertools import count
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs.research.database import Base as ResearchBase
from discordbot.services.economy.database import Base
from discordbot.cogs.games.fishing.database import Base as FishingBase


@pytest.fixture
async def economy_isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Points the economy ledger at a per-test SQLite file carrying the full schema.

    `create_all` alone, so none of the seed rows `_ensure_schema` writes (the jackpot pools, the
    casino ledger) exist until a test calls into a path that bootstraps them.

    Args:
        tmp_path (Path): Directory the throwaway `economy.db` is created in.
        monkeypatch (pytest.MonkeyPatch): Swaps `services.economy.database._engine`.
    """
    economy_db_path = tmp_path / "economy.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{economy_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr("discordbot.services.economy.database._engine", engine)
    yield
    await engine.dispose()


@pytest.fixture
async def research_isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Points the research session store at a per-test `reply.db`.

    Covers the `research` table only. `services/memory/database.py` keeps a second engine over the
    same production file for its own tables, and `memory_isolated_dir` redirects that one, so a
    test touching both takes both fixtures.

    Args:
        tmp_path (Path): Directory the throwaway `reply.db` is created in.
        monkeypatch (pytest.MonkeyPatch): Swaps `cogs.research.database._engine` and clears its
            per-engine schema-bootstrap cache.
    """
    research_db_path = tmp_path / "reply.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{research_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(ResearchBase.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs.research.database._engine", engine)
    monkeypatch.setattr("discordbot.cogs.research.database._schema_ready_for", None)
    yield
    await engine.dispose()


@pytest.fixture
def memory_isolated_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirects every durable holding of the memory subsystem at `tmp_path` for one test.

    There are three of them and missing any one leaks across tests: the markdown store's root
    directory, the process-local caches, cooldowns and in-flight registries `store.py` /
    `pipeline.py` keep for the life of the process, and the `memory_job` engine over `reply.db`.
    The git committer is pinned off on top, so no test can ever commit against the real store.

    The returned directory is deliberately not created. The store makes each scope directory on
    its first write, and a test that needs the root on disk beforehand (`test_memory_git.py`)
    creates it itself.

    Args:
        tmp_path (Path): Directory the memory root and the throwaway `reply.db` are created in.
        monkeypatch (pytest.MonkeyPatch): Swaps the store directory, the memory `reply.db` engine,
            and every process-local dict the store and pipeline hold.

    Returns:
        The redirected `data/memories` root, which does not exist yet.
    """
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
    monkeypatch.setattr("discordbot.services.memory.pipeline._inflight_tasks", {})
    monkeypatch.setattr("discordbot.services.memory.pipeline._pending_updates", {})
    monkeypatch.setattr("discordbot.services.memory.pipeline._inflight_loop", None)
    monkeypatch.setattr("discordbot.services.memory.pipeline._last_consolidation", {})
    monkeypatch.setattr("discordbot.services.memory.pipeline._last_regeneration", {})
    monkeypatch.setattr("discordbot.services.memory.pipeline._consecutive_rejections", {})
    monkeypatch.setattr("discordbot.services.memory.pipeline._db_tasks", set())
    # Point the memory_job engine at a throwaway reply.db so no test ever writes the
    # real file: every schedule_memory_update now persists, and those writes are
    # swallowed best-effort, so a missing swap would pass green while polluting the
    # real DB. NullPool closes each connection on return (no async dispose needed in
    # this sync fixture); the schema bootstraps lazily on the first helper call.
    memory_db_engine = create_async_engine(
        url=f"sqlite+aiosqlite:///{tmp_path / 'memory_reply.db'}", poolclass=NullPool
    )
    monkeypatch.setattr("discordbot.services.memory.database._engine", memory_db_engine)
    monkeypatch.setattr("discordbot.services.memory.database._schema_ready_for", None)
    monkeypatch.setattr("discordbot.services.memory.database._token_sequence", count(start=1))
    monkeypatch.setattr("discordbot.services.memory.database._token_block_bases", {})
    # _scope_locks, _staging_locks, _regeneration_tasks, and the memory semaphore are
    # loop-local helpers that rebuild on the per-test event loop, so they need no reset.
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

    Args:
        tmp_path (Path): Directory the throwaway usage dir is placed under.
        monkeypatch (pytest.MonkeyPatch): Sets `USAGE_LOG_DIR` and `USAGE_LOG_ENABLED` for the
            test, since the recorder reads them rather than taking a handle.

    Returns:
        The directory every recorder built during the test appends into. It is created by the
        first record written, not here.
    """
    usage_dir = tmp_path / "usage"
    monkeypatch.setenv(name="USAGE_LOG_DIR", value=str(usage_dir))
    monkeypatch.setenv(name="USAGE_LOG_ENABLED", value="true")
    return usage_dir


@pytest.fixture
async def fishing_isolated_db(
    economy_isolated_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Points the fishing tables at a per-test SQLite file, over an isolated economy ledger too.

    Fishing settles across two databases — a gear purchase debits the wallet and a payout credits
    it — so it depends on `economy_isolated_db` rather than letting those halves land in the real
    `economy.db`. The catalog is not seeded: `defaults.py` is carried in by a test that wants it.

    Args:
        economy_isolated_db (None): The economy isolation this fixture layers on; requested for
            its effect, never read.
        tmp_path (Path): Directory the throwaway fishing DB is created in.
        monkeypatch (pytest.MonkeyPatch): Swaps `cogs.games.fishing.database._engine` and clears
            its per-engine schema-bootstrap cache.
    """
    fishing_db_path = tmp_path / "fishing.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{fishing_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(FishingBase.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs.games.fishing.database._engine", engine)
    monkeypatch.setattr("discordbot.cogs.games.fishing.database._schema_ready_for", None)
    yield
    await engine.dispose()
