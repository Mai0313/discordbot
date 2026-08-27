"""Shared pytest fixtures.

Each `*_isolated_db` fixture points the owning module's module-level engine at a fresh
`tmp_path` SQLite file for one test and disposes it afterwards. `memory_isolated_dir` covers
more than a directory: the store dir, the `memory_job` engine, the process-local caches,
counters and task registries the store and pipeline hold, and the git committer. The autouse
fixtures are the other half of that isolation, keeping a real deployment's `.env` and `data/`
out of every test whether or not it asked for them.
"""

import os
from pathlib import Path
from itertools import count
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs.feedback.database import Base as FeedbackBase
from discordbot.cogs.research.database import Base as ResearchBase
from discordbot.cogs.gen_reply.ask_store import Base as AskTurnBase
from discordbot.services.economy.database import Base


@pytest.fixture
async def economy_isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Per-test SQLite file with the full economy schema."""
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
    """Per-test SQLite file with the research schema (reply.db)."""
    research_db_path = tmp_path / "reply.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{research_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(ResearchBase.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs.research.database._engine", engine)
    yield
    await engine.dispose()


@pytest.fixture
async def ask_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Per-test SQLite file with the `/ask` conversation schema (reply.db)."""
    ask_db_path = tmp_path / "reply.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{ask_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(AskTurnBase.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs.gen_reply.ask_store._engine", engine)
    yield
    await engine.dispose()


@pytest.fixture
async def feedback_isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Per-test SQLite file with the user-report schema (feedback.db)."""
    feedback_db_path = tmp_path / "feedback.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{feedback_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(FeedbackBase.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs.feedback.database._engine", engine)
    yield
    await engine.dispose()


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
    # real DB. NullPool closes each connection on return (no async dispose needed in
    # this sync fixture); the schema bootstraps lazily on the first helper call.
    memory_db_engine = create_async_engine(
        url=f"sqlite+aiosqlite:///{tmp_path / 'memory_reply.db'}", poolclass=NullPool
    )
    monkeypatch.setattr("discordbot.services.memory.database._engine", memory_db_engine)
    monkeypatch.setattr("discordbot.services.memory.database._token_sequence", count(start=1))
    monkeypatch.setattr("discordbot.services.memory.database._token_block_bases", {})
    # _scope_locks, _staging_locks, _inflight_tasks, _pending_updates, _regeneration_tasks
    # and the memory semaphore are loop-local helpers that rebuild on the per-test event
    # loop, so they need no manual reset. The two in-flight registries used to be reset by
    # hand here; `LoopLocalRegistry` is what took that over, and an `asyncio.Task` left in
    # one is unusable on the next loop anyway, so the rebuild is the correctness rule
    # rather than a test convenience.
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
def feedback_env_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps a real deployment's reporting credentials out of every test.

    Autouse because `FeedbackConfig` reads the environment and `.env` is loaded at import,
    which in a git worktree is the *parent* checkout's file. Without this a machine with a
    configured GitHub App quietly turns "no credentials" tests into "credentials present"
    ones — and `model_validate` does not save you: it skips the settings sources only for
    the keys it is handed, so any field a test does not name still comes from the process
    environment.
    """
    for name in (
        "FEEDBACK_ENABLED",
        "FEEDBACK_GITHUB_TOKEN",
        "FEEDBACK_GITHUB_APP_ID",
        "FEEDBACK_GITHUB_APP_PRIVATE_KEY_PATH",
        "FEEDBACK_GITHUB_REPOSITORY",
        "FEEDBACK_CONTACT",
        "FEEDBACK_MAX_OPEN_REPORTS",
        "FEEDBACK_SUBMIT_COOLDOWN_SECONDS",
    ):
        monkeypatch.delenv(name=name, raising=False)


@pytest.fixture(autouse=True)
def gemini_key_set_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leaves every test a keyless deployment whatever the checkout has configured.

    Autouse for the reason `feedback_env_isolated` is: an unconfigured deployment is what the
    reply tests are built for — they set the key explicitly when they are about a Gemini-only
    path — and leaving the real one in place made a test's outcome depend on whether the
    developer's `.env` happened to be visible, which in a git worktree is the parent
    checkout's file.

    The numbered variables go too. Nothing reads them since the key balancer was reverted,
    but they stay live in a real deployment's `.env` for the proxy's own pooled media
    aliases, and isolation should not have to track which spellings could come back.
    """
    monkeypatch.delenv(name="GEMINI_API_KEY", raising=False)
    for name in [name for name in os.environ if name.startswith("GEMINI_API_KEY_")]:
        monkeypatch.delenv(name=name, raising=False)
