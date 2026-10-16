"""Shared pytest fixtures.

Before any test module imports `discordbot`, this module turns `.env` loading off and clears
every variable `.env.example` names from the process, so no setting reaches a test from the
developer's `.env` or shell: a test that needs one sets it with `monkeypatch.setenv`. It also
refuses every connection past loopback for the whole run (`_refuse_remote`).

Each `*_isolated_db` fixture points the owning module's module-level engine at a fresh
`tmp_path` SQLite file for one test, all through `_isolate_engine`, and every one is autouse.
`memory_isolated_dir` is the one a test requests, and it covers more than a directory: the
store dir, the `memory_job` engine, the process-local caches, counters and task registries the
store and pipeline hold, and the git committer. The autouse fixtures are the other half of that
isolation, keeping a real deployment's databases and files out of every test whether or not it
asked for them; the memory store stays live for a test that does not request
`memory_isolated_dir`.
"""

import os
import sys
from time import monotonic
from typing import Any
from pathlib import Path
import ipaddress
from itertools import count
from collections.abc import Iterator

import pytest
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine

from tests.helpers.source_tree import env_example_names

# `discordbot` loads `.env` when it is imported. Settings match a name in any case, and the
# SDKs read variables of their own (google-genai falls back to `GOOGLE_API_KEY` for the empty key
# a keyless deployment passes), so every name under their prefixes goes too, as does a proxy,
# which would carry a request out through a loopback port `_refuse_remote` lets through.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
cleared = env_example_names() | {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}
for name in list(os.environ):
    if name.upper() in cleared or name.upper().startswith((
        "OPENAI_",
        "GOOGLE_",
        "GEMINI_",
        "ANTHROPIC_",
        "XAI_",
    )):
        del os.environ[name]

from discordbot.utils.model_pricing import _LoadedTable  # noqa: E402 -- `.env` must be off first


class NetworkAccessError(Exception):
    """A test reached past loopback.

    Not an `OSError`, which the code under test retries or degrades on: this must surface.
    """


_remote_attempts: list[str] = []


def _refuse_remote(event: str, args: tuple[Any, ...]) -> None:
    """Audit hook raising on, and recording, any DNS lookup or connect past loopback.

    It sees only what goes through Python's `socket` module: a client that resolves and
    connects in C, such as grpc (`xai-sdk`), goes past it. Running the suite in a network
    namespace (`bwrap --unshare-net --dev-bind / / uv run pytest`) covers those too.
    """
    if event == "socket.getaddrinfo":
        host = args[0]
    elif event == "socket.connect" and isinstance(args[1], tuple):
        host = args[1][0]
    else:
        return
    if isinstance(host, bytes):
        host = host.decode()
    if host in {None, "", "localhost"}:
        return
    try:
        if ipaddress.ip_address(address=host.partition("%")[0]).is_loopback:
            return
    except ValueError:
        pass
    _remote_attempts.append(host)
    raise NetworkAccessError(f"a test reached past loopback: {host}")


sys.addaudithook(_refuse_remote)


@pytest.fixture(autouse=True)
def network_unreachable() -> Iterator[None]:
    """Fails a test that reached past loopback, even when the code under test swallowed it."""
    _remote_attempts.clear()
    yield
    assert not _remote_attempts, f"reached past loopback: {_remote_attempts}"


def _isolate_engine(monkeypatch: pytest.MonkeyPatch, target: str, db_path: Path) -> None:
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


@pytest.fixture(autouse=True)
def research_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points the research session table at a throwaway `reply.db`.

    Autouse for the reason `expansion_store_isolated` is: a test that launches a run without the
    swap would leave a `researching` row in the live `reply.db`, which the next real start would
    try to resume.
    """
    _isolate_engine(
        monkeypatch=monkeypatch,
        target="discordbot.cogs.research.database._engine",
        db_path=tmp_path / "reply.db",
    )


@pytest.fixture(autouse=True)
def ask_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points the `/ask` conversation table at a throwaway `reply.db`.

    Autouse for the reason `expansion_store_isolated` is: every `/ask` turn records itself and a
    failed write is swallowed, so a test missing the swap would pass green while writing its fake
    turns into the live table.
    """
    _isolate_engine(
        monkeypatch=monkeypatch,
        target="discordbot.cogs.gen_reply.ask_store._engine",
        db_path=tmp_path / "reply.db",
    )


@pytest.fixture(autouse=True)
def messages_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points the message log at a throwaway `messages.db`.

    Autouse for the reason `expansion_store_isolated` is: the log writes in a background task
    whose failure is swallowed, so a test missing the swap would pass green while inserting its
    fake messages into the live file.
    """
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
    monkeypatch.setattr("discordbot.services.memory.catchup._last_catchup", {})
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

    Autouse because `UsageRecorder`'s default config writes under the working directory, so
    any cog a test constructs would otherwise append to the live `data/usage` file — the one
    store in this repo that nothing ever prunes.
    """
    usage_dir = tmp_path / "usage"
    monkeypatch.setenv(name="USAGE_LOG_DIR", value=str(usage_dir))
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
def model_price_mirror_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Holds an empty LiteLLM price table and points its mirror at a throwaway file.

    Autouse because the first lookup would otherwise fetch the table from GitHub, and what a
    worker then held depended on whether it was online. Held as upstream just served it, the
    empty table never starts a fetch, and every model reads as unknown in every run. A test
    about rates or modalities installs the entries it needs; `tests/utils/test_model_pricing.py`
    puts `None` back to drive the loading chain itself, whose mirror lands here rather than in
    the live `data/`.
    """
    monkeypatch.setattr(
        "discordbot.utils.model_pricing._LOADED_TABLE",
        _LoadedTable(prices={}, served_at=monotonic()),
    )
    mirror_path = tmp_path / "model_prices_and_context_window.json"
    monkeypatch.setattr("discordbot.utils.model_pricing.MODEL_INFO_CACHE_PATH", mirror_path)
    return mirror_path


@pytest.fixture(autouse=True)
def git_environment_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps every git a test runs inside the repository the test names, with no user config.

    Autouse because git exports `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE` and friends to a hook
    or a `git rebase --exec` command run from a linked worktree, and a `git -C <tmp_path>` under
    those still acts on the enclosing repository, writing a test identity and `core.bare` into
    the checkout's own `.git/config`. The developer's global and system config (a signing key,
    a hooks path) stays out too, and git never looks for a repository above the test's own
    directory, which a `--basetemp` inside a checkout would otherwise hand it.
    """
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name=name, raising=False)
    monkeypatch.setenv(name="GIT_CONFIG_GLOBAL", value=os.devnull)
    monkeypatch.setenv(name="GIT_CONFIG_NOSYSTEM", value="1")
    monkeypatch.setenv(name="GIT_CEILING_DIRECTORIES", value=str(tmp_path.parent))
