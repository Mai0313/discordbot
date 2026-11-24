"""Guards ensuring every autouse isolation fixture in `tests/conftest.py` stays active.

A test that requests nothing but `tmp_path` must still never reach a live database, log directory,
git config, price mirror, or remote endpoint. An AST check over `tests/conftest.py` also ensures
that adding a new autouse fixture registers a matching guard here.
"""

import os
import ast
import socket
from pathlib import Path
from collections.abc import Callable

import pytest
import requests

from discordbot.utils import expansion_cog, model_pricing, sqlite_config, message_cleanup
from discordbot.typings import config as bot_config
from discordbot.cogs.games import database as games_database
from discordbot.cogs.log_msg import cog as log_msg_cog
from discordbot.services.memory import database as memory_database
from discordbot.services.economy import database as economy_database
from discordbot.cogs.parse_facebook.cog import FacebookCogs

import tests.conftest as root_conftest
from tests.helpers.source_tree import REPO_ROOT


def _assert_economy_isolated(tmp_path: Path) -> None:
    """Checks `economy_isolated_db` points at `tmp_path` with empty leaderboard caches."""
    assert economy_database._engine.url.database == str(tmp_path / "economy.db")
    assert economy_database._top_n_cache == {}
    assert economy_database._top_losers_cache == {}


def _assert_reply_isolated(tmp_path: Path) -> None:
    """Checks `reply_isolated_db` points `reply_db_engine` at `tmp_path` and resets token state."""
    assert sqlite_config.reply_db_engine.url.database == str(tmp_path / "reply.db")
    assert memory_database._token_block_bases == {}


def _assert_messages_isolated(tmp_path: Path) -> None:
    """Checks `messages_isolated_db` points `log_msg`'s engine at `tmp_path`."""
    assert log_msg_cog._engine.url.database == str(tmp_path / "messages.db")


def _assert_cleanup_isolated(tmp_path: Path) -> None:
    """Checks `cleanup_store_isolated` points `message_cleanup`'s engine at `tmp_path`."""
    assert message_cleanup._engine.url.database == str(tmp_path / "cleanup.db")


def _assert_games_isolated(tmp_path: Path) -> None:
    """Checks `games_history_isolated_db` points `games`'s engine at `tmp_path`."""
    assert games_database._engine.url.database == str(tmp_path / "games_history.db")


def _assert_bot_config_isolated(tmp_path: Path) -> None:
    """Checks `bot_config_isolated` redirects `LOCAL_CONFIG_PATH` into `tmp_path` and clears cache."""
    assert tmp_path / "config.toml" == bot_config.LOCAL_CONFIG_PATH
    assert bot_config._cached_bot_config is None
    assert "DISCORDBOT_LOCAL_CONFIG_DISABLED" not in os.environ


def _assert_usage_log_isolated(tmp_path: Path) -> None:
    """Checks `usage_log_isolated_dir` redirects `USAGE_LOG_DIR` into `tmp_path`."""
    assert os.environ["USAGE_LOG_DIR"] == str(tmp_path / "usage")


def _assert_model_price_mirror_isolated(tmp_path: Path) -> None:
    """Checks `model_price_mirror_isolated` holds an empty table and throwaway mirror path."""
    assert tmp_path / "model_prices_and_context_window.json" == model_pricing.MODEL_INFO_CACHE_PATH
    assert model_pricing._LOADED_TABLE is not None
    assert model_pricing._LOADED_TABLE.prices == {}


def _assert_upgraded_sections_reset(tmp_path: Path) -> None:
    """Checks `upgraded_sections_reset` pre-marks every repo-root database section."""
    del tmp_path
    repo_root = REPO_ROOT.resolve()
    assert sqlite_config._upgraded_sections == {
        (repo_root, section) for section in ("cleanup", "economy", "games", "reply", "messages")
    }


def _assert_git_environment_isolated(tmp_path: Path) -> None:
    """Checks `git_environment_isolated` neutralizes global/system git config and ceilings."""
    assert os.environ["GIT_CONFIG_GLOBAL"] == os.devnull
    assert os.environ["GIT_CONFIG_NOSYSTEM"] == "1"
    assert os.environ["GIT_CEILING_DIRECTORIES"] == str(tmp_path.parent)
    assert "GIT_DIR" not in os.environ
    assert "GIT_WORK_TREE" not in os.environ


def _assert_card_images_unreachable(tmp_path: Path) -> None:
    """Checks `card_images_unreachable` refuses card image fetches."""
    del tmp_path
    with pytest.raises(requests.ConnectionError):
        expansion_cog.fetch_card_image(url="https://example.test/image.png")


def _assert_card_videos_unreachable(tmp_path: Path) -> None:
    """Checks `card_videos_unreachable` refuses card video downloads on both paths."""
    with pytest.raises(requests.ConnectionError):
        expansion_cog.stream_to_file(
            url="https://example.test/clip.mp4",
            filepath=tmp_path / "clip.mp4",
            headers={},
            timeout=1.0,
        )
    with pytest.raises(requests.ConnectionError):
        FacebookCogs.video_downloader_factory(output_folder=str(tmp_path)).download(
            url="https://example.test/clip.mp4"
        )


def _assert_network_unreachable(tmp_path: Path) -> None:
    """Checks `network_unreachable`'s installed audit hook refuses remote lookups."""
    del tmp_path
    assert root_conftest._remote_attempts == []
    try:
        with pytest.raises(root_conftest.NetworkAccessError):
            socket.getaddrinfo(host="example.test", port=443)
        assert root_conftest._remote_attempts == ["example.test"]
    finally:
        root_conftest._remote_attempts.clear()


_AUTOUSE_CHECKS: dict[str, Callable[[Path], None]] = {
    "economy_isolated_db": _assert_economy_isolated,
    "reply_isolated_db": _assert_reply_isolated,
    "messages_isolated_db": _assert_messages_isolated,
    "cleanup_store_isolated": _assert_cleanup_isolated,
    "games_history_isolated_db": _assert_games_isolated,
    "bot_config_isolated": _assert_bot_config_isolated,
    "usage_log_isolated_dir": _assert_usage_log_isolated,
    "model_price_mirror_isolated": _assert_model_price_mirror_isolated,
    "upgraded_sections_reset": _assert_upgraded_sections_reset,
    "git_environment_isolated": _assert_git_environment_isolated,
    "card_images_unreachable": _assert_card_images_unreachable,
    "card_videos_unreachable": _assert_card_videos_unreachable,
    "network_unreachable": _assert_network_unreachable,
}


def _conftest_autouse_fixtures() -> set[str]:
    """Returns every `@pytest.fixture(autouse=True)` function name declared in `tests/conftest.py`."""
    tree = ast.parse(source=(REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if any(
                kw.arg == "autouse"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in decorator.keywords
            ):
                names.add(node.name)
    return names


@pytest.mark.parametrize(
    argnames="fixture_name", argvalues=sorted(_AUTOUSE_CHECKS), ids=sorted(_AUTOUSE_CHECKS)
)
def test_autouse_isolation_guard_is_active_without_opting_in(
    tmp_path: Path, fixture_name: str
) -> None:
    """Every autouse isolation fixture applies to a test that requests nothing but `tmp_path`."""
    _AUTOUSE_CHECKS[fixture_name](tmp_path)


def test_every_conftest_autouse_fixture_has_an_isolation_guard() -> None:
    """Adding or removing an autouse fixture in `tests/conftest.py` updates `_AUTOUSE_CHECKS`."""
    assert set(_AUTOUSE_CHECKS) == _conftest_autouse_fixtures()
