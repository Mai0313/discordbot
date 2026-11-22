"""Every database's migrations build exactly the schema its models declare.

A model change that ships without its migration fails here, before a deployment boots on a
schema the code does not expect.
"""

import sys
import json
from pathlib import Path
import subprocess

import pytest
from alembic import command

from tests.helpers.migrations import migration_config
from tests.helpers.source_tree import REPO_ROOT


@pytest.mark.parametrize("database", ["cleanup", "economy", "games", "reply", "messages"])
def test_migrations_build_the_schema_the_models_declare(
    database: str, migrated_databases: dict[str, Path]
) -> None:
    """`alembic check` finds nothing to generate against a database migrated to head."""
    command.check(config=migration_config(database=database, db_path=migrated_databases[database]))


@pytest.mark.parametrize(
    ("section", "forbidden_prefixes"),
    [
        (
            "cleanup",
            ("discordbot.cogs.", "discordbot.services.economy", "discordbot.services.memory"),
        ),
        (
            "economy",
            ("discordbot.cogs.", "discordbot.utils.message_cleanup", "discordbot.services.memory"),
        ),
        (
            "games",
            (
                "discordbot.cogs.log_msg",
                "discordbot.cogs.research",
                "discordbot.cogs.gen_reply",
                "discordbot.services.economy",
            ),
        ),
        (
            "reply",
            (
                "discordbot.cogs.log_msg",
                "discordbot.cogs.games",
                "discordbot.services.economy",
                "discordbot.utils.message_cleanup",
            ),
        ),
        (
            "messages",
            (
                "discordbot.cogs.games",
                "discordbot.cogs.research",
                "discordbot.cogs.gen_reply",
                "discordbot.services.economy",
            ),
        ),
    ],
)
def test_alembic_env_imports_only_the_active_sections_metadata(
    tmp_path: Path, section: str, forbidden_prefixes: tuple[str, ...]
) -> None:
    """Upgrading one section never imports metadata belonging to other sections or cogs."""
    db_path = tmp_path / f"{section}.db"
    probe = (
        "import json, sys; "
        "from pathlib import Path; "
        "from alembic import command; "
        "from alembic.config import Config; "
        f"cfg = Config(file_={str(REPO_ROOT / 'alembic.ini')!r}, ini_section={section!r}); "
        f"cfg.set_main_option(name='sqlalchemy.url', value={f'sqlite:///{db_path}'!r}); "
        "command.upgrade(config=cfg, revision='head'); "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('discordbot.'))))"
    )
    result = subprocess.run(  # noqa: S603 -- runs the test interpreter with fixed literal probe code
        args=[sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    loaded = json.loads(s=result.stdout.strip().splitlines()[-1])
    offenders = [name for name in loaded if name.startswith(forbidden_prefixes)]
    assert not offenders, f"section {section!r} imported unrelated modules: {offenders}"


def test_reply_upgrade_survives_removed_optional_cogs(tmp_path: Path) -> None:
    """Deleting `cogs/research` or `cogs/gen_reply` does not crash `alembic -n reply upgrade`."""
    db_path = tmp_path / "reply.db"
    probe = (
        "import sys; "
        "from alembic import command; "
        "from alembic.config import Config; "
        "sys.modules['discordbot.cogs.research'] = None; "
        "sys.modules['discordbot.cogs.gen_reply'] = None; "
        f"cfg = Config(file_={str(REPO_ROOT / 'alembic.ini')!r}, ini_section='reply'); "
        f"cfg.set_main_option(name='sqlalchemy.url', value={f'sqlite:///{db_path}'!r}); "
        "command.upgrade(config=cfg, revision='head')"
    )
    subprocess.run(  # noqa: S603 -- runs the test interpreter with fixed literal probe code
        args=[sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert db_path.is_file()
