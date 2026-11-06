"""Alembic configuration pointed at a test's own SQLite file."""

from pathlib import Path

from alembic.config import Config

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def migration_config(database: str, db_path: Path) -> Config:
    """Returns the `alembic.ini` section for `database`, aimed at `db_path` instead of `data/`.

    Args:
        database (str): The `alembic.ini` section: `economy`, `games`, `reply` or `messages`.
        db_path (Path): The SQLite file to migrate or check.
    """
    config = Config(file_=ALEMBIC_INI, ini_section=database)
    config.set_main_option(name="sqlalchemy.url", value=f"sqlite:///{db_path}")
    return config
