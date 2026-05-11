"""Migrates the legacy per-channel message log DB into one messages table."""

import logging
from pathlib import Path
import sqlite3
import argparse
from dataclasses import dataclass
from urllib.parse import quote
from collections.abc import Sequence

LOGGER = logging.getLogger(name=__name__)

CREATE_MESSAGES_TABLE_SQL = """
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    author TEXT,
    author_id TEXT,
    content TEXT,
    created_at TEXT,
    channel_name TEXT,
    channel_id TEXT,
    attachments TEXT,
    stickers TEXT
)
"""

CREATE_MESSAGES_INDEX_SQL: tuple[str, ...] = (
    "CREATE INDEX ix_messages_created_at ON messages(created_at)",
    "CREATE INDEX ix_messages_channel_id_created_at ON messages(channel_id, created_at)",
    "CREATE INDEX ix_messages_author_id_created_at ON messages(author_id, created_at)",
)

LEGACY_TABLE_QUERY = """
SELECT name
FROM source_db.sqlite_master
WHERE type = 'table'
  AND (name GLOB 'channel_*' OR name GLOB 'DM_*')
ORDER BY name
"""


@dataclass(frozen=True)
class MigrationSummary:
    """Result summary for one message DB migration."""

    table_count: int
    row_count: int
    integrity_check: str


def _sqlite_readonly_uri(path: Path) -> str:
    """Builds a SQLite read-only URI for ``path``."""
    return f"file:{quote(string=str(path.resolve()), safe='/')}?mode=ro"


def _quote_identifier(identifier: str) -> str:
    """Quotes a SQLite identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def _source_type_for_table(table_name: str) -> str:
    """Returns the canonical source type for a legacy table name."""
    if table_name.startswith("DM_"):
        return "dm"
    return "guild"


def _connect_dest(dest: Path) -> sqlite3.Connection:
    """Opens the destination DB and applies write-friendly PRAGMAs."""
    conn = sqlite3.connect(database=str(dest), uri=True)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _prepare_dest(conn: sqlite3.Connection, source: Path) -> list[str]:
    """Creates the destination schema and attaches the source DB read-only."""
    conn.execute("ATTACH DATABASE ? AS source_db", (_sqlite_readonly_uri(path=source),))
    conn.execute(CREATE_MESSAGES_TABLE_SQL)
    rows = conn.execute(LEGACY_TABLE_QUERY).fetchall()
    return [row[0] for row in rows]


def _copy_table(conn: sqlite3.Connection, table_name: str) -> int:
    """Copies one legacy table into the destination messages table."""
    quoted_table = _quote_identifier(identifier=table_name)
    source_type = _source_type_for_table(table_name=table_name)
    cursor = conn.execute(
        f"""
        INSERT INTO messages (
            source_type,
            author,
            author_id,
            content,
            created_at,
            channel_name,
            channel_id,
            attachments,
            stickers
        )
        SELECT
            ?,
            CAST(author AS TEXT),
            CAST(author_id AS TEXT),
            CAST(content AS TEXT),
            CAST(created_at AS TEXT),
            CAST(channel_name AS TEXT),
            CAST(channel_id AS TEXT),
            CAST(attachments AS TEXT),
            CAST(stickers AS TEXT)
        FROM source_db.{quoted_table}
        """,  # noqa: S608 -- table name comes from sqlite_master and is quoted as an identifier.
        (source_type,),
    )
    return cursor.rowcount


def migrate_messages_db(
    *, source: Path, dest: Path, overwrite: bool = False, progress_every: int = 1000
) -> MigrationSummary:
    """Migrates a legacy message DB into a fresh canonical message DB.

    Args:
        source: Existing per-channel/per-DM SQLite database.
        dest: Destination SQLite database to create.
        overwrite: Whether an existing destination file may be replaced.
        progress_every: Log progress after this many legacy tables. Use 0 to disable progress logs.

    Returns:
        A summary containing migrated table count, row count, and integrity result.

    Raises:
        FileNotFoundError: The source DB does not exist.
        FileExistsError: The destination exists and ``overwrite`` is false.
        ValueError: Source and destination resolve to the same path.
        RuntimeError: The migrated DB fails SQLite integrity check.
    """
    source = source.resolve()
    dest = dest.resolve()
    if source == dest:
        raise ValueError("source and dest must be different files")
    if not source.exists():
        raise FileNotFoundError(source)
    if dest.exists():
        if not overwrite:
            raise FileExistsError(dest)
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect_dest(dest=dest)
    try:
        tables = _prepare_dest(conn=conn, source=source)
        row_count = 0
        for index, table_name in enumerate(tables, start=1):
            row_count += _copy_table(conn=conn, table_name=table_name)
            if progress_every > 0 and index % progress_every == 0:
                LOGGER.info(
                    "Migrated %s/%s legacy tables (%s rows)",
                    f"{index:,}",
                    f"{len(tables):,}",
                    f"{row_count:,}",
                )

        for statement in CREATE_MESSAGES_INDEX_SQL:
            conn.execute(statement)
        conn.execute("PRAGMA optimize")
        conn.commit()

        integrity_check = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity_check != "ok":
            raise RuntimeError(f"SQLite integrity_check failed: {integrity_check}")
        return MigrationSummary(
            table_count=len(tables), row_count=row_count, integrity_check=integrity_check
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy message log tables into one messages table."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/messages.backup.db"),
        help="Legacy source DB to read without modifying.",
    )
    parser.add_argument(
        "--dest", type=Path, default=Path("data/messages.db"), help="New destination DB to create."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace the destination DB if it already exists."
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Log progress after this many legacy tables; use 0 to disable.",
    )
    return parser.parse_args(args=argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the message DB migration CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv=argv)
    summary = migrate_messages_db(
        source=args.source,
        dest=args.dest,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    LOGGER.info(
        "Migration complete: %s tables, %s rows, integrity_check=%s",
        f"{summary.table_count:,}",
        f"{summary.row_count:,}",
        summary.integrity_check,
    )


if __name__ == "__main__":
    main()
