"""Every database's migrations build exactly the schema its models declare.

A model change that ships without its migration fails here, before a deployment boots on a
schema the code does not expect.
"""

from pathlib import Path

import pytest
from alembic import command

from tests.helpers.migrations import migration_config


@pytest.mark.parametrize("database", ["economy", "games", "reply", "messages"])
def test_migrations_build_the_schema_the_models_declare(
    database: str, migrated_databases: dict[str, Path]
) -> None:
    """`alembic check` finds nothing to generate against a database migrated to head."""
    command.check(config=migration_config(database=database, db_path=migrated_databases[database]))
