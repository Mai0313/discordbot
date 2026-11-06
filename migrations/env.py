"""Alembic environment shared by every database; the `alembic.ini` section names the database."""

from typing import Any, Literal

from alembic import context
from sqlalchemy import Connection, event, engine_from_config
from alembic.autogenerate.api import AutogenContext

from discordbot.cogs.log_msg.cog import metadata as messages_metadata
from discordbot.cogs.games.database import Base as GamesBase
from discordbot.utils.stored_integer import StoredInteger
from discordbot.utils.message_cleanup import metadata as cleanup_metadata
from discordbot.cogs.research.database import Base as ResearchBase
from discordbot.cogs.gen_reply.ask_store import Base as AskBase
from discordbot.services.memory.database import Base as MemoryBase
from discordbot.services.economy.database import Base as EconomyBase
from discordbot.utils.expansion_placeholder import Base as ExpansionBase

target_metadata = {
    "economy": [EconomyBase.metadata],
    "games": [GamesBase.metadata, cleanup_metadata],
    "reply": [
        ResearchBase.metadata,
        AskBase.metadata,
        ExpansionBase.metadata,
        MemoryBase.metadata,
    ],
    "messages": [messages_metadata],
}[context.config.config_ini_section]

engine = engine_from_config(
    configuration=context.config.get_section(name=context.config.config_ini_section, default={}),
    prefix="sqlalchemy.",
)


# pysqlite runs DDL outside the transaction it opens for DML, so a migration failing halfway
# through, a batch table rebuild included, would stay half applied. SQLAlchemy's documented
# fix is to hand every transaction to the driver explicitly.
def _driver_autocommit(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Stops pysqlite from opening and committing transactions on its own."""
    dbapi_connection.isolation_level = None


def _begin(connection: Connection) -> None:
    """Opens the transaction pysqlite no longer opens itself."""
    connection.exec_driver_sql("BEGIN")


def _render_item(type_: str, obj: object, autogen_context: AutogenContext) -> str | Literal[False]:
    """Renders `StoredInteger` columns with an import, so a generated revision runs as written."""
    if type_ == "type" and isinstance(obj, StoredInteger):
        autogen_context.imports.add("from discordbot.utils.stored_integer import StoredInteger")
        return "StoredInteger()"
    return False


event.listen(target=engine, identifier="connect", fn=_driver_autocommit)
event.listen(target=engine, identifier="begin", fn=_begin)

with engine.connect() as connection:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_item=_render_item,
        render_as_batch=True,
        transactional_ddl=True,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()
