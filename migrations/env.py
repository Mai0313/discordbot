"""Alembic environment shared by every database; the `alembic.ini` section names the database."""

from typing import Any, Literal
import importlib

from alembic import context
from sqlalchemy import MetaData, Connection, event, engine_from_config
from alembic.autogenerate.api import AutogenContext

from discordbot.utils.stored_integer import StoredInteger

# `(module_path, attribute, optional_cog)` pairs per `alembic.ini` section.
# Optional cogs under `cogs/` may be deleted in minimal deployments; skipping an absent cog's
# own package keeps `alembic upgrade` working for surviving cogs that share that database.
_SECTION_METADATA: dict[str, tuple[tuple[str, str, bool], ...]] = {
    "cleanup": (("discordbot.utils.message_cleanup", "metadata", False),),
    "economy": (("discordbot.services.economy.database", "Base", False),),
    "games": (("discordbot.cogs.games.database", "Base", False),),
    "reply": (
        ("discordbot.cogs.research.database", "Base", True),
        ("discordbot.cogs.gen_reply.ask_store", "Base", True),
        ("discordbot.cogs.gen_reply.feedback_store", "Base", True),
        ("discordbot.utils.expansion_placeholder", "Base", False),
        ("discordbot.services.memory.database", "Base", False),
    ),
    "messages": (("discordbot.cogs.log_msg.cog", "metadata", False),),
    "settings": (("discordbot.utils.server_settings", "Base", False),),
}


def _load_target_metadata(section: str) -> list[MetaData]:
    """Loads only the SQLAlchemy metadata objects owned by `section`."""
    entries = _SECTION_METADATA[section]
    loaded: list[MetaData] = []
    for module_path, attr_name, optional_cog in entries:
        try:
            module = importlib.import_module(name=module_path)
        except ModuleNotFoundError as exc:
            # Skip only when the optional cog's own package is absent, never a missing third-party dep.
            if (
                optional_cog
                and exc.name is not None
                and (module_path == exc.name or module_path.startswith(f"{exc.name}."))
            ):
                continue
            raise
        obj = getattr(module, attr_name)
        loaded.append(obj if isinstance(obj, MetaData) else obj.metadata)
    return loaded


target_metadata = _load_target_metadata(section=context.config.config_ini_section)

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
