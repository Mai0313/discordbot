"""SQLAlchemy helpers for decimal-text integer storage."""

from typing import Any, cast

from sqlalchemy import Text, func
from sqlalchemy.types import TypeDecorator
from sqlalchemy.sql.elements import ColumnElement


def stored_int_to_int(value: object) -> int:
    """Parses a persisted decimal-string integer into a Python int."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        return stored_int_to_int(value=value.decode())
    if isinstance(value, str):
        normalized = value.strip()
        return int(normalized or "0")
    msg = f"Unsupported stored integer type: {type(value)!r}"
    raise TypeError(msg)


def stored_int_to_text(value: int) -> str:
    """Returns canonical decimal text for a persisted integer."""
    return str(value)


def sqlite_int_add_text(left: Any, right: Any) -> str:  # noqa: ANN401 -- SQLite UDF inputs can be any scalar type
    """Adds two persisted integers and returns canonical decimal text."""
    return stored_int_to_text(
        value=stored_int_to_int(value=left) + stored_int_to_int(value=right)
    )


def sqlite_int_compare_text(left: Any, right: Any) -> int:  # noqa: ANN401 -- SQLite UDF inputs can be any scalar type
    """Compares two persisted integers for SQLite predicates."""
    left_int = stored_int_to_int(value=left)
    right_int = stored_int_to_int(value=right)
    return (left_int > right_int) - (left_int < right_int)


def int_add_text(column: ColumnElement[Any], delta: int) -> ColumnElement[Any]:
    """Builds a SQLite expression that adds `delta` to a decimal-text column."""
    return cast(
        "ColumnElement[Any]",
        func.discordbot_int_add_text(column, stored_int_to_text(value=delta)),
    )


def int_compare_text(column: ColumnElement[Any], value: int) -> ColumnElement[int]:
    """Builds a SQLite expression that compares a decimal-text column."""
    return cast(
        "ColumnElement[int]",
        func.discordbot_int_compare_text(column, stored_int_to_text(value=value)),
    )


class StoredIntegerComparator(TypeDecorator.Comparator[int]):
    """Routes SQL arithmetic and comparisons through integer-aware UDFs."""

    def __add__(self, other: object) -> ColumnElement[Any]:
        return int_add_text(
            column=cast("ColumnElement[Any]", self.expr), delta=stored_int_to_int(value=other)
        )

    def __sub__(self, other: object) -> ColumnElement[Any]:
        return int_add_text(
            column=cast("ColumnElement[Any]", self.expr), delta=-stored_int_to_int(value=other)
        )

    def __gt__(self, other: object) -> ColumnElement[bool]:
        return cast(
            "ColumnElement[bool]",
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr),
                value=stored_int_to_int(value=other),
            )
            > 0,
        )

    def __ge__(self, other: object) -> ColumnElement[bool]:
        return cast(
            "ColumnElement[bool]",
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr),
                value=stored_int_to_int(value=other),
            )
            >= 0,
        )

    def __lt__(self, other: object) -> ColumnElement[bool]:
        return cast(
            "ColumnElement[bool]",
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr),
                value=stored_int_to_int(value=other),
            )
            < 0,
        )

    def __le__(self, other: object) -> ColumnElement[bool]:
        return cast(
            "ColumnElement[bool]",
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr),
                value=stored_int_to_int(value=other),
            )
            <= 0,
        )


class StoredInteger(TypeDecorator[int]):
    """Persists Python integers as decimal text in SQLite."""

    impl = Text
    cache_ok = True
    comparator_factory = StoredIntegerComparator

    def process_bind_param(self, value: object | None, dialect: Any) -> str:  # noqa: ANN401 -- SQLAlchemy hook signature
        """Converts a Python integer into canonical decimal text."""
        return stored_int_to_text(value=stored_int_to_int(value=value))

    def process_result_value(self, value: object | None, dialect: Any) -> int:  # noqa: ANN401 -- SQLAlchemy hook signature
        """Converts persisted decimal text into a Python integer."""
        return stored_int_to_int(value=value)


def configure_sqlite_stored_integer_functions(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Registers SQLite UDFs used by `StoredInteger` SQL expressions."""
    dbapi_connection.create_function("discordbot_int_add_text", 2, sqlite_int_add_text)
    dbapi_connection.create_function("discordbot_int_compare_text", 2, sqlite_int_compare_text)


__all__ = [
    "StoredInteger",
    "configure_sqlite_stored_integer_functions",
    "int_add_text",
    "int_compare_text",
    "stored_int_to_int",
    "stored_int_to_text",
]
