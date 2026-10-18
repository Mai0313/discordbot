"""SQLAlchemy helpers for decimal-text integer storage."""

from typing import Any, cast

from sqlalchemy import Text, func
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.types import TypeDecorator
from sqlalchemy.sql.elements import ColumnElement

# A mapped column is spelled `Model.column`, which is an `InstrumentedAttribute`
# rather than a `ColumnElement`; SQLAlchemy coerces it, but the annotation has to
# say so or every ORM call site is a type error.
type _StoredIntegerColumn = ColumnElement[Any] | InstrumentedAttribute[Any]


def stored_int_to_int(value: object) -> int:
    """Returns `value` parsed as a Python int from a persisted decimal string or int.

    Args:
        value (object): Persisted decimal string, bytes, int, or None.

    Raises:
        TypeError: `value` is not `int`, `str`, `bytes`, or `None`.
    """
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
    """Returns canonical decimal text for a persisted integer.

    Args:
        value (int): Integer to serialize.
    """
    return str(value)


def sqlite_int_add_text(left: Any, right: Any) -> str:  # noqa: ANN401 -- SQLite UDF inputs can be any scalar type
    """Returns canonical decimal text for `left + right`.

    Args:
        left (Any): Left scalar operand from SQLite.
        right (Any): Right scalar operand from SQLite.
    """
    return stored_int_to_text(value=stored_int_to_int(value=left) + stored_int_to_int(value=right))


def sqlite_int_compare_text(left: Any, right: Any) -> int:  # noqa: ANN401 -- SQLite UDF inputs can be any scalar type
    """Returns `-1`, `0`, or `1` comparing `left` and `right` as integers.

    Args:
        left (Any): Left scalar operand from SQLite.
        right (Any): Right scalar operand from SQLite.
    """
    left_int = stored_int_to_int(value=left)
    right_int = stored_int_to_int(value=right)
    return (left_int > right_int) - (left_int < right_int)


def int_add_text(column: _StoredIntegerColumn, delta: int) -> ColumnElement[Any]:
    """Returns a SQLite expression that adds `delta` to a decimal-text `column`.

    Args:
        column (_StoredIntegerColumn): Column expression holding decimal text.
        delta (int): Integer delta to add.
    """
    return cast(
        "ColumnElement[Any]", func.discordbot_int_add_text(column, stored_int_to_text(value=delta))
    )


def int_compare_text(column: _StoredIntegerColumn, value: int) -> ColumnElement[int]:
    """Returns a SQLite expression that compares a decimal-text `column` against `value`.

    Args:
        column (_StoredIntegerColumn): Column expression holding decimal text.
        value (int): Integer value to compare against.
    """
    return cast(
        "ColumnElement[int]",
        func.discordbot_int_compare_text(column, stored_int_to_text(value=value)),
    )


class _StoredIntegerComparator(TypeDecorator.Comparator[int]):
    """Routes SQL arithmetic and comparisons through integer-aware UDFs."""

    def __add__(self, other: object) -> ColumnElement[Any]:
        """Returns a SQLite expression adding `other` to the column."""
        return int_add_text(
            column=cast("ColumnElement[Any]", self.expr), delta=stored_int_to_int(value=other)
        )

    def __sub__(self, other: object) -> ColumnElement[Any]:
        """Returns a SQLite expression subtracting `other` from the column."""
        return int_add_text(
            column=cast("ColumnElement[Any]", self.expr), delta=-stored_int_to_int(value=other)
        )

    def __gt__(self, other: object) -> ColumnElement[bool]:
        """Returns a SQLite predicate testing whether the column is greater than `other`."""
        return (
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=stored_int_to_int(value=other)
            )
            > 0
        )

    def __ge__(self, other: object) -> ColumnElement[bool]:
        """Returns a SQLite predicate testing whether the column is at least `other`."""
        return (
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=stored_int_to_int(value=other)
            )
            >= 0
        )

    def __lt__(self, other: object) -> ColumnElement[bool]:
        """Returns a SQLite predicate testing whether the column is less than `other`."""
        return (
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=stored_int_to_int(value=other)
            )
            < 0
        )

    def __le__(self, other: object) -> ColumnElement[bool]:
        """Returns a SQLite predicate testing whether the column is at most `other`."""
        return (
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=stored_int_to_int(value=other)
            )
            <= 0
        )


class StoredInteger(TypeDecorator[int]):
    """Persists Python integers as decimal text in SQLite.

    Text rather than `INTEGER` because a balance can outgrow SQLite's 64-bit integer, which
    Python's own int never does; the comparator keeps SQL arithmetic and the `<`, `<=`, `>`,
    `>=` comparisons numeric. `==` and `!=` are not routed through it and compare the text,
    which is exact only because every write stores canonical decimal text; the pinned
    conditional `UPDATE`s depend on that. Ordering is not covered either: SQLAlchemy exposes no
    `ORDER BY` hook here, so a bare `order_by` on one of these columns sorts lexically, and
    every DB-side ranking builds its own decimal-aware order terms instead.
    """

    impl = Text
    cache_ok = True
    comparator_factory = _StoredIntegerComparator

    def process_bind_param(self, value: object | None, dialect: Any) -> str:  # noqa: ANN401 -- SQLAlchemy hook signature
        """Returns canonical decimal text for `value`.

        Args:
            value (object | None): Python integer or decimal text to bind.
            dialect (Any): Active SQLAlchemy dialect.
        """
        return stored_int_to_text(value=stored_int_to_int(value=value))

    def process_result_value(self, value: object | None, dialect: Any) -> int:  # noqa: ANN401 -- SQLAlchemy hook signature
        """Returns `value` converted from persisted decimal text into a Python integer.

        Args:
            value (object | None): Persisted decimal text read from SQLite.
            dialect (Any): Active SQLAlchemy dialect.
        """
        return stored_int_to_int(value=value)


def configure_sqlite_stored_integer_functions(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Registers SQLite UDFs used by `StoredInteger` SQL expressions.

    Args:
        dbapi_connection (Any): Raw SQLite DBAPI connection to register functions on.
    """
    dbapi_connection.create_function("discordbot_int_add_text", 2, sqlite_int_add_text)
    dbapi_connection.create_function("discordbot_int_compare_text", 2, sqlite_int_compare_text)
