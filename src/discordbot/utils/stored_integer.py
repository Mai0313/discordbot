"""Storing an integer as decimal text in SQLite, and keeping SQL treating it as a number.

SQLite's INTEGER is 64-bit and this bot's balances are allowed past that ceiling, so every money
and share column — `services/economy/database.py`, `services/stock/database.py`,
`cogs/games/database.py`, `cogs/games/fishing/database.py` — is a `StoredInteger`: canonical
decimal text on disk, an unbounded Python `int` in the ORM.

The whole cost of that choice is paid on the SQL side, and this module is what pays it. The column
is `Text`, so SQLAlchemy would compile `column + n` to string concatenation and SQLite would answer
`column > n` lexicographically, where "9" beats "10". `StoredIntegerComparator` therefore rewrites
`+`, `-` and the four inequalities into two UDFs, `discordbot_int_add_text` and
`discordbot_int_compare_text`, which parse both operands into Python ints first;
`configure_sqlite_stored_integer_functions` installs them. SQLite scopes a function to the one
connection that created it, which is why `utils/sqlite_config.py` runs that per connection instead
of once at startup, and why an engine holding no such column opts out with
`register_stored_integer=False`. It registers from both an engine's connect and checkout hooks;
why the checkout one is needed too is `ensure_sqlite_hooks`'s own story.

What it deliberately does not carry. The UDF names have to exist on the connection, so a
`StoredInteger` column works on SQLite alone. `==` and `!=` keep the plain text comparison, which
is correct only because every value on both sides goes through `stored_int_to_text` first. A NULL
reads back as 0 and a None binds as "0", so nothing written through this type is ever SQL NULL and
the mapped attributes stay `Mapped[int]`. Numeric ORDER BY is not here either: a two-argument
comparison yields only a sign, so the economy and fishing leaderboards each build their own
sign / length / text ordering on top of `discordbot_int_compare_text`.

It sits in `utils/` because the column type has to be importable by every storage layer that holds
money without any of them importing another, and it carries no domain state that would give it a
home in one of them.
"""

from typing import Any, cast

from sqlalchemy import Text, func
from sqlalchemy.types import TypeDecorator
from sqlalchemy.sql.elements import ColumnElement


def stored_int_to_int(value: object) -> int:
    """Parses a persisted decimal-text integer into a Python int.

    Takes every shape the value can arrive in: an int the driver already decoded, the bytes a
    `text_factory` can hand back, or the stored text. None and blank text both read as 0, which
    is what keeps a NULL column out of the `Mapped[int]` attributes it feeds.

    Args:
        value (object): A column value, a UDF argument, or an operand from a comparator.

    Returns:
        The integer the value denotes, or 0 for None and whitespace-only text.

    Raises:
        TypeError: The value is none of None, int, bytes or str.
        ValueError: The text is not a decimal integer.
    """  # noqa: DOC502 -- ValueError is propagated from `int()`, not raised here
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
    """Renders an integer as the canonical decimal text the column stores.

    Canonical — no padding, no plus sign, no separators — is what lets `==` and the callers'
    ORDER BY expressions work on the stored text itself.

    Args:
        value (int): The integer to store.

    Returns:
        Its decimal text form.
    """
    return str(value)


def sqlite_int_add_text(left: Any, right: Any) -> str:  # noqa: ANN401 -- SQLite UDF inputs can be any scalar type
    """SQLite UDF behind `discordbot_int_add_text`: adds two persisted integers.

    The addition runs here, in Python, where an int has no width; handing the two operands to
    SQLite's own `+` would convert a sum past the 64-bit ceiling to REAL and silently drop its
    low-order digits, which on a money column is the same defect the text storage exists to escape,
    by a different mechanism.

    Args:
        left (Any): Left operand, as SQLite passes it (text, int, bytes or NULL).
        right (Any): Right operand, in the same shapes.

    Returns:
        The sum as canonical decimal text.
    """
    return stored_int_to_text(value=stored_int_to_int(value=left) + stored_int_to_int(value=right))


def sqlite_int_compare_text(left: Any, right: Any) -> int:  # noqa: ANN401 -- SQLite UDF inputs can be any scalar type
    """SQLite UDF behind `discordbot_int_compare_text`: orders two persisted integers.

    Answers with a three-way sign rather than a boolean so one registered function serves all four
    inequalities and the descending ORDER BY expressions the leaderboards build on it.

    Args:
        left (Any): Left operand, as SQLite passes it (text, int, bytes or NULL).
        right (Any): Right operand, in the same shapes.

    Returns:
        -1, 0 or 1 as `left` is below, equal to, or above `right`.
    """
    left_int = stored_int_to_int(value=left)
    right_int = stored_int_to_int(value=right)
    return (left_int > right_int) - (left_int < right_int)


def int_add_text(column: ColumnElement[Any], delta: int) -> ColumnElement[Any]:
    """Builds the SQL expression that adds `delta` to a decimal-text column.

    The expression is a call to a UDF, so a statement carrying it only runs on a connection
    `configure_sqlite_stored_integer_functions` has already been applied to.

    Args:
        column (ColumnElement[Any]): The decimal-text column to add to.
        delta (int): The amount to add; negate it to subtract, there is no subtract UDF.

    Returns:
        The `discordbot_int_add_text` call, usable anywhere a column expression is.
    """
    return cast(
        "ColumnElement[Any]", func.discordbot_int_add_text(column, stored_int_to_text(value=delta))
    )


def int_compare_text(column: ColumnElement[Any], value: int) -> ColumnElement[int]:
    """Builds the SQL expression that compares a decimal-text column with `value`.

    The result is the UDF's sign, not a predicate: the four comparators wrap it in `> 0`, `>= 0`,
    `< 0` and `<= 0`, and nothing outside this module calls this helper. The leaderboards spell
    `func.discordbot_int_compare_text` themselves and order by five terms built on that sign.

    Args:
        column (ColumnElement[Any]): The decimal-text column to compare.
        value (int): The integer to compare it against.

    Returns:
        The `discordbot_int_compare_text` call, yielding -1, 0 or 1.
    """
    return cast(
        "ColumnElement[int]",
        func.discordbot_int_compare_text(column, stored_int_to_text(value=value)),
    )


class StoredIntegerComparator(TypeDecorator.Comparator[int]):
    """Operator overrides that keep a decimal-text column behaving like a number in SQL.

    Installed as `StoredInteger.comparator_factory`, so an ordinary `UserWallet.balance + amount`
    or `UserWallet.balance >= amount` in a statement compiles to the UDF calls instead of the
    `Text` impl's operators, which would concatenate and compare lexicographically. `==` and `!=`
    are deliberately not overridden: both sides are canonical decimal text by then, so the plain
    text comparison already answers correctly.
    """

    def __add__(self, other: object) -> ColumnElement[Any]:
        """Compiles `column + other` into the integer-aware add UDF.

        Args:
            other (object): The addend, parsed by `stored_int_to_int`.

        Returns:
            The addition expression.
        """
        return int_add_text(
            column=cast("ColumnElement[Any]", self.expr), delta=stored_int_to_int(value=other)
        )

    def __sub__(self, other: object) -> ColumnElement[Any]:
        """Compiles `column - other` by adding the negated operand.

        Args:
            other (object): The subtrahend, parsed by `stored_int_to_int`.

        Returns:
            The addition expression carrying the negated operand.
        """
        return int_add_text(
            column=cast("ColumnElement[Any]", self.expr), delta=-stored_int_to_int(value=other)
        )

    def __gt__(self, other: object) -> ColumnElement[bool]:
        """Compiles `column > other` into a sign test on the compare UDF.

        Args:
            other (object): The bound to compare against, parsed by `stored_int_to_int`.

        Returns:
            The predicate expression.
        """
        return (
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=stored_int_to_int(value=other)
            )
            > 0
        )

    def __ge__(self, other: object) -> ColumnElement[bool]:
        """Compiles `column >= other` into a sign test on the compare UDF.

        Args:
            other (object): The bound to compare against, parsed by `stored_int_to_int`.

        Returns:
            The predicate expression.
        """
        return (
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=stored_int_to_int(value=other)
            )
            >= 0
        )

    def __lt__(self, other: object) -> ColumnElement[bool]:
        """Compiles `column < other` into a sign test on the compare UDF.

        Args:
            other (object): The bound to compare against, parsed by `stored_int_to_int`.

        Returns:
            The predicate expression.
        """
        return (
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=stored_int_to_int(value=other)
            )
            < 0
        )

    def __le__(self, other: object) -> ColumnElement[bool]:
        """Compiles `column <= other` into a sign test on the compare UDF.

        Args:
            other (object): The bound to compare against, parsed by `stored_int_to_int`.

        Returns:
            The predicate expression.
        """
        return (
            int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=stored_int_to_int(value=other)
            )
            <= 0
        )


class StoredInteger(TypeDecorator[int]):
    """Column type persisting an unbounded Python integer as decimal text.

    On disk it is an ordinary `Text` column, so the value survives any tool that reads the file;
    the comparator factory is what keeps SQL arithmetic and comparison numeric over it. Both
    directions are total — a None binds as "0" and a NULL reads as 0 — so a column of this type
    never yields None and is mapped as `Mapped[int]`. `cache_ok` is safe because the type carries
    no per-instance state that could change the statement it compiles to.
    """

    impl = Text
    cache_ok = True
    comparator_factory = StoredIntegerComparator

    def process_bind_param(self, value: object | None, dialect: Any) -> str:  # noqa: ANN401 -- SQLAlchemy hook signature
        """Renders the bound value as the canonical decimal text the column stores.

        Canonicalizing here is what makes the untouched `==` correct, since a literal compared
        against the column passes through this first.

        Args:
            value (object | None): The value being bound; None binds as "0".
            dialect (Any): The active dialect, unused — the text form is the same everywhere.

        Returns:
            Canonical decimal text.
        """
        return stored_int_to_text(value=stored_int_to_int(value=value))

    def process_result_value(self, value: object | None, dialect: Any) -> int:  # noqa: ANN401 -- SQLAlchemy hook signature
        """Parses the persisted decimal text back into a Python integer.

        Args:
            value (object | None): The raw column value from the driver; NULL reads as 0.
            dialect (Any): The active dialect, unused — the text form is the same everywhere.

        Returns:
            The integer the column holds.
        """
        return stored_int_to_int(value=value)


def configure_sqlite_stored_integer_functions(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Registers the two integer-aware UDFs on one SQLite connection.

    SQLite scopes a function to the connection that created it, so this runs per connection rather
    than once at startup; re-registering a name simply replaces it. `utils/sqlite_config.py` calls
    it from both an engine's connect and checkout hooks, and `ensure_sqlite_hooks` documents why
    the checkout one is needed too. Any statement built by `int_add_text` / `int_compare_text`,
    or spelling `func.discordbot_int_*` directly as the economy and fishing leaderboards do, fails
    with "no such function" on a connection that missed this.

    Args:
        dbapi_connection (Any): The DBAPI connection to register on, freshly opened on the connect
            hook or handed out of the pool on the checkout one.
    """
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
