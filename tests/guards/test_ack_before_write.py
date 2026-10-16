"""Every economy command and loan button acknowledges its interaction before it writes the ledger."""

import ast

from tests.helpers.source_tree import PACKAGE

# Everything `cogs/economy/` imports from the ledger. Split rather than listed: a new import
# fails the sweep below until someone decides which half it belongs in, which is the only thing
# that keeps this from silently missing the next money-moving call.
_LEDGER_MUTATIONS = frozenset({
    "adjust_balance",
    "buy_vip",
    "call_central_bank_loans",
    "call_personal_loans",
    "create_central_bank_loan_request",
    "create_personal_loan_request",
    "repay_central_bank_loans",
    "repay_personal_loans",
    "transfer",
    # Both accrue and persist interest before returning, so they take the same write lock.
    "get_portfolio",
    "list_loan_contracts",
    "accept_loan_proposal",
    "cancel_loan_proposal",
    "reject_expired_loan_proposal",
    "reject_loan_proposal",
    # An upsert of its own row, so it takes the write lock like any other.
    "record_guild_participant",
})


_LEDGER_READS = frozenset({
    "get_account",
    "get_admin",
    "get_balance",
    "get_casino_ledger",
    "get_spendable_balance",
    "get_central_bank_status",
    "get_credit_ceiling",
    "get_vip",
    "top_losers",
    "top_n",
})


# The one writer in these modules with no interaction to acknowledge: a view's expiry fires on
# its own timer. Exempt by name rather than by skipping the file it sits in, so the buttons
# beside it stay swept.
_ACK_SWEEP_EXEMPT = frozenset({("views.py", "on_timeout")})


def _first_unavoidable_ack(node: ast.AST) -> ast.Call | None:
    """The first acknowledgement a write inside this function cannot go around.

    Recurses into `try` and `with` bodies, which always run, and deliberately not into `if`,
    `for` or `while`: a permission guard that acks and returns sits inside an `if`, and an ack
    found there would vouch for a write further down that never passes through it.
    """
    for statement in getattr(node, "body", []):
        if isinstance(statement, ast.Try | ast.With | ast.AsyncWith):
            found = _first_unavoidable_ack(statement)
            if found is not None:
                return found
            continue
        # `AsyncFunctionDef` is not a subclass of `FunctionDef`, and in an async cog a nested
        # def is far more likely to be the async one — an ack inside a closure may never run.
        if isinstance(
            statement, ast.If | ast.For | ast.While | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        for inner in ast.walk(statement):
            if not isinstance(inner, ast.Call):
                continue
            name = ast.unparse(inner.func)
            if (
                name.endswith("response.defer")
                or name.endswith("response.send_message")
                or name.endswith("send_ephemeral_response")
            ):
                return inner
    return None


def _edits_the_original_without_deferring(node: ast.AsyncFunctionDef, ack: ast.Call | None) -> str:
    """Why an ack that is not a `defer` breaks a handler answering by editing its own message.

    `edit_original_message` targets whatever the interaction answered with, so a handler that
    replied with `send_message` first edits THAT — the settlement embed lands in an ephemeral
    notice and the panel keeps its live buttons over a decided loan. It is silent, where the
    `response.edit_message` this replaced would have raised `InteractionResponded`.
    """
    edits = any(
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "edit_response_embed"
        for inner in ast.walk(node)
    )
    if not edits or (ack is not None and ast.unparse(ack.func).endswith("response.defer")):
        return ""
    return f"{node.name} edits the original response without having deferred it"


def _writes_before(node: ast.AsyncFunctionDef, ack: ast.Call | None) -> list[str]:
    """Ledger writes in this function that run before it has acknowledged anything."""
    acked_at = ack.lineno if ack is not None else None
    found: list[str] = []
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call) or not isinstance(inner.func, ast.Name):
            continue
        if inner.func.id in _LEDGER_MUTATIONS and (acked_at is None or inner.lineno < acked_at):
            found.append(f"{node.name} writes at line {inner.lineno} before any ack")
    return found


def test_every_money_command_acknowledges_before_it_mutates() -> None:
    """Discord kills the token three seconds after dispatch; a SQLite writer may wait longer.

    A command that writes first and acks on the result can therefore commit someone's money
    and then fail to say so, reaching them as Discord's "the application did not respond" with
    no statement of what happened.

    Structural because the race needs a contended write lock to observe, and because this is
    the kind of ordering someone reintroduces by moving an `if`.

    Covers the buttons as well as the commands: approving a loan is where the lender is
    actually debited, and its fix is a different one (a component defers into an edit of the
    panel, not into a followup), so it is exactly the pair someone changes one half of.
    """
    economy_dir = PACKAGE / "cogs" / "economy"
    scanned: list[str] = []
    exempted: set[tuple[str, str]] = set()
    offenders: list[str] = []
    for path in sorted(economy_dir.glob("*.py")):
        scanned.append(path.name)
        for node in ast.walk(ast.parse(source=path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            if (path.name, node.name) in _ACK_SWEEP_EXEMPT:
                exempted.add((path.name, node.name))
                continue
            ack = _first_unavoidable_ack(node)
            offenders.extend(_writes_before(node=node, ack=ack))
            offenders.extend(filter(None, [_edits_the_original_without_deferring(node, ack)]))

    assert {"cog.py", "views.py"} <= set(scanned), "the sweep missed a module it must cover"
    # An exemption is the other way a writer leaves this guard while the guard stays green.
    assert exempted == _ACK_SWEEP_EXEMPT, f"exemption never matched anything: {_ACK_SWEEP_EXEMPT}"
    assert not offenders, f"money handlers whose acknowledgement is wrong: {sorted(offenders)}"


def test_the_ack_sweep_accounts_for_every_ledger_name_the_cogs_import() -> None:
    """A new import has to be classified before it can be missed.

    The sweep can only see writes it was told are writes, and a hand-kept list of those goes
    stale the first time someone imports one more.
    """
    economy_dir = PACKAGE / "cogs" / "economy"
    imported: set[str] = set()
    for path in sorted(economy_dir.glob("*.py")):
        for node in ast.walk(ast.parse(source=path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                "economy.database"
            ):
                imported.update(alias.name for alias in node.names)

    unclassified = imported - _LEDGER_MUTATIONS - _LEDGER_READS
    assert imported, "no ledger imports found, so this guard is watching nothing"
    assert not unclassified, (
        f"classify these as a ledger write or a read before the ack sweep can cover them: "
        f"{sorted(unclassified)}"
    )
