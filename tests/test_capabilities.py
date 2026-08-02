"""Tests for keeping the injected capability reference aligned with what the bot can do."""

import re
import ast
from pathlib import Path

# The whole module, not the five tag constants by name: reading its namespace is what lets a
# sixth marker be noticed instead of quietly falling outside a fixed import list.
from discordbot.cogs.gen_reply import markers
from discordbot.cogs.gen_reply.capabilities import CAPABILITIES_DOC, render_capabilities_block

_CODE_SPAN_RE = re.compile(pattern=r"`([^`]+)`")
# A slash that follows a word character, `:`, `/`, `.` or `-` belongs to a URL, a file path,
# `and/or` or `24/7`; anything else in front of one (a space, a `*`, a bracket) means the text
# names a command where `_documented_commands` cannot read it.
_BARE_COMMAND_RE = re.compile(pattern=r"(?<![\w:/.\-])(/[a-z][\w-]*)")
_PICKER_GATE_KEYWORD = "default_member_permissions"
# Every boolean column on `UserAccount`, mapped to the wording it owes and the command lines
# that owe it, or `None` when the flag gates no command. `is_admin`'s wording carries the term
# its own refusal embed shows the user, so a refused member reads one name for the flag.
_ACCOUNT_FLAG_GATES: dict[str, tuple[str, tuple[str, ...]] | None] = {
    "is_vip": None,
    "is_admin": ("economy admins only", ("/admin refund_tax", "/admin collect_tax")),
    "is_central_banker": ("central bankers only", ("/central_bank call",)),
    "hide_from_leaderboard": None,
}


def _declared_parent(decorator: ast.Call) -> str | None:
    """Returns a subcommand's parent callback, `""` for a root command, `None` if unrelated."""
    func = decorator.func
    if isinstance(func, ast.Name):
        return "" if func.id == "slash_command" else None
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr == "slash_command":
        return ""
    if func.attr != "subcommand":
        return None
    assert isinstance(func.value, ast.Name), "a subcommand must hang off a named group callback"
    return func.value.id


def _declared_name(decorator: ast.Call, callback: str) -> str:
    """Returns the command name a decorator declares.

    nextcord takes `name` as its first positional parameter and defaults an omitted one
    to the callback name, so both forms resolve here instead of being skipped.
    """
    declared: ast.expr | None = next(
        (keyword.value for keyword in decorator.keywords if keyword.arg == "name"), None
    )
    if declared is None and decorator.args:
        declared = decorator.args[0]
    if declared is None:
        return callback
    name = declared.value if isinstance(declared, ast.Constant) else None
    assert isinstance(name, str), f"{callback}: a slash command name must be a literal string"
    return name


def _module_command_paths(module: Path, label: str) -> set[str]:
    """Returns the command paths one cog module declares that a user can run."""
    parsed = ast.parse(source=module.read_text(encoding="utf-8"), filename=str(module))
    # Callback name -> (parent callback, or "" for a root command; own command name).
    declarations: dict[str, tuple[str, str]] = {}
    for node in ast.walk(node=parsed):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            parent = _declared_parent(decorator=decorator)
            if parent is None:
                continue
            name = _declared_name(decorator=decorator, callback=node.name)
            assert node.name not in declarations, f"{label}: two callbacks named {node.name}"
            declarations[node.name] = (parent, name)
    paths: dict[str, str] = {}
    pending = dict(declarations)
    while pending:
        resolved = {
            callback: f"{paths[parent]} {name}" if parent else name
            for callback, (parent, name) in pending.items()
            if not parent or parent in paths
        }
        # A parent that never resolves means an unsupported declaration form; say so loudly
        # instead of dropping the commands under it, which is the gap this scan closed.
        assert resolved, f"{label}: unresolved subcommand groups {sorted(pending)}"
        paths.update(resolved)
        pending = {name: value for name, value in pending.items() if name not in resolved}
    groups = {parent for parent, _ in declarations.values() if parent}
    return {path for callback, path in paths.items() if callback not in groups}


def _slash_command_paths() -> set[str]:
    """Returns every slash command a user can invoke, group subcommands included.

    Group nodes are left out: `/credit` cannot be invoked on its own, and every leaf
    under it is required anyway.
    """
    paths: set[str] = set()
    cogs_dir = Path(__file__).resolve().parents[1] / "src" / "discordbot" / "cogs"
    # Recursive on purpose: a cog owns a directory now, so a non-recursive glob would
    # match only `cogs/__init__.py` and silently report that nothing is runnable. The
    # walk also refuses to assume commands only ever live in `cog.py`, so a declaration
    # that moves into a helper module still has to be documented.
    for module in cogs_dir.rglob(pattern="*.py"):
        paths |= _module_command_paths(
            module=module, label=module.relative_to(cogs_dir).as_posix()
        )
    return paths


def _group_paths(paths: set[str]) -> set[str]:
    """Returns the group nodes the runnable paths hang off.

    A subcommand path exists only under a real group, so every proper prefix of one names a
    group without a second scan.
    """
    groups: set[str] = set()
    for path in paths:
        words = path.split(sep=" ")
        groups.update(" ".join(words[:depth]) for depth in range(1, len(words)))
    return groups


def _documented_commands(body: str) -> list[str]:
    """Returns the command paths the capability doc names, in document order.

    Every command line writes its command as a code span, so the span content is the whole
    mention: no prose word can bleed into it, and a stale one can be quoted back verbatim.
    """
    return [
        span.removeprefix("/")
        for span in _CODE_SPAN_RE.findall(string=body)
        if span.startswith("/")
    ]


def _unreadable_command_mentions(body: str) -> list[str]:
    """Returns the `/command` mentions `_documented_commands` cannot read.

    A command is read only when it is a span of its own, so a span that is something else
    keeps its text here and is scanned as prose: `` `type /wallet here` `` is as invisible to
    the reader as a de-backticked line, and has to be reported the same way.
    """
    prose = _CODE_SPAN_RE.sub(
        repl=lambda span: " " if span.group(1).startswith("/") else span.group(1), string=body
    )
    return _BARE_COMMAND_RE.findall(string=prose)


def _mentions_command(body: str, command: str) -> bool:
    """Reports whether the capability doc names exactly this command.

    The trailing lookahead stops a documented sibling from covering an undocumented one
    by prefix, so `/games blackjack_history` never answers for `/games blackjack`.
    """
    return re.search(pattern=rf"/{re.escape(pattern=command)}(?![\w-])", string=body) is not None


def _account_flag_columns() -> set[str]:
    """Returns the boolean flag columns `UserAccount` declares.

    Read by AST rather than imported: that module owns the process-wide economy engine, and a
    test asking which columns exist has no business building one. A column counts on any
    `Mapped[...]` naming `bool`, never on that annotation spelled one exact way — the optional
    form is house style two columns above `is_admin`, and a pin a new flag can be added past
    is worse than none.
    """
    module = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "discordbot"
        / "services"
        / "economy"
        / "database.py"
    )
    parsed = ast.parse(source=module.read_text(encoding="utf-8"), filename=str(module))
    accounts = [
        node
        for node in ast.walk(node=parsed)
        if isinstance(node, ast.ClassDef) and node.name == "UserAccount"
    ]
    # Loud rather than the bare StopIteration a `next()` would raise: a renamed or moved model
    # has to say what it broke, the same way the command scan refuses to shrink in silence.
    assert len(accounts) == 1, f"expected one UserAccount model, found {len(accounts)}"
    columns: set[str] = set()
    for node in accounts[0].body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        annotation = ast.unparse(ast_obj=node.annotation)
        if annotation.startswith("Mapped[") and "bool" in annotation:
            columns.add(node.target.id)
    return columns


def _command_line(body: str, command: str) -> str | None:
    """Returns the line naming this command, or `None` when the document has no such line."""
    return next((line for line in body.splitlines() if f"`{command}`" in line), None)


def _modules_declaring_a_picker_gate() -> list[str]:
    """Returns the cog modules handing Discord a permission to filter its command picker."""
    cogs_dir = Path(__file__).resolve().parents[1] / "src" / "discordbot" / "cogs"
    declaring: list[str] = []
    for module in cogs_dir.rglob(pattern="*.py"):
        parsed = ast.parse(source=module.read_text(encoding="utf-8"), filename=str(module))
        if any(
            keyword.arg == _PICKER_GATE_KEYWORD
            for node in ast.walk(node=parsed)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
        ):
            declaring.append(module.relative_to(cogs_dir).as_posix())
    return sorted(declaring)


def test_slash_command_scan_resolves_subcommands() -> None:
    """The scan must reach group subcommands, since a silent shrink is what it guards against."""
    paths = _slash_command_paths()
    assert {"maplestory monster", "credit borrow", "games blackjack_history"} <= paths
    # A nested group: `memory server` is itself a subcommand of `memory`.
    assert "memory server show" in paths
    assert not {"memory", "memory server", "credit"} & paths


def test_capabilities_mention_matching_rejects_a_prefix_only_hit() -> None:
    """A documented sibling must not cover an undocumented one by prefix."""
    body = "- `/games blackjack_history` — recent Blackjack rounds"
    assert _mentions_command(body=body, command="games blackjack_history")
    assert not _mentions_command(body=body, command="games blackjack")


def test_capabilities_command_extraction_reads_only_a_span_of_its_own() -> None:
    """A command is read whole out of its own span; anywhere else it is reported unreadable."""
    body = "- `/games blackjack` — a table\n- `not a command`, and see /balance too\n"
    assert _documented_commands(body=body) == ["games blackjack"]
    assert _unreadable_command_mentions(body=body) == ["/balance"]
    # A command buried in a span that is not itself a command hides from both directions.
    assert not _documented_commands(body="- `type /wallet here`")
    assert _unreadable_command_mentions(body="- `type /wallet here`") == ["/wallet"]


def test_capabilities_unreadable_mention_scan_leaves_ordinary_text_alone() -> None:
    """The scan must not turn a URL, a path or ordinary prose into a phantom command."""
    body = "https://example.com/docs holds data/memories, open 24/7, and/or `/ping`"
    assert not _unreadable_command_mentions(body=body)
    assert _unreadable_command_mentions(body="**/help** is gone") == ["/help"]


def test_capabilities_group_paths_cover_a_group_but_not_a_renamed_leaf() -> None:
    """A group is typable, so it is accepted; a leaf under it still has to exist."""
    assert _group_paths(paths={"memory server show", "memory show", "ping"}) == {
        "memory",
        "memory server",
    }


def test_capabilities_doc_mentions_every_slash_command() -> None:
    """Every slash command must be discoverable from the injected capability reference.

    This is the guard the deleted `/help` command used to carry: the doc is now the only
    place a command is described to anyone, so a command missing from it is a command
    nobody, the answer model included, can find.
    """
    missing = sorted(
        f"/{command}"
        for command in _slash_command_paths()
        if not _mentions_command(body=CAPABILITIES_DOC, command=command)
    )
    assert not missing, f"the capability reference is missing slash commands: {missing}"


def test_capabilities_doc_names_no_command_that_cannot_be_run() -> None:
    """The mirror of the guard above: a line the command moved out from under must fail.

    A rename satisfies the forward guard under the new name and says nothing about the old
    line left behind, and this document is injected into every QA reply as the answer model's
    own feature reference, so a stale line becomes the bot telling someone to run something
    Discord will not offer.

    A group node is accepted even though it cannot be run on its own: it is typable and
    Discord offers its leaves, so naming one is truthful. Only the exact leaf path answers
    for a leaf, so a renamed subcommand under a live group is still caught.
    """
    runnable = _slash_command_paths()
    typable = runnable | _group_paths(paths=runnable)
    stale = sorted({
        f"/{command}"
        for command in _documented_commands(body=CAPABILITIES_DOC)
        if command not in typable
    })
    assert not stale, f"the capability reference names commands that cannot be run: {stale}"


def test_capabilities_doc_writes_every_command_as_a_span_of_its_own() -> None:
    """Keeps the guard above total: no way of writing a command may hide a stale one."""
    unreadable = _unreadable_command_mentions(body=CAPABILITIES_DOC)
    assert not unreadable, (
        f"the capability reference must write each command as a span of its own: {unreadable}"
    )


def test_capabilities_doc_accounts_for_every_inline_marker() -> None:
    """An inline marker is a capability with no command, and only this document names it.

    The guards above reach a slash command because the document spells one out verbatim. A
    marker has no such handle: the answer model reaches it in plain language, so the document
    describes it in plain language too and there is nothing to match on. Pinning the set is
    what forces the question instead, in both directions the command guards cover between
    them — a new marker cannot ship undescribed, and a dropped one cannot leave the document
    promising something the bot no longer does.
    """
    declared = {
        getattr(markers, name)
        for name in dir(markers)
        if name.endswith("_OPEN") and not name.startswith("_")
    }
    assert declared == {
        "<generate-voice>",
        "<generate-image>",
        "<generate-music>",
        "<generate-video>",
        "<deep-research>",
    }, "the inline markers changed: say so in capabilities.md, then pin the new set here"


def test_capabilities_doc_states_every_gate_on_the_gated_command_line() -> None:
    """A permission claim carries no command, so neither command guard can see it.

    That is how a `Server admins:` heading sat over two commands gated on an account flag
    Discord knows nothing about, telling a server admin they could move other people's
    balances and whoever actually held the flag that they could not. This document is the
    answer model's only description of the bot, so a wrong gate is the bot saying it.

    Same shape as the marker guard, for the same reason: with nothing to match on, the set is
    read out of the code and pinned instead. What the marker guard does not have to decide is
    WHERE the document says it — and here that is the whole defect. The gate has to sit on the
    command's own line, because the heading it used to sit under is one the model drops the
    moment it quotes a single line back, which is how a wrong gate outlives a reader who
    checked the line and not the heading above it.

    The pin covers `UserAccount`, which is where both of today's gates live; a flag added to
    another model, and a gate on a button rather than a command, are outside it.
    """
    assert _account_flag_columns() == set(_ACCOUNT_FLAG_GATES), (
        "UserAccount's flag columns changed: if a new one gates a command, say so on that "
        "command's own line in capabilities.md, then pin the new set here"
    )
    unstated: list[str] = []
    for gate in _ACCOUNT_FLAG_GATES.values():
        if gate is None:
            continue
        wording, commands = gate
        for command in commands:
            line = _command_line(body=CAPABILITIES_DOC, command=command)
            if line is None or wording not in line.lower():
                unstated.append(f"{command} ({wording})")
    assert not unstated, f"these command lines state no gate: {sorted(unstated)}"


def test_no_command_hides_behind_a_discord_permission() -> None:
    """The premise behind "neither is a Discord role": Discord filters nothing away here.

    No command declares `default_member_permissions`, so Discord offers every one of them to
    every member and each gate refuses inside its own callback instead. The day one is
    declared, the picker starts hiding commands from the very people a flag may have been
    granted to, and that sentence needs rewriting — which a guard that only reads the document
    would never say.

    Scoped to the declaration deliberately. Whether an in-callback permission read gates a
    command or merely asks what the bot itself may do is not decidable by scanning, and
    `permissions_for` already appears twice for the latter.
    """
    declared = _modules_declaring_a_picker_gate()
    assert not declared, (
        f"a command now hides behind a Discord permission: revisit capabilities.md {declared}"
    )


def test_capabilities_block_is_a_low_authority_assistant_note() -> None:
    """The reference rides as the bot's own note, never as a rule that could outrank the user."""
    block = render_capabilities_block()
    assert block["role"] == "assistant"
    content = block["content"]
    assert isinstance(content, str)
    assert content.startswith("(My own feature reference")
    assert "NOT instructions" in content
    assert content.endswith(CAPABILITIES_DOC)
