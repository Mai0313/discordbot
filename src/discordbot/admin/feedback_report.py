"""Offline report over reply turns and user feedback stored in `data/database/reply.db`.

Reads `reply_turn`, `reply_feedback`, and optionally `data/database/messages.db` in SQLite
read-only mode so it is safe to run alongside a live bot writing through WAL.

Run from the repo root::

    uv run feedback-report
    uv run feedback-report --route QA --down-only
    uv run feedback-report --asker-only --model gemini-4-neon
    uv run feedback-report --inspect --down-only --context 5
    uv run feedback-report --show 1371849201948271616
"""

from pathlib import Path
import sqlite3
import argparse
from datetime import date
from collections import Counter, defaultdict

from rich import box
from pydantic import Field, BaseModel
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

DEFAULT_DB_PATH = Path("data/database/reply.db")
DEFAULT_MESSAGES_DB_PATH = Path("data/database/messages.db")
REASON_TAGS = ("off_topic", "inaccurate", "tone", "verbose", "other")
REASON_LABELS = {
    "off_topic": "答非所問",
    "inaccurate": "資訊錯誤",
    "tone": "語氣怪",
    "verbose": "太囉嗦",
    "other": "其他",
}


class TurnRecord(BaseModel):
    """One `reply_turn` row read from `reply.db`."""

    reply_message_id: int = Field(..., description="Discord message id holding the buttons.")
    source_message_id: int = Field(..., description="Triggering message or interaction id.")
    guild_id: int | None = Field(..., description="Discord guild id, or None in a DM.")
    channel_id: int = Field(..., description="Discord channel id.")
    author_id: int = Field(..., description="Discord user id who asked the prompt.")
    author_name: str | None = Field(
        default=None, description="Write-time username of the prompt author, if recorded."
    )
    author_display_name: str | None = Field(
        default=None, description="Write-time display name or guild nick of the prompt author."
    )
    surface: str = Field(..., description="Surface that ran the turn (`gateway` or `ask`).")
    route: str = Field(..., description="Pipeline route (`QA`, `IMAGE`, or `VIDEO`).")
    model: str = Field(..., description="Actual model recorded for this turn.")
    effort: str = Field(..., description="Reasoning effort recorded for this turn.")
    backend: str = Field(..., description="Backend label (`responses` or `interactions`).")
    question: str = Field(..., description="Snapshot of the user prompt.")
    answer: str = Field(..., description="Snapshot of the finalized reply text.")
    created_at: str = Field(..., description="ISO timestamp written to `reply_turn`.")


class FeedbackRecord(BaseModel):
    """One joined `reply_feedback` + `reply_turn` record."""

    reply_message_id: int = Field(..., description="Discord message id voted on.")
    source_message_id: int = Field(..., description="Triggering message id for log lookup.")
    author_id: int = Field(..., description="Original asker's Discord user id.")
    route: str = Field(..., description="Pipeline route of the rated reply.")
    model: str = Field(..., description="Actual model of the rated reply.")
    effort: str = Field(..., description="Reasoning effort of the rated reply.")
    question: str = Field(..., description="Snapshot of the original question.")
    answer: str = Field(..., description="Snapshot of the original answer.")
    user_id: int = Field(..., description="Discord user id who voted.")
    user_name: str = Field(..., description="Username snapshot at vote time.")
    user_display_name: str | None = Field(
        default=None, description="Display name or guild nick snapshot at vote time."
    )
    vote: str = Field(..., description="Vote value (`up` or `down`).")
    reason_tag: str | None = Field(..., description="Optional downvote reason category.")
    reason_text: str | None = Field(..., description="Optional free-text detail for `other`.")
    updated_at: str = Field(..., description="Last update timestamp on the feedback row.")

    @property
    def is_asker(self) -> bool:
        """Returns whether the voter is the original author of the prompt."""
        return self.user_id == self.author_id


class VoteRecord(BaseModel):
    """One `reply_feedback` vote row shown inside a turn scene card."""

    user_id: int = Field(..., description="Discord user id who voted.")
    user_name: str = Field(..., description="Username snapshot at vote time.")
    user_display_name: str | None = Field(
        default=None, description="Display name or guild nick snapshot at vote time."
    )
    vote: str = Field(..., description="Vote value (`up` or `down`).")
    reason_tag: str | None = Field(..., description="Optional downvote reason category.")
    reason_text: str | None = Field(..., description="Optional free-text detail for `other`.")
    updated_at: str = Field(..., description="Last update timestamp on the feedback row.")


class ContextMessage(BaseModel):
    """One preceding channel message read from `messages.db`."""

    created_at: str = Field(..., description="Timestamp stored in `messages.created_at`.")
    author: str = Field(..., description="Author username stored in `messages.author`.")
    author_id: str = Field(..., description="Author id string stored in `messages.author_id`.")
    content: str = Field(..., description="Message body stored in `messages.content`.")
    attachments: str = Field(..., description="Semicolon-separated attachment URLs, or empty.")
    stickers: str = Field(..., description="Semicolon-separated sticker URLs, or empty.")


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Opens `db_path` in SQLite read-only mode; exits cleanly when the file is missing."""
    if not db_path.is_file():
        raise SystemExit(f"no reply database at {db_path}")
    conn = sqlite3.connect(database=f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _connect_optional_readonly(db_path: Path) -> sqlite3.Connection | None:
    """Opens an optional SQLite database read-only, returning None when absent or unreadable."""
    if not db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(database=f"file:{db_path.resolve()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_console(out: Console | None) -> Console:
    """Returns `out` or a default Console widened to at least 160 columns when non-interactive."""
    if out is not None:
        return out
    default_console = Console()
    if not default_console.is_terminal:
        return Console(width=max(default_console.width, 160))
    return default_console


def _validate_since(value: str | None) -> str | None:
    """Validates a `YYYY-MM-DD` date filter string, raising ValueError on invalid input."""
    if value is None:
        return None
    if len(value) != 10:
        raise ValueError("expected YYYY-MM-DD")
    date.fromisoformat(value)
    return value


def _load_turns(
    conn: sqlite3.Connection, route: str | None, model: str | None, since: str | None
) -> list[TurnRecord]:
    """Returns `reply_turn` rows matching optional `route`, `model`, and `since` filters."""
    rows = conn.execute(
        "SELECT reply_message_id, source_message_id, guild_id, channel_id, author_id, "
        "author_name, author_display_name, surface, route, model, effort, backend, question, "
        "answer, created_at "
        "FROM reply_turn "
        "WHERE (? IS NULL OR route = ?) "
        "AND (? IS NULL OR model = ?) "
        "AND (? IS NULL OR created_at >= ?) "
        "ORDER BY created_at ASC",
        (route, route, model, model, since, since),
    ).fetchall()
    return [TurnRecord(**dict(row)) for row in rows]


def _load_feedback(
    conn: sqlite3.Connection,
    route: str | None,
    model: str | None,
    asker_only: bool,
    since: str | None,
) -> list[FeedbackRecord]:
    """Returns joined feedback rows matching filters, newest first."""
    rows = conn.execute(
        "SELECT f.reply_message_id, t.source_message_id, t.author_id, t.route, t.model, "
        "t.effort, t.question, t.answer, f.user_id, f.user_name, f.user_display_name, f.vote, "
        "f.reason_tag, f.reason_text, f.updated_at "
        "FROM reply_feedback AS f "
        "JOIN reply_turn AS t ON t.reply_message_id = f.reply_message_id "
        "WHERE (? IS NULL OR t.route = ?) "
        "AND (? IS NULL OR t.model = ?) "
        "AND (? = 0 OR f.user_id = t.author_id) "
        "AND (? IS NULL OR t.created_at >= ?) "
        "ORDER BY f.updated_at DESC",
        (route, route, model, model, 1 if asker_only else 0, since, since),
    ).fetchall()
    return [FeedbackRecord(**dict(row)) for row in rows]


def _filter_feedback(
    feedback: list[FeedbackRecord], down_only: bool, reason: str | None
) -> list[FeedbackRecord]:
    """Narrows feedback rows by `--down-only` and `--reason`."""
    return [
        item
        for item in feedback
        if (not down_only or item.vote == "down") and (reason is None or item.reason_tag == reason)
    ]


def _lookup_author_name(
    messages_conn: sqlite3.Connection | None, author_id: int, cache: dict[int, str | None]
) -> str | None:
    """Looks up the latest logged username for `author_id` in `messages.db` via its author index."""
    if author_id in cache:
        return cache[author_id]
    if messages_conn is None:
        cache[author_id] = None
        return None
    try:
        row = messages_conn.execute(
            "SELECT author FROM messages "
            "WHERE author_id = ? AND author IS NOT NULL AND author != '' "
            "ORDER BY created_at DESC LIMIT 1",
            (str(author_id),),
        ).fetchone()
    except sqlite3.Error:
        cache[author_id] = None
        return None
    name = str(row["author"]) if row is not None and row["author"] else None
    cache[author_id] = name
    return name


def _load_preceding_context(
    messages_conn: sqlite3.Connection | None, turn: TurnRecord, limit: int
) -> tuple[list[ContextMessage], str | None]:
    """Reads up to `limit` channel messages before `turn.source_message_id` from `messages.db`."""
    if limit <= 0:
        return [], None
    if messages_conn is None:
        return [], "messages.db not found"
    try:
        anchor = messages_conn.execute(
            "SELECT id, created_at FROM messages WHERE discord_message_id = ? LIMIT 1",
            (str(turn.source_message_id),),
        ).fetchone()
        if anchor is None:
            return [], "source message not found in messages.db"
        anchor_id = anchor["id"]
        anchor_created_at = anchor["created_at"]
        if turn.guild_id is None:
            rows = messages_conn.execute(
                "SELECT author, author_id, content, created_at, attachments, stickers "
                "FROM messages "
                "WHERE source_type = 'dm' AND channel_id = ? "
                "AND created_at <= ? AND (created_at < ? OR id < ?) "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (str(turn.author_id), anchor_created_at, anchor_created_at, anchor_id, limit),
            ).fetchall()
        else:
            rows = messages_conn.execute(
                "SELECT author, author_id, content, created_at, attachments, stickers "
                "FROM messages "
                "WHERE channel_id = ? "
                "AND created_at <= ? AND (created_at < ? OR id < ?) "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (str(turn.channel_id), anchor_created_at, anchor_created_at, anchor_id, limit),
            ).fetchall()
    except sqlite3.Error:
        return [], "messages.db unreadable"
    messages = [
        ContextMessage(
            created_at=str(row["created_at"] or ""),
            author=str(row["author"] or ""),
            author_id=str(row["author_id"] or ""),
            content=str(row["content"] or ""),
            attachments=str(row["attachments"] or ""),
            stickers=str(row["stickers"] or ""),
        )
        for row in reversed(rows)
    ]
    return messages, None


def _snippet(text: str, limit: int = 48) -> str:
    """Flattens newlines and clips `text` to `limit` characters for compact table display."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 1]}…"


def _format_identity(
    user_id: int, username: str | None = None, display_name: str | None = None
) -> str:
    """Formats a Discord user identity as `display_name (@username, id)`, `username (id)`, or `id`."""
    if display_name and username and display_name != username:
        return f"{display_name} (@{username}, {user_id})"
    if display_name:
        return f"{display_name} ({user_id})"
    if username:
        return f"{username} ({user_id})"
    return str(user_id)


def _format_reason(tag: str | None, text: str | None) -> str:
    """Formats a downvote reason tag and optional detail for table or detail rendering."""
    if tag is None:
        return "-"
    label = REASON_LABELS.get(tag, tag)
    return f"{label}: {text}" if text else label


def _format_model_reasons(reasons: Counter[str]) -> str:
    """Formats per-model downvote reason counts as `太囉嗦x3, 其他x1` using ASCII `x`."""
    if not reasons:
        return "-"
    order = {tag: idx for idx, tag in enumerate(REASON_TAGS)}
    ordered = sorted(
        reasons.items(), key=lambda pair: (-pair[1], order.get(pair[0], len(REASON_TAGS)), pair[0])
    )
    return ", ".join(f"{REASON_LABELS.get(tag, tag)}x{count}" for tag, count in ordered)


def _format_media_markers(attachments: str, stickers: str) -> str:
    """Returns compact `[attachment]` / `[sticker]` markers when a logged message carried media."""
    attachment_count = len([part for part in attachments.split(";") if part.strip()])
    sticker_count = len([part for part in stickers.split(";") if part.strip()])
    markers: list[str] = []
    if attachment_count == 1:
        markers.append("[attachment]")
    elif attachment_count > 1:
        markers.append(f"[attachments x{attachment_count}]")
    if sticker_count == 1:
        markers.append("[sticker]")
    elif sticker_count > 1:
        markers.append(f"[stickers x{sticker_count}]")
    return " ".join(markers)


def _render_overview(turns: list[TurnRecord], feedback: list[FeedbackRecord]) -> Table:
    """Returns the top-level summary grid."""
    up = sum(1 for item in feedback if item.vote == "up")
    down = sum(1 for item in feedback if item.vote == "down")
    voters = len({item.user_id for item in feedback})
    period = (
        f"{turns[0].created_at[:10]} - {turns[-1].created_at[:10]}"
        if turns
        else "no turns recorded"
    )
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan")
    grid.add_column()
    grid.add_row("period", Text(text=period))
    grid.add_row("turns", str(len(turns)))
    grid.add_row("votes", f"{len(feedback)}  (👍 {up} / 👎 {down}, {voters} voter(s))")
    return grid


def _render_model_summary(turns: list[TurnRecord], feedback: list[FeedbackRecord]) -> Table:
    """Returns one row per `(route, model)` summarizing turns, votes, approval, and reasons."""
    turn_counts: Counter[tuple[str, str]] = Counter((turn.route, turn.model) for turn in turns)
    up_counts: Counter[tuple[str, str]] = Counter()
    down_counts: Counter[tuple[str, str]] = Counter()
    reason_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for item in feedback:
        key = (item.route, item.model)
        if item.vote == "up":
            up_counts[key] += 1
        else:
            down_counts[key] += 1
            if item.reason_tag is not None:
                reason_counts[key][item.reason_tag] += 1

    keys = sorted(
        set(turn_counts) | set(up_counts) | set(down_counts),
        key=lambda k: (k[0], -(up_counts[k] + down_counts[k]), -turn_counts[k], k[1]),
    )
    table = Table(title="By Model", box=box.SIMPLE_HEAVY, show_edge=False)
    table.add_column("Route", no_wrap=True)
    table.add_column("Model", style="bold", no_wrap=True)
    table.add_column("Turns", justify="right", no_wrap=True)
    table.add_column("👍", justify="right", style="green", no_wrap=True)
    table.add_column("👎", justify="right", style="red", no_wrap=True)
    table.add_column("Appr", justify="right", no_wrap=True)
    table.add_column("Reasons")

    for key in keys:
        up = up_counts[key]
        down = down_counts[key]
        total = up + down
        approval = f"{up / total * 100:.1f}%" if total else "-"
        table.add_row(
            Text(text=key[0]),
            Text(text=key[1]),
            str(turn_counts[key]),
            str(up),
            str(down),
            approval,
            Text(text=_format_model_reasons(reasons=reason_counts[key])),
        )
    return table


def _render_details(
    feedback: list[FeedbackRecord], down_only: bool, reason: str | None, limit: int
) -> Table:
    """Returns recent feedback rows with voter, model, reason, question, and answer snippets."""
    shown = _filter_feedback(feedback=feedback, down_only=down_only, reason=reason)[:limit]
    title = f"Recent Feedback ({len(shown)} shown)"
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_edge=False)
    table.add_column("Time")
    table.add_column("Reply ID", no_wrap=True)
    table.add_column("Model")
    table.add_column("Voter")
    table.add_column("Vote")
    table.add_column("Reason")
    table.add_column("Question")
    table.add_column("Answer")
    for item in shown:
        model_label = f"{item.model} ({item.effort})" if item.effort else item.model
        voter_identity = _format_identity(
            user_id=item.user_id, username=item.user_name, display_name=item.user_display_name
        )
        voter_label = f"{voter_identity} *" if item.is_asker else voter_identity
        vote_label = Text(
            text="👍" if item.vote == "up" else "👎", style="green" if item.vote == "up" else "red"
        )
        table.add_row(
            Text(text=item.updated_at[:16].replace("T", " ")),
            str(item.reply_message_id),
            Text(text=model_label),
            Text(text=voter_label),
            vote_label,
            Text(text=_format_reason(tag=item.reason_tag, text=item.reason_text)),
            Text(text=_snippet(text=item.question)),
            Text(text=_snippet(text=item.answer)),
        )
    return table


def _render_context_panel(messages: list[ContextMessage]) -> Panel:
    """Renders preceding channel messages as a literal-text Rich panel."""
    body = Text()
    for index, msg in enumerate(messages):
        if index > 0:
            body.append("\n")
        sender = f"{msg.author} ({msg.author_id})" if msg.author else msg.author_id or "unknown"
        markers = _format_media_markers(attachments=msg.attachments, stickers=msg.stickers)
        parts = [part for part in (msg.content.strip(), markers) if part]
        text_body = " ".join(parts) if parts else "(empty)"
        body.append(f"[{msg.created_at}] ", style="dim")
        body.append(sender, style="bold")
        body.append(": ")
        body.append(text_body)
    return Panel(renderable=body, title=f"Context (preceding {len(messages)})", border_style="dim")


def _show_turn(  # noqa: PLR0913 -- scene rendering reads both DBs, cache, context cap, and console
    conn: sqlite3.Connection,
    messages_conn: sqlite3.Connection | None,
    reply_message_id: int,
    context_limit: int,
    author_cache: dict[int, str | None],
    out: Console,
) -> None:
    """Prints full metadata, preceding context, Q&A, and every vote on `reply_message_id`."""
    row = conn.execute(
        "SELECT reply_message_id, source_message_id, guild_id, channel_id, author_id, "
        "author_name, author_display_name, surface, route, model, effort, backend, question, "
        "answer, created_at "
        "FROM reply_turn WHERE reply_message_id = ?",
        (reply_message_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"no reply_turn found for reply_message_id={reply_message_id}")
    turn = TurnRecord(**dict(row))
    vote_rows = conn.execute(
        "SELECT user_id, user_name, user_display_name, vote, reason_tag, reason_text, updated_at "
        "FROM reply_feedback WHERE reply_message_id = ? ORDER BY updated_at ASC",
        (reply_message_id,),
    ).fetchall()
    votes = [VoteRecord(**dict(vote_row)) for vote_row in vote_rows]

    author_name = turn.author_name or _lookup_author_name(
        messages_conn=messages_conn, author_id=turn.author_id, cache=author_cache
    )
    asker_identity = _format_identity(
        user_id=turn.author_id, username=author_name, display_name=turn.author_display_name
    )
    location = (
        f"DM / {turn.channel_id}"
        if turn.guild_id is None
        else f"{turn.guild_id} / {turn.channel_id}"
    )

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold cyan")
    meta.add_column()
    meta.add_row("reply_message_id", str(turn.reply_message_id))
    meta.add_row("source_message_id", str(turn.source_message_id))
    meta.add_row("route / surface", Text(text=f"{turn.route} / {turn.surface}"))
    meta.add_row(
        "model / effort / backend",
        Text(text=f"{turn.model} / {turn.effort or '-'} / {turn.backend}"),
    )
    meta.add_row("asker", Text(text=asker_identity))
    meta.add_row("guild / channel", Text(text=location))
    meta.add_row("created_at", Text(text=turn.created_at))
    out.print(meta)

    if context_limit > 0:
        context_messages, unavailable = _load_preceding_context(
            messages_conn=messages_conn, turn=turn, limit=context_limit
        )
        if context_messages:
            out.print(_render_context_panel(messages=context_messages))
        elif unavailable is not None:
            out.print(Text(text=f"Context: unavailable ({unavailable})", style="dim"))

    out.print(
        Panel(
            renderable=Text(text=turn.question or "(empty)"),
            title=Text(text=f"Question ({asker_identity})"),
            border_style="cyan",
        )
    )
    out.print(
        Panel(renderable=Text(text=turn.answer or "(empty)"), title="Answer", border_style="green")
    )

    vote_table = Table(title="Votes", box=box.SIMPLE_HEAVY, show_edge=False)
    vote_table.add_column("Time")
    vote_table.add_column("Voter")
    vote_table.add_column("Vote")
    vote_table.add_column("Reason")
    for vote in votes:
        voter_identity = _format_identity(
            user_id=vote.user_id, username=vote.user_name, display_name=vote.user_display_name
        )
        voter = f"{voter_identity} (asker)" if vote.user_id == turn.author_id else voter_identity
        vote_table.add_row(
            Text(text=vote.updated_at[:19].replace("T", " ")),
            Text(text=voter),
            "👍" if vote.vote == "up" else "👎",
            Text(text=_format_reason(tag=vote.reason_tag, text=vote.reason_text)),
        )
    out.print(vote_table)


def _inspect_turns(  # noqa: PLR0913 -- batch scene inspection carries DBs, filtered feedback, and limits
    conn: sqlite3.Connection,
    messages_conn: sqlite3.Connection | None,
    feedback: list[FeedbackRecord],
    down_only: bool,
    reason: str | None,
    limit: int,
    context_limit: int,
    out: Console,
) -> None:
    """Prints deduplicated full-scene cards for feedback turns matching the active filters."""
    filtered = _filter_feedback(feedback=feedback, down_only=down_only, reason=reason)
    reply_ids: list[int] = []
    seen: set[int] = set()
    for item in filtered:
        if len(reply_ids) >= limit:
            break
        if item.reply_message_id not in seen:
            seen.add(item.reply_message_id)
            reply_ids.append(item.reply_message_id)
    if not reply_ids:
        out.print(Text(text="No matching feedback turns.", style="dim"))
        return

    author_cache: dict[int, str | None] = {}
    for index, reply_id in enumerate(reply_ids, start=1):
        out.rule(title=Text(text=f"[{index}/{len(reply_ids)}] reply {reply_id}"))
        _show_turn(
            conn=conn,
            messages_conn=messages_conn,
            reply_message_id=reply_id,
            context_limit=context_limit,
            author_cache=author_cache,
            out=out,
        )


def run_report(argv: list[str] | None = None, out: Console | None = None) -> None:
    """Parses CLI arguments and renders the feedback report to `out`."""
    parser = argparse.ArgumentParser(
        description="Inspect reply_turn snapshots and user feedback in data/database/reply.db."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to reply.db.")
    parser.add_argument(
        "--messages-db",
        type=Path,
        default=DEFAULT_MESSAGES_DB_PATH,
        help="Path to messages.db for preceding context and author-name fallback.",
    )
    parser.add_argument(
        "--route", choices=("QA", "IMAGE", "VIDEO"), default=None, help="Filter by reply route."
    )
    parser.add_argument("--model", default=None, help="Filter by exact recorded model name.")
    parser.add_argument(
        "--down-only", action="store_true", help="Show only thumbs-down rows in details / inspect."
    )
    parser.add_argument(
        "--asker-only",
        action="store_true",
        help="Include only votes by the original prompt author.",
    )
    parser.add_argument(
        "--reason",
        choices=REASON_TAGS,
        default=None,
        help="Filter detail / inspect rows by downvote reason tag.",
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="Include only turns created on or after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum detail rows or inspected turns (default: 30).",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=5,
        help="Preceding channel messages to read from messages.db in --show / --inspect (default: 5).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print batch full-scene cards for matching feedback turns instead of summary tables.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=None,
        metavar="REPLY_MESSAGE_ID",
        help="Show the full scene, question, answer, and votes for one reply_message_id.",
    )
    args = parser.parse_args(args=argv)
    try:
        since = _validate_since(value=args.since)
    except ValueError:
        parser.error("--since must be in YYYY-MM-DD format")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.context < 0:
        parser.error("--context must be >= 0")

    target = _resolve_console(out=out)
    conn = _connect_readonly(db_path=args.db)
    messages_conn = _connect_optional_readonly(db_path=args.messages_db)
    try:
        if args.show is not None:
            _show_turn(
                conn=conn,
                messages_conn=messages_conn,
                reply_message_id=args.show,
                context_limit=args.context,
                author_cache={},
                out=target,
            )
            return
        try:
            turns = _load_turns(conn=conn, route=args.route, model=args.model, since=since)
            feedback = _load_feedback(
                conn=conn,
                route=args.route,
                model=args.model,
                asker_only=args.asker_only,
                since=since,
            )
        except sqlite3.OperationalError as exc:
            raise SystemExit(
                f"reply feedback tables missing in {args.db} (run migrations first): {exc}"
            ) from exc
        if args.inspect:
            _inspect_turns(
                conn=conn,
                messages_conn=messages_conn,
                feedback=feedback,
                down_only=args.down_only,
                reason=args.reason,
                limit=args.limit,
                context_limit=args.context,
                out=target,
            )
            return
    finally:
        conn.close()
        if messages_conn is not None:
            messages_conn.close()

    target.print(_render_overview(turns=turns, feedback=feedback))
    target.print()
    target.print(_render_model_summary(turns=turns, feedback=feedback))
    target.print()
    target.print(
        _render_details(
            feedback=feedback, down_only=args.down_only, reason=args.reason, limit=args.limit
        )
    )


def main() -> None:
    """Entry point for `uv run feedback-report`."""
    run_report()


if __name__ == "__main__":
    main()
