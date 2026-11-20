"""Offline report over reply turns and user feedback stored in `data/database/reply.db`.

Reads `reply_turn` and `reply_feedback` in SQLite read-only mode so it is safe to run alongside a
live bot writing through WAL.

Run from the repo root::

    uv run feedback-report
    uv run feedback-report --route QA --down-only
    uv run feedback-report --asker-only --model gemini-4-neon
    uv run feedback-report --show 1371849201948271616
"""

from pathlib import Path
import sqlite3
import argparse
from collections import Counter, defaultdict

from rich import box
from pydantic import Field, BaseModel
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

DEFAULT_DB_PATH = Path("data/database/reply.db")
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
    vote: str = Field(..., description="Vote value (`up` or `down`).")
    reason_tag: str | None = Field(..., description="Optional downvote reason category.")
    reason_text: str | None = Field(..., description="Optional free-text detail for `other`.")
    updated_at: str = Field(..., description="Last update timestamp on the feedback row.")

    @property
    def is_asker(self) -> bool:
        """Returns whether the voter is the original author of the prompt."""
        return self.user_id == self.author_id


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Opens `db_path` in SQLite read-only mode; exits cleanly when the file is missing."""
    if not db_path.is_file():
        raise SystemExit(f"no reply database at {db_path}")
    conn = sqlite3.connect(database=f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _load_turns(
    conn: sqlite3.Connection, route: str | None, model: str | None
) -> list[TurnRecord]:
    """Returns `reply_turn` rows matching optional `route` and `model` filters."""
    rows = conn.execute(
        "SELECT reply_message_id, source_message_id, guild_id, channel_id, author_id, surface, "
        "route, model, effort, backend, question, answer, created_at "
        "FROM reply_turn "
        "WHERE (? IS NULL OR route = ?) AND (? IS NULL OR model = ?) "
        "ORDER BY created_at ASC",
        (route, route, model, model),
    ).fetchall()
    return [TurnRecord(**dict(row)) for row in rows]


def _load_feedback(
    conn: sqlite3.Connection, route: str | None, model: str | None, asker_only: bool
) -> list[FeedbackRecord]:
    """Returns joined feedback rows matching filters, newest first."""
    rows = conn.execute(
        "SELECT f.reply_message_id, t.source_message_id, t.author_id, t.route, t.model, "
        "t.effort, t.question, t.answer, f.user_id, f.user_name, f.vote, f.reason_tag, "
        "f.reason_text, f.updated_at "
        "FROM reply_feedback AS f "
        "JOIN reply_turn AS t ON t.reply_message_id = f.reply_message_id "
        "WHERE (? IS NULL OR t.route = ?) "
        "AND (? IS NULL OR t.model = ?) "
        "AND (? = 0 OR f.user_id = t.author_id) "
        "ORDER BY f.updated_at DESC",
        (route, route, model, model, 1 if asker_only else 0),
    ).fetchall()
    return [FeedbackRecord(**dict(row)) for row in rows]


def _snippet(text: str, limit: int = 48) -> str:
    """Flattens newlines and clips `text` to `limit` characters for compact table display."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 1]}…"


def _format_reason(tag: str | None, text: str | None) -> str:
    """Formats a downvote reason tag and optional detail for table or detail rendering."""
    if tag is None:
        return "-"
    label = REASON_LABELS.get(tag, tag)
    return f"{label}: {text}" if text else label


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
    grid.add_row("period", period)
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
    table.add_column("Route")
    table.add_column("Model", style="bold")
    table.add_column("Turns", justify="right")
    table.add_column("👍", justify="right", style="green")
    table.add_column("👎", justify="right", style="red")
    table.add_column("Approval", justify="right")
    for tag in REASON_TAGS:
        table.add_column(REASON_LABELS[tag], justify="right")

    for key in keys:
        up = up_counts[key]
        down = down_counts[key]
        total = up + down
        approval = f"{up / total * 100:.1f}%" if total else "-"
        reasons = reason_counts[key]
        table.add_row(
            Text(text=key[0]),
            Text(text=key[1]),
            str(turn_counts[key]),
            str(up),
            str(down),
            approval,
            *(str(reasons[tag]) for tag in REASON_TAGS),
        )
    return table


def _render_details(feedback: list[FeedbackRecord], down_only: bool, limit: int) -> Table:
    """Returns recent feedback rows with voter, model, reason, and question snippet."""
    shown = [item for item in feedback if not down_only or item.vote == "down"][:limit]
    title = f"Recent Feedback ({len(shown)} shown)"
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_edge=False)
    table.add_column("Time")
    table.add_column("Reply ID")
    table.add_column("Model")
    table.add_column("Voter")
    table.add_column("Vote")
    table.add_column("Reason")
    table.add_column("Question")
    for item in shown:
        model_label = f"{item.model} ({item.effort})" if item.effort else item.model
        voter_label = f"{item.user_name} *" if item.is_asker else item.user_name
        vote_label = "[green]👍[/green]" if item.vote == "up" else "[red]👎[/red]"
        table.add_row(
            item.updated_at[:16].replace("T", " "),
            str(item.reply_message_id),
            Text(text=model_label),
            Text(text=voter_label),
            vote_label,
            Text(text=_format_reason(tag=item.reason_tag, text=item.reason_text)),
            Text(text=_snippet(text=item.question)),
        )
    return table


def _show_turn(conn: sqlite3.Connection, reply_message_id: int, out: Console) -> None:
    """Prints full metadata, Q&A, and every vote on `reply_message_id`."""
    row = conn.execute(
        "SELECT reply_message_id, source_message_id, guild_id, channel_id, author_id, surface, "
        "route, model, effort, backend, question, answer, created_at "
        "FROM reply_turn WHERE reply_message_id = ?",
        (reply_message_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"no reply_turn found for reply_message_id={reply_message_id}")
    turn = TurnRecord(**dict(row))
    votes = conn.execute(
        "SELECT user_id, user_name, vote, reason_tag, reason_text, updated_at "
        "FROM reply_feedback WHERE reply_message_id = ? ORDER BY updated_at ASC",
        (reply_message_id,),
    ).fetchall()

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold cyan")
    meta.add_column()
    meta.add_row("reply_message_id", str(turn.reply_message_id))
    meta.add_row("source_message_id", str(turn.source_message_id))
    meta.add_row("route / surface", f"{turn.route} / {turn.surface}")
    meta.add_row(
        "model / effort / backend", f"{turn.model} / {turn.effort or '-'} / {turn.backend}"
    )
    meta.add_row("author_id / channel_id", f"{turn.author_id} / {turn.channel_id}")
    meta.add_row("created_at", turn.created_at)
    out.print(meta)
    out.print(
        Panel(
            renderable=Text(text=turn.question or "(empty)"), title="Question", border_style="cyan"
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
    for vote_row in votes:
        voter = (
            f"{vote_row['user_name']} (asker)"
            if vote_row["user_id"] == turn.author_id
            else str(vote_row["user_name"])
        )
        vote_table.add_row(
            str(vote_row["updated_at"])[:19].replace("T", " "),
            Text(text=voter),
            "👍" if vote_row["vote"] == "up" else "👎",
            Text(text=_format_reason(tag=vote_row["reason_tag"], text=vote_row["reason_text"])),
        )
    out.print(vote_table)


def run_report(argv: list[str] | None = None, out: Console | None = None) -> None:
    """Parses CLI arguments and renders the feedback report to `out`."""
    parser = argparse.ArgumentParser(
        description="Inspect reply_turn snapshots and user feedback in data/database/reply.db."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to reply.db.")
    parser.add_argument(
        "--route", choices=("QA", "IMAGE", "VIDEO"), default=None, help="Filter by reply route."
    )
    parser.add_argument("--model", default=None, help="Filter by exact recorded model name.")
    parser.add_argument(
        "--down-only", action="store_true", help="Show only thumbs-down rows in the detail table."
    )
    parser.add_argument(
        "--asker-only",
        action="store_true",
        help="Include only votes by the original prompt author.",
    )
    parser.add_argument(
        "--limit", type=int, default=30, help="Maximum detail rows to show (default: 30)."
    )
    parser.add_argument(
        "--show",
        type=int,
        default=None,
        metavar="REPLY_MESSAGE_ID",
        help="Show the full question, answer, and votes for one reply_message_id.",
    )
    args = parser.parse_args(args=argv)
    target = out or Console()

    conn = _connect_readonly(db_path=args.db)
    try:
        if args.show is not None:
            _show_turn(conn=conn, reply_message_id=args.show, out=target)
            return
        try:
            turns = _load_turns(conn=conn, route=args.route, model=args.model)
            feedback = _load_feedback(
                conn=conn, route=args.route, model=args.model, asker_only=args.asker_only
            )
        except sqlite3.OperationalError as exc:
            raise SystemExit(
                f"reply feedback tables missing in {args.db} (run migrations first): {exc}"
            ) from exc
    finally:
        conn.close()

    target.print(_render_overview(turns=turns, feedback=feedback))
    target.print()
    target.print(_render_model_summary(turns=turns, feedback=feedback))
    target.print()
    target.print(_render_details(feedback=feedback, down_only=args.down_only, limit=args.limit))


def main() -> None:
    """Entry point for `uv run feedback-report`."""
    run_report()


if __name__ == "__main__":
    main()
