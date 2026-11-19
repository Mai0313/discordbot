"""Offline stocktake of the append-only usage records under `data/usage`.

Reads the monthly JSONL files written per slash invocation and per AI reply. Safe against
a live bot: nothing writes back, and a partially written tail line is counted as
unreadable rather than silently dropped.

Every number is a count of invocations. The records carry no success field, no arguments
and no content, so a share below is a share of uses and never a success rate. A command
that exists but was never run leaves no record at all, so the tables name the features
that were used, not the ones that were not.

A user is grouped by id and labelled with the username the newest of their records carries.
That username is a write-time snapshot: someone who renamed leaves both names in the file,
so grouping by it would split them in two, and a record written before the field existed
carries none at all and falls back to its id. A guild is named by id throughout, since
nothing records its name.

Run from the repo root::

    uv run usage-report              # every month on disk
    uv run usage-report 2026-08      # one month
"""

from typing import TYPE_CHECKING
from pathlib import Path
import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

from rich import box
from rich.table import Table
from rich.console import Console

from discordbot.utils.usage_log import UsageRecord, UsageLogConfig

if TYPE_CHECKING:
    from datetime import datetime

    from discordbot.utils.usage_log import UsageKind

console = Console()


def _month_files(directory: Path, month: str | None) -> list[Path]:
    """Returns matching monthly JSONL files sorted oldest first; exits if none exist."""
    if not directory.is_dir():
        raise SystemExit(f"no usage records at {directory}")
    files = sorted(directory.glob(pattern="*.jsonl"))
    if month is not None:
        files = [path for path in files if path.stem == month]
    if not files:
        raise SystemExit(f"no usage records for {month or 'any month'} in {directory}")
    return files


def _read_records(paths: Sequence[Path]) -> tuple[list[UsageRecord], int]:
    """Returns time-sorted records parsed from `paths` and the count of unreadable lines."""
    records: list[UsageRecord] = []
    unreadable = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(UsageRecord.model_validate_json(json_data=line))
            except ValueError:
                unreadable += 1
    records.sort(key=lambda record: record.at)
    return records, unreadable


def _share(value: int, total: int) -> str:
    """Renders `value` as a percentage of `total`."""
    return f"{value / total * 100:.1f}%" if total else "-"


def _bar(value: int, peak: int) -> str:
    """Renders `value` as a block bar scaled against the largest value in its column."""
    if peak <= 0:
        return ""
    # Keeps the widest table within 80 columns so rich never crops the bar.
    bar_width = 16
    filled = round(value / peak * bar_width)
    # Render a thin sliver for non-zero counts that round to zero blocks.
    return "█" * filled if filled else "▏"


def _overview(records: list[UsageRecord], paths: Sequence[Path], unreadable: int) -> Table:
    """Returns the header grid: what was read, over what window, by how many people."""
    kinds = Counter(record.kind for record in records)
    active_days = {f"{record.at:%Y-%m-%d}" for record in records}
    span_days = (records[-1].at.date() - records[0].at.date()).days + 1
    guilds = {record.guild_id for record in records if record.guild_id is not None}

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("files", ", ".join(path.stem for path in paths))
    table.add_row(
        "period",
        f"{records[0].at:%Y-%m-%d} - {records[-1].at:%Y-%m-%d}"
        f"  ({span_days} day(s), {len(active_days)} with activity)",
    )
    table.add_row("records", f"{len(records)}  ({len(records) / span_days:.1f}/day)")
    table.add_row(
        "slash", f"{kinds['slash']}  ({_share(value=kinds['slash'], total=len(records))})"
    )
    table.add_row(
        "replies", f"{kinds['reply']}  ({_share(value=kinds['reply'], total=len(records))})"
    )
    table.add_row("users", str(len({record.user_id for record in records})))
    dms = sum(1 for record in records if record.guild_id is None)
    table.add_row("guilds", f"{len(guilds)}" + (f"  ({dms} uses came from DMs)" if dms else ""))
    if unreadable:
        table.add_row("unreadable", f"[yellow]{unreadable} line(s) skipped[/yellow]")
    return table


def _feature_table(records: list[UsageRecord], kind: "UsageKind", title: str, label: str) -> Table:
    """Returns one row per distinct feature of `kind`, most used first."""
    rows = [record for record in records if record.kind == kind]
    counts = Counter(record.name for record in rows)
    users: defaultdict[str, set[int]] = defaultdict(set)
    last_used: dict[str, datetime] = {}
    for record in rows:
        users[record.name].add(record.user_id)
        # The records are in time order, so the final write per name is its latest use.
        last_used[record.name] = record.at

    table = Table(
        title=f"{title} - {len(rows)} uses across {len(counts)} {label}s",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
    )
    table.add_column(label, no_wrap=True)
    table.add_column("uses", justify="right")
    table.add_column("share", justify="right")
    table.add_column("", style="cyan")
    table.add_column("users", justify="right")
    table.add_column("last used")
    peak = max(counts.values(), default=0)
    for name, count in counts.most_common():
        table.add_row(
            name,
            str(count),
            _share(value=count, total=len(rows)),
            _bar(value=count, peak=peak),
            str(len(users[name])),
            f"{last_used[name]:%m-%d %H:%M}",
        )
    return table


def _daily_table(records: list[UsageRecord]) -> Table:
    """Returns one daily summary row per calendar day, oldest first."""
    per_day: defaultdict[str, Counter[str]] = defaultdict(Counter)
    users: defaultdict[str, set[int]] = defaultdict(set)
    for record in records:
        day = f"{record.at:%Y-%m-%d}"
        per_day[day][record.kind] += 1
        users[day].add(record.user_id)

    table = Table(title="daily", title_justify="left", title_style="bold", box=box.SIMPLE_HEAD)
    table.add_column("date")
    table.add_column("slash", justify="right")
    table.add_column("replies", justify="right")
    table.add_column("total", justify="right")
    table.add_column("", style="cyan")
    table.add_column("users", justify="right")
    peak = max((sum(counts.values()) for counts in per_day.values()), default=0)
    for day in sorted(per_day):
        total = sum(per_day[day].values())
        table.add_row(
            day,
            str(per_day[day]["slash"]),
            str(per_day[day]["reply"]),
            str(total),
            _bar(value=total, peak=peak),
            str(len(users[day])),
        )
    return table


def _top_table(
    records: list[UsageRecord],
    title: str,
    label: str,
    key: Callable[[UsageRecord], str],
    display: dict[str, str] | None = None,
) -> Table:
    """Returns the busiest `key` buckets (top 10 plus a tail row), relabelled via `display`."""
    counts = Counter(key(record) for record in records)
    features: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        features[key(record)][record.name] += 1

    table = Table(
        title=f"{title} - {len(counts)} total",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
    )
    table.add_column(label, no_wrap=True)
    table.add_column("uses", justify="right")
    table.add_column("share", justify="right")
    table.add_column("", style="cyan")
    table.add_column("most used")
    top_rows = 10
    top = counts.most_common(top_rows)
    peak = max((count for _, count in top), default=0)
    for name, count in top:
        table.add_row(
            (display or {}).get(name, name),
            str(count),
            _share(value=count, total=len(records)),
            _bar(value=count, peak=peak),
            features[name].most_common(1)[0][0],
        )
    tail = len(records) - sum(count for _, count in top)
    if tail:
        table.add_row(
            f"[dim]{len(counts) - len(top)} more[/dim]",
            f"[dim]{tail}[/dim]",
            f"[dim]{_share(value=tail, total=len(records))}[/dim]",
            "",
            "",
        )
    return table


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the usage-report CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "month", nargs="?", default=None, help="YYYY-MM; omit to read every month on disk."
    )
    parser.add_argument(
        "--dir",
        default=UsageLogConfig().directory,
        help="Directory holding the monthly record files (defaults to USAGE_LOG_DIR).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Reads the requested months and prints the stocktake."""
    args = _parse_args()
    paths = _month_files(directory=Path(args.dir), month=args.month)
    records, unreadable = _read_records(paths=paths)
    if not records:
        raise SystemExit(f"no readable records in {', '.join(str(path) for path in paths)}")
    console.print(_overview(records=records, paths=paths, unreadable=unreadable))
    console.print()
    console.print(
        _feature_table(records=records, kind="slash", title="slash commands", label="command")
    )
    console.print(_feature_table(records=records, kind="reply", title="ai replies", label="route"))
    console.print(_daily_table(records=records))
    console.print(
        _top_table(
            records=records,
            # Avoids "guilds" title because DM traffic occupies one row alongside guilds.
            title="where used",
            label="guild id",
            key=lambda record: "DM" if record.guild_id is None else str(record.guild_id),
        )
    )
    # The records are in time order, so the last name written per id is the current one.
    names = {str(record.user_id): record.user_name for record in records if record.user_name}
    console.print(
        _top_table(
            records=records,
            title="who used",
            label="user",
            key=lambda record: str(record.user_id),
            display=names,
        )
    )


if __name__ == "__main__":
    main()
