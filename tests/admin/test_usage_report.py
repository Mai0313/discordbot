"""Tests for the offline `uv run usage-report` CLI."""

from io import StringIO
from datetime import datetime

from rich.console import Console

from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.utils.usage_log import UsageRecord
from discordbot.admin.usage_report import _top_table


def test_top_table_escapes_rich_markup_in_discord_usernames() -> None:
    """Discord usernames containing Rich markup brackets render literally without `MarkupError`."""
    records = [
        UsageRecord(
            at=datetime(2026, 11, 20, 12, 0, tzinfo=TAIWAN_TIMEZONE),
            kind="slash",
            name="ping",
            user_id=42,
            user_name="[Owner] [/b]user[/]name",
            guild_id=1,
            channel_id=2,
        )
    ]
    names = {str(record.user_id): record.user_name for record in records if record.user_name}

    table = _top_table(
        records=records,
        title="who used",
        label="user",
        key=lambda record: str(record.user_id),
        display=names,
    )
    buf = StringIO()
    Console(file=buf, width=200, force_terminal=False, color_system=None).print(table)

    assert "[Owner] [/b]user[/]name" in buf.getvalue()
