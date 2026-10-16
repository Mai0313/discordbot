"""Tests for the `/memory` views: pagination and the clear confirmation."""

from typing import Any, cast
from pathlib import Path
from collections.abc import Callable

import pytest
from nextcord import Embed
from nextcord.ui import Button

from discordbot.services.memory import database as memory_db
from discordbot.cogs.memory.views import (
    MemoryPagesView,
    MemoryClearConfirmView,
    paginate_on_lines,
    memory_footer_text,
)
from discordbot.services.memory.store import (
    read_tone,
    write_fact,
    cleared_since,
    count_raw_entries,
)

from tests.helpers.memory import assert_cleared_row
from tests.helpers.casting import as_interaction
from tests.helpers.user_memory import (
    USER_SCOPE,
    _stage_row,
    _memory_text,
    _stored_fact,
    _populate_every_tier,
    _fail_the_file_delete,
)
from tests.helpers.discord_mocks import FakeInteraction


def test_paginate_on_lines_single_page_passthrough() -> None:
    assert paginate_on_lines(text="a\nb", limit=10) == ["a\nb"]
    assert paginate_on_lines(text="", limit=10) == [""]


def test_paginate_on_lines_splits_on_line_boundaries() -> None:
    lines = [f"* 第 {index} 行的記憶內容" for index in range(50)]
    text = "\n".join(lines)
    pages = paginate_on_lines(text=text, limit=100)
    assert len(pages) > 1
    for page in pages:
        assert len(page) <= 100
    # Joining the pages back reproduces the text exactly: no line was torn.
    assert "\n".join(pages) == text


def test_paginate_on_lines_hard_splits_oversized_line() -> None:
    pages = paginate_on_lines(text="x" * 250, limit=100)
    assert [len(page) for page in pages] == [100, 100, 50]


def test_paginate_on_lines_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        paginate_on_lines(text="x", limit=0)


async def test_memory_pages_view_navigates_and_disables_bounds() -> None:
    view = MemoryPagesView(
        pages=["第一頁", "第二頁", "第三頁"],
        footer_text=memory_footer_text(pending_count=1),
        title="🧠 我對你的記憶",
    )
    prev_button = cast("Button[Any]", view.previous_page)
    next_button = cast("Button[Any]", view.next_page)
    assert prev_button.disabled is True
    assert next_button.disabled is False

    interaction = FakeInteraction()
    await next_button.callback(as_interaction(fake=interaction))
    assert view.page_index == 1
    embed = interaction.response.edited[-1]["embed"]
    assert isinstance(embed, Embed)
    assert embed.description == "第二頁"
    assert "第 2/3 頁" in (embed.footer.text or "")
    assert "1 筆" in (embed.footer.text or "")
    assert prev_button.disabled is False

    await next_button.callback(as_interaction(fake=interaction))
    assert view.page_index == 2
    assert next_button.disabled is True

    await prev_button.callback(as_interaction(fake=interaction))
    assert view.page_index == 1
    edited_embed = interaction.response.edited[-1]["embed"]
    assert isinstance(edited_embed, Embed)
    assert edited_embed.description == "第二頁"


@pytest.mark.parametrize(
    "build_view",
    [
        pytest.param(
            lambda: MemoryPagesView(
                pages=["第一頁", "第二頁"],
                footer_text=memory_footer_text(pending_count=0),
                title="🧠 我對你的記憶",
            ),
            id="pages",
        ),
        pytest.param(lambda: MemoryClearConfirmView(scope=USER_SCOPE), id="clear-confirm"),
    ],
)
async def test_an_idle_memory_view_disables_its_buttons(
    build_view: Callable[[], MemoryPagesView | MemoryClearConfirmView],
) -> None:
    """An idle view goes inert, so an abandoned clear prompt is no live one-click wipe."""
    view = build_view()
    # Without a bound origin the timeout is a silent no-op.
    await view.on_timeout()

    origin = FakeInteraction()
    view.bind_origin(interaction=as_interaction(fake=origin))
    await view.on_timeout()
    assert origin.edits[-1]["view"] is view
    assert all(child.disabled for child in view.children if isinstance(child, Button))


@pytest.mark.parametrize("last_button", ["previous_page", "next_page"])
async def test_a_paged_memory_view_times_out_through_its_newest_press(last_button: str) -> None:
    """Every press restarts the timer, so the timeout edit rides the newest press's token.

    Paging long enough pushes the timeout past the command's own token, which can no longer
    disable the buttons.
    """
    view = MemoryPagesView(
        pages=["第一頁", "第二頁", "第三頁"],
        footer_text=memory_footer_text(pending_count=0),
        title="🧠 我對你的記憶",
    )
    command = FakeInteraction()
    view.bind_origin(interaction=as_interaction(fake=command))
    earlier_press = FakeInteraction()
    newest_press = FakeInteraction()

    await cast("Button[Any]", view.next_page).callback(as_interaction(fake=earlier_press))
    await cast("Button[Any]", getattr(view, last_button)).callback(
        as_interaction(fake=newest_press)
    )
    await view.on_timeout()

    assert (command.edits, earlier_press.edits) == ([], [])
    assert newest_press.edits[-1]["view"] is view
    assert all(child.disabled for child in view.children if isinstance(child, Button))


def _confirm_button(view: MemoryClearConfirmView) -> "Button[Any]":
    """Returns the view's confirm button, which `View.__init__` bound over the callback."""
    return cast("Button[Any]", view.confirm_clear)


async def test_memory_clear_confirm_button_erases_memory(memory_isolated_dir: Path) -> None:
    _populate_every_tier()
    await _stage_row()
    view = MemoryClearConfirmView(scope=USER_SCOPE)
    interaction = FakeInteraction()

    await _confirm_button(view=view).callback(as_interaction(fake=interaction))

    assert _memory_text() == ""
    await assert_cleared_row(scope=USER_SCOPE)
    # Acked before the work so a slow clear cannot miss Discord's response window.
    assert interaction.response.deferred is True
    payload = interaction.edits[-1]
    assert payload["view"] is None
    embed = payload["embed"]
    assert isinstance(embed, Embed)
    assert "都清掉了" in (embed.description or "")
    assert view.is_finished() is True


async def test_memory_clear_confirm_button_reports_an_empty_scope(
    memory_isolated_dir: Path,
) -> None:
    view = MemoryClearConfirmView(scope=USER_SCOPE)
    interaction = FakeInteraction()

    await _confirm_button(view=view).callback(as_interaction(fake=interaction))

    embed = interaction.edits[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "沒有東西需要清除" in (embed.description or "")


async def test_memory_clear_cancel_button_keeps_memory(memory_isolated_dir: Path) -> None:
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="舊記憶"))
    view = MemoryClearConfirmView(scope=USER_SCOPE)
    interaction = FakeInteraction()

    await cast("Button[Any]", view.cancel_clear).callback(as_interaction(fake=interaction))

    assert "舊記憶" in _memory_text()
    # A cancel must not even stamp the scope, or it would abort in-flight turns.
    assert cleared_since(scope=USER_SCOPE, started_at=0.0) is False
    edited = interaction.response.edited[-1]
    embed = edited["embed"]
    assert isinstance(embed, Embed)
    assert "已取消" in (embed.description or "")
    assert edited["view"] is None


async def test_memory_clear_second_click_does_not_overwrite_the_outcome(
    memory_isolated_dir: Path,
) -> None:
    """A double click must not re-run the clear and report "nothing to clear"."""
    _populate_every_tier()
    view = MemoryClearConfirmView(scope=USER_SCOPE)
    first = FakeInteraction()
    await _confirm_button(view=view).callback(as_interaction(fake=first))

    second = FakeInteraction()
    await _confirm_button(view=view).callback(as_interaction(fake=second))

    # The second press is acked and dropped, leaving the first press's message.
    assert second.response.deferred is True
    assert second.edits == []
    embed = first.edits[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "都清掉了" in (embed.description or "")


async def test_memory_clear_failure_keeps_memory_and_says_so(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reply.db failure must not half-clear: the tombstone write runs before any unlink."""
    _populate_every_tier()

    async def exploding_clear_job(scope: str, flavor: str, token: int) -> bool:
        raise RuntimeError("reply.db unavailable")

    monkeypatch.setattr(memory_db, "clear_job", exploding_clear_job)
    view = MemoryClearConfirmView(scope=USER_SCOPE)
    interaction = FakeInteraction()

    await _confirm_button(view=view).callback(as_interaction(fake=interaction))

    assert "新記憶" in _memory_text()
    assert "本群記憶" in _memory_text()
    assert read_tone(scope=USER_SCOPE) == "## 語氣偏好\n* 輕鬆"
    assert count_raw_entries(scope=USER_SCOPE) == 1
    embed = interaction.edits[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "沒有完成" in (embed.description or "")
    # The stamp is deliberately NOT rolled back, which is why the message must
    # not claim nothing happened: turns in flight for this scope still abort.
    assert cleared_since(scope=USER_SCOPE, started_at=0.0) is True


async def test_memory_clear_reports_a_file_failure_without_claiming_success(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file half walks the tiers one at a time, so it can stop part way.

    The message must not claim the memory survived intact (the reply.db row is
    already a scrubbed tombstone by then) nor that the clear succeeded; a retry
    finishes it.
    """
    _populate_every_tier()
    await _stage_row()
    _fail_the_file_delete(monkeypatch=monkeypatch)
    view = MemoryClearConfirmView(scope=USER_SCOPE)
    interaction = FakeInteraction()

    await _confirm_button(view=view).callback(as_interaction(fake=interaction))

    embed = interaction.edits[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "沒有完成" in (embed.description or "")
    # The durable marker goes before the files, so recovery can finish later.
    await assert_cleared_row(scope=USER_SCOPE)
