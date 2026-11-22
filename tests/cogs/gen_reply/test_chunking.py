"""Unit tests for code-fence-aware Discord reply chunking."""

import pytest

from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.cogs.gen_reply.chunking import (
    TRUNCATED_NOTICE,
    _cut_cleanly,
    split_reply_for_discord,
)

_SPLIT_FOOTER = "\n\n-# model · ⬆ 1 ⬇ 2 · $0.00000001"
_FAMILY_EMOJI = "\U0001f468\u200d\U0001f469\u200d\U0001f467"


def _code_lines(count: int) -> str:
    """Python source of `count` distinct lines, each with spaces a word break could take."""
    return "\n".join(
        f"value_{index:03d} = compute(index={index}, scale=2.5)" for index in range(count)
    )


def test_split_reply_for_discord_rejects_oversized_footer() -> None:
    """A footer that alone reaches Discord's message limit raises `ValueError`."""
    with pytest.raises(ValueError, match="Usage footer is too long"):
        split_reply_for_discord(content="hi", footer="x" * DISCORD_MESSAGE_LIMIT)


def test_split_reply_for_discord_keeps_single_message_answer_intact_when_only_footer_overflows() -> (
    None
):
    """An answer that fits one message is not split just for its footer, which follows alone."""
    content = "字" * (DISCORD_MESSAGE_LIMIT - 5)

    parent, chunks = split_reply_for_discord(content=content, footer=_SPLIT_FOOTER)

    assert parent == content
    assert chunks == [_SPLIT_FOOTER]


@pytest.mark.parametrize("max_messages", [1, 2])
def test_only_an_answer_past_the_budget_is_cut(max_messages: int) -> None:
    """The notice's room is held back only once something must go, so a fitting answer is whole."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    fits = "字" * (max_messages * DISCORD_MESSAGE_LIMIT - len(footer))

    parent, chunks = split_reply_for_discord(
        content=fits, footer=footer, max_messages=max_messages
    )
    over_parent, over_chunks = split_reply_for_discord(
        content=f"{fits}字", footer=footer, max_messages=max_messages
    )

    fit = [parent, *chunks]
    assert "".join(fit) == f"{fits}{footer}"
    assert len(fit) <= max_messages
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in fit)
    over = [over_parent, *over_chunks]
    assert len(over) == max_messages
    assert TRUNCATED_NOTICE.strip() in over[-1]
    assert over[-1].endswith(footer)
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in over)


def test_an_answer_that_fits_only_packed_full_is_not_cut_for_its_breaks() -> None:
    """Clean cuts leave room unused, which a capped surface cannot spare from an answer that fits."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    content = "\n\n".join(["字" * 1400, "字" * 1400, ""])
    content += "字" * (2 * DISCORD_MESSAGE_LIMIT - len(footer) - len(content))

    parent, chunks = split_reply_for_discord(content=content, footer=footer, max_messages=2)
    _, uncapped_chunks = split_reply_for_discord(content=content, footer=footer)

    assert len(uncapped_chunks) > 1, "clean cuts fit the cap anyway, so this test proves nothing"
    assert len(chunks) == 1
    assert f"{parent}{chunks[0]}" == f"{content}{footer}"
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in [parent, *chunks])


def test_an_answer_past_the_budget_is_still_cut_where_a_reader_would() -> None:
    """Running out of messages changes where the answer ends, not how its messages are cut."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    mention = "<@123456789012345678>"
    content = "b" * 1990 + f" {mention} thanks\n\n" + "\n\n".join(["word " * 100] * 25)

    parent, chunks = split_reply_for_discord(content=content, footer=footer, max_messages=6)

    messages = [parent, *chunks]
    assert len(messages) == 6
    assert parent == "b" * 1990
    assert any(mention in message for message in messages)
    assert messages[-1].endswith(f"{TRUNCATED_NOTICE}{footer}")
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in messages)


def test_the_last_message_a_budget_allows_shows_what_it_can_of_a_code_block() -> None:
    """With no message after it, ending before the block would leave the room empty for nothing."""
    content = "prose " * 333 + f"\nHere it is:\n```python\n{_code_lines(count=150)}\n```"

    _parent, chunks = split_reply_for_discord(
        content=content, footer=_SPLIT_FOOTER, max_messages=2
    )

    assert len(chunks) == 1
    assert chunks[0].count("```") == 2
    assert "value_000 = compute(index=0, scale=2.5)" in chunks[0]
    assert len(chunks[0]) > DISCORD_MESSAGE_LIMIT // 2
    assert chunks[0].endswith(f"{TRUNCATED_NOTICE}{_SPLIT_FOOTER}")


def test_an_uncapped_surface_splits_the_whole_answer() -> None:
    """The gateway path has no budget, so nothing about the existing split changes."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    content = "字" * 9000

    parent, chunks = split_reply_for_discord(content=content, footer=footer)

    assert TRUNCATED_NOTICE.strip() not in "".join([parent, *chunks])
    assert "".join([parent, *chunks]) == f"{content}{footer}"


@pytest.mark.parametrize(
    ("content", "whole"),
    [
        pytest.param(
            "x" * 1990 + "\n```python\nprint('hello world')\n```\nafter",
            "```python\nprint('hello world')\n```",
            id="code-block",
        ),
        pytest.param("a" * 1995 + " hello wonderful world", "hello", id="word"),
        pytest.param(
            "b" * 1990 + " <@123456789012345678> thanks", "<@123456789012345678>", id="mention"
        ),
        pytest.param(
            "字" * 1990 + "<@123456789012345678>謝謝",
            "<@123456789012345678>",
            id="mention-no-space",
        ),
        pytest.param("\n\n".join(letter * 1500 for letter in "abc"), "b" * 1500, id="paragraph"),
        pytest.param("c" * 1999 + _FAMILY_EMOJI, _FAMILY_EMOJI, id="zwj-emoji"),
        pytest.param(
            "Type ``` to open a block. " + "a" * 1970 + " hello wonderful world",
            "hello",
            id="unpaired-fence-is-text",
        ),
        pytest.param(
            f"```python\n{_code_lines(count=48)}\n```",
            f"```python\n{_code_lines(count=48)}\n```",
            id="code-block-filling-the-message",
        ),
        pytest.param(
            "word " * 50 + f"\n```python\n{_code_lines(count=44)}\n```",
            f"```python\n{_code_lines(count=44)}\n```",
            id="code-block-after-short-intro",
        ),
    ],
)
def test_a_long_answer_is_split_where_a_reader_would(content: str, whole: str) -> None:
    """Whatever straddles the 2000th character lands whole in one message, and nothing is lost."""
    parent, chunks = split_reply_for_discord(content=content, footer=_SPLIT_FOOTER)

    messages = [parent, *chunks]
    assert len(messages) > 1, "the answer did not split, so this test proves nothing"
    assert any(whole in message for message in messages)
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in messages)
    assert "".join(messages) == f"{content}{_SPLIT_FOOTER}"


def test_the_footer_still_follows_answer_text_after_a_cut() -> None:
    """A footer alone loses its blank line to Discord's trim, and history then fails to strip it."""
    content = "para " * 300 + "\n\n" + "z" * 1980

    parent, chunks = split_reply_for_discord(content=content, footer=_SPLIT_FOOTER)

    assert parent == "para " * 300
    assert USAGE_FOOTER_RE.search(chunks[-1].strip()) is not None
    assert "".join([parent, *chunks]) == f"{content}{_SPLIT_FOOTER}"


def test_a_code_block_longer_than_a_message_renders_as_code_in_each() -> None:
    """The block starts its own message, and each cut through it closes and reopens the fence."""
    code = _code_lines(count=90)
    content = f"Here is the code:\n```python\n{code}\n```\nThat is all."

    parent, chunks = split_reply_for_discord(content=content, footer=_SPLIT_FOOTER)

    messages = [parent, *chunks]
    assert parent == "Here is the code:\n"
    assert len(chunks) > 1, "the block did not need cutting, so this test proves nothing"
    assert all(chunk.startswith("```python\n") and chunk.count("```") == 2 for chunk in chunks)
    assert all(any(line in chunk for chunk in chunks) for line in code.splitlines())
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in messages)
    assert "".join(messages).replace("\n``````python", "") == f"{content}{_SPLIT_FOOTER}"


def test_reopening_a_code_block_never_puts_back_what_the_cut_took() -> None:
    """A fence line too long to be a language tag is not carried over, so every cut shrinks the rest.

    Carried over whole, it would be re-added as fast as it is cut off, and the split, which runs
    on the event loop, would never end.
    """
    text = "```" + "a" * 1200 + "\n" + "b" * 1500 + "\n```\nok"

    head, rest = _cut_cleanly(text=text, budget=DISCORD_MESSAGE_LIMIT, earliest=0)

    assert len(head) <= DISCORD_MESSAGE_LIMIT
    assert len(rest) < len(text) - DISCORD_MESSAGE_LIMIT // 2


def test_split_reply_for_discord_handles_zero_max_messages() -> None:
    """`max_messages=0` still truncates cleanly instead of bypassing the cap."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    parent, chunks = split_reply_for_discord(
        content="x" * (DISCORD_MESSAGE_LIMIT + 100), footer=footer, max_messages=0
    )

    assert chunks == []
    assert len(parent) <= DISCORD_MESSAGE_LIMIT
    assert TRUNCATED_NOTICE.strip() in parent
    assert parent.endswith(footer)
