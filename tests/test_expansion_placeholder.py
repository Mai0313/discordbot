"""Tests for the reply slot a link expansion claims before it has anything to show."""

import pytest
from nextcord import Embed, HTTPException

from discordbot.utils.expansion_placeholder import ExpansionPlaceholder, send_expansion_placeholder

from tests.helpers.casting import as_message, make_forbidden, make_not_found
from tests.helpers.discord_mocks import FakeDiscordMessage

_TEXT = "-# 正在讀取貼文⋯"


async def _placeholder() -> tuple[FakeDiscordMessage, ExpansionPlaceholder]:
    """Returns the message carrying the link and the placeholder posted under it."""
    source = FakeDiscordMessage()
    placeholder = await send_expansion_placeholder(message=as_message(fake=source), text=_TEXT)
    assert placeholder is not None  # a FakeDiscordMessage never refuses the reply
    return source, placeholder


async def test_the_placeholder_replies_to_the_message_carrying_the_link() -> None:
    """It has to be a reply, or it is just another message somewhere below the link."""
    source, placeholder = await _placeholder()

    assert source.replies[0]["content"] == _TEXT
    assert source.replies[0]["mention_author"] is False
    assert placeholder.message is source.reply_messages[0]


async def test_an_expansion_with_no_text_clears_the_placeholder_with_an_empty_string() -> None:
    """None would leave the placeholder line sitting above the card it was replaced by.

    An expansion always carries a file — its media, or the embed spacer — so the edit goes out
    as multipart, and nextcord drops a None content out of that body instead of clearing it.
    """
    source, placeholder = await _placeholder()

    await placeholder.deliver(embeds=[Embed(description="the card")])

    assert source.reply_messages[0].edits[0]["content"] == ""


async def test_a_delivered_placeholder_is_never_taken_back() -> None:
    """`discard` runs unconditionally in a `finally`, so delivery has to disarm it."""
    source, placeholder = await _placeholder()

    await placeholder.deliver(embeds=[Embed(description="the card")])
    await placeholder.discard()

    assert source.reply_messages[0].deleted is False


async def test_an_undelivered_placeholder_is_removed() -> None:
    """A failed expansion leaves the reaction and nothing else."""
    source, placeholder = await _placeholder()

    await placeholder.discard()

    assert source.reply_messages[0].deleted is True


async def test_a_removal_that_fails_never_replaces_the_failure_that_called_it() -> None:
    """Discarding runs in a `finally` under whatever sent the expansion down that path."""
    source, placeholder = await _placeholder()

    async def refuse() -> None:
        """Fails the way a deleted placeholder or a lost connection does."""
        raise RuntimeError("discord said no")

    source.reply_messages[0].delete = refuse  # ty: ignore[invalid-assignment]

    await placeholder.discard()


async def _refusing(error: Exception) -> FakeDiscordMessage:
    """Builds a message whose reply is refused the way a real channel refuses one."""
    source = FakeDiscordMessage()

    async def refuse(**kwargs: object) -> FakeDiscordMessage:
        """Raises instead of posting, keeping the recorded payload out of the way."""
        del kwargs
        raise error

    source.reply = refuse  # ty: ignore[invalid-assignment]
    return source


async def test_a_channel_that_refuses_the_placeholder_answers_none() -> None:
    """A channel that will not take one line will not take the card either.

    Answering None rather than raising is what keeps a read-only channel from logging an
    unexpected error per pasted link: the caller marks the expansion failed and reads nothing.
    """
    source = await _refusing(error=make_forbidden())

    assert await send_expansion_placeholder(message=as_message(fake=source), text=_TEXT) is None


async def test_a_link_deleted_before_the_placeholder_answers_none() -> None:
    """Discord answers 50035 rather than NotFound when the message replied to is gone."""
    gone = HTTPException(
        response=make_not_found().response, message={"code": 50035, "message": "Invalid Form Body"}
    )
    source = await _refusing(error=gone)

    assert await send_expansion_placeholder(message=as_message(fake=source), text=_TEXT) is None


async def test_any_other_send_failure_still_raises() -> None:
    """Only the two refusals are routine; anything else is the listener's to report."""
    source = await _refusing(error=RuntimeError("discord exploded"))

    with pytest.raises(RuntimeError):
        await send_expansion_placeholder(message=as_message(fake=source), text=_TEXT)
