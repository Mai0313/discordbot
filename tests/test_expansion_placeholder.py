"""Tests for the reply slot a link expansion claims before it has anything to show."""

import pytest
from nextcord import Embed, Message, HTTPException

from discordbot.utils.expansion_placeholder import (
    EXPANSION_WORKING_EMOJI,
    ExpansionPlaceholder,
    send_expansion_placeholder,
    resume_expansion_placeholders,
)

from tests.helpers.casting import as_bot, as_message, make_forbidden, make_not_found
from tests.helpers.discord_mocks import FakeDiscordMessage

_TEXT = "-# 正在讀取貼文⋯"
_SOURCE = "threads"
_URL = "https://www.threads.com/@someone/post/AAA"


async def _placeholder() -> tuple[FakeDiscordMessage, ExpansionPlaceholder]:
    """Returns the message carrying the link and the placeholder posted under it."""
    source = FakeDiscordMessage()
    placeholder = await send_expansion_placeholder(
        message=as_message(fake=source), text=_TEXT, source=_SOURCE, url=_URL
    )
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

    assert (
        await send_expansion_placeholder(
            message=as_message(fake=source), text=_TEXT, source=_SOURCE, url=_URL
        )
        is None
    )


async def test_a_link_deleted_before_the_placeholder_answers_none() -> None:
    """Discord answers 50035 rather than NotFound when the message replied to is gone."""
    gone = HTTPException(
        response=make_not_found().response, message={"code": 50035, "message": "Invalid Form Body"}
    )
    source = await _refusing(error=gone)

    assert (
        await send_expansion_placeholder(
            message=as_message(fake=source), text=_TEXT, source=_SOURCE, url=_URL
        )
        is None
    )


async def test_any_other_send_failure_still_raises() -> None:
    """Only the two refusals are routine; anything else is the listener's to report."""
    source = await _refusing(error=RuntimeError("discord exploded"))

    with pytest.raises(RuntimeError):
        await send_expansion_placeholder(
            message=as_message(fake=source), text=_TEXT, source=_SOURCE, url=_URL
        )


class _FakeChannel:
    """A channel the restart sweep can read its two messages back out of.

    Only `fetch_message` is here because that is the whole of what `MessageFetcher` asks
    for, and being a Protocol rather than `nextcord.abc.Messageable` is what lets this
    double satisfy the `isinstance` in the sweep at all.
    """

    def __init__(self, *, messages: dict[int, FakeDiscordMessage]) -> None:
        """Holds the messages by id; anything else answers the way a deleted one does."""
        self.id = 2
        self.messages = messages

    async def fetch_message(self, id: int, /) -> Message:  # noqa: A002 -- nextcord's own name
        """Answers with the stored message, or `NotFound` when it is gone."""
        message = self.messages.get(id)
        if message is None:
            raise make_not_found()
        return as_message(fake=message)


class _FakeBot:
    """A bot that resolves exactly one channel, the way `on_ready` finds a cached one."""

    def __init__(self, *, channel: _FakeChannel | None) -> None:
        """Holds the channel `get_channel` answers with, or None for one that is gone."""
        self.channel = channel

    def get_channel(self, channel_id: int, /) -> _FakeChannel | None:
        """Answers from the cache, as nextcord's own does."""
        del channel_id
        return self.channel

    async def fetch_channel(self, channel_id: int, /) -> _FakeChannel:
        """The uncached fallback; a channel that is really gone answers `NotFound`."""
        del channel_id
        if self.channel is None:
            raise make_not_found()
        return self.channel


class _RecordingExpand:
    """Stands in for a cog's `_expand`, recording the call and choosing its outcome."""

    def __init__(self, *, deliver: bool = True) -> None:
        """Records nothing yet; `deliver` picks between a successful and a failed expansion."""
        self.calls: list[dict[str, object]] = []
        self.deliver = deliver

    async def __call__(
        self, *, message: Message, url: str, current_emoji: str, placeholder: ExpansionPlaceholder
    ) -> None:
        """Records what it was handed, then delivers or returns as a real `_expand` does."""
        self.calls.append({"message": message, "url": url, "current_emoji": current_emoji})
        if self.deliver:
            await placeholder.deliver(embeds=[Embed(description="the card")])


async def _interrupted() -> tuple[FakeDiscordMessage, FakeDiscordMessage, _FakeChannel]:
    """Leaves a recorded placeholder behind with nothing delivered onto it, as a restart does."""
    source = FakeDiscordMessage()
    # The double gives every message the same id, and the sweep tells the two apart by id.
    source.id = 11
    placeholder = await send_expansion_placeholder(
        message=as_message(fake=source), text=_TEXT, source=_SOURCE, url=_URL
    )
    assert placeholder is not None
    posted = source.reply_messages[0]
    return source, posted, _FakeChannel(messages={source.id: source, posted.id: posted})


async def test_a_restart_runs_the_interrupted_expansion_again() -> None:
    """The entry point is all that is new: what runs is the cog's own `_expand`, unchanged."""
    source, placeholder_message, channel = await _interrupted()
    expand = _RecordingExpand()

    await resume_expansion_placeholders(
        bot=as_bot(fake=_FakeBot(channel=channel)), source=_SOURCE, expand=expand
    )

    assert len(expand.calls) == 1
    assert expand.calls[0]["url"] == _URL
    # The reaction the listener already put on the source, so the resumed expansion replaces
    # that one rather than leaving a second working ring behind.
    assert expand.calls[0]["current_emoji"] == EXPANSION_WORKING_EMOJI
    assert placeholder_message.edits, "the resumed expansion never reached its placeholder"
    assert source.deleted is False


async def test_a_resumed_expansion_that_fails_takes_its_placeholder_back() -> None:
    """The fallback the user asked for: what cannot be retried is at least not left behind."""
    _source, placeholder_message, channel = await _interrupted()

    await resume_expansion_placeholders(
        bot=as_bot(fake=_FakeBot(channel=channel)),
        source=_SOURCE,
        expand=_RecordingExpand(deliver=False),
    )

    assert placeholder_message.deleted is True


async def test_a_link_withdrawn_while_the_bot_was_down_is_never_expanded() -> None:
    """Posting nothing is what a live expansion does for a deleted source, restart or not."""
    source, placeholder_message, channel = await _interrupted()
    del channel.messages[source.id]
    expand = _RecordingExpand()

    await resume_expansion_placeholders(
        bot=as_bot(fake=_FakeBot(channel=channel)), source=_SOURCE, expand=expand
    )

    assert expand.calls == []
    assert placeholder_message.deleted is True


async def test_a_placeholder_removed_by_hand_ends_the_row() -> None:
    """Nothing is left to deliver onto, so the row goes and the sweep never runs twice."""
    _source, placeholder_message, channel = await _interrupted()
    del channel.messages[placeholder_message.id]
    expand = _RecordingExpand()

    await resume_expansion_placeholders(
        bot=as_bot(fake=_FakeBot(channel=channel)), source=_SOURCE, expand=expand
    )
    await resume_expansion_placeholders(
        bot=as_bot(fake=_FakeBot(channel=channel)), source=_SOURCE, expand=expand
    )

    assert expand.calls == []


async def test_a_delivered_expansion_leaves_nothing_for_a_restart_to_find() -> None:
    """The row is the in-flight state alone; a finished expansion must never be redone."""
    _source, _placeholder, channel = await _interrupted()
    expand = _RecordingExpand()
    bot = as_bot(fake=_FakeBot(channel=channel))

    await resume_expansion_placeholders(bot=bot, source=_SOURCE, expand=expand)
    await resume_expansion_placeholders(bot=bot, source=_SOURCE, expand=expand)

    assert len(expand.calls) == 1


async def test_a_cog_never_resumes_another_platforms_expansion() -> None:
    """Each cog sweeps its own rows, so a Threads row can never reach the Douyin `_expand`."""
    _source, _placeholder, channel = await _interrupted()
    expand = _RecordingExpand()

    await resume_expansion_placeholders(
        bot=as_bot(fake=_FakeBot(channel=channel)), source="douyin", expand=expand
    )

    assert expand.calls == []


async def test_a_channel_the_bot_can_no_longer_reach_drops_the_row() -> None:
    """A kicked guild or a deleted channel must not stall the rest of the backlog."""
    _source, _placeholder, channel = await _interrupted()
    expand = _RecordingExpand()

    await resume_expansion_placeholders(
        bot=as_bot(fake=_FakeBot(channel=None)), source=_SOURCE, expand=expand
    )
    await resume_expansion_placeholders(
        bot=as_bot(fake=_FakeBot(channel=channel)), source=_SOURCE, expand=expand
    )

    assert expand.calls == []
