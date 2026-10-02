"""`/ask`: the conversational entry point for the contexts a user install reaches.

The fakes here are local rather than pulled from `tests/helpers/discord_mocks.py` because this
route needs something that package deliberately does not model: real `nextcord.Message` objects,
built over a connection state, on a channel the bot is not a member of. Everything else about a
`/ask` turn is the ordinary pipeline, and `tests/test_gen_reply.py` already covers that.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from datetime import datetime, timedelta
from collections.abc import Callable, Awaitable

import pytest
from nextcord import Message, Attachment, ChannelType, PartialMessageable
from nextcord.enums import InteractionContextType
from nextcord.utils import utcnow

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.gen_reply import ask_store
from discordbot.typings.timeouts import INTERACTION_DELIVERY_MARGIN_SECONDS
from discordbot.cogs.gen_reply.cog import ReplyGeneratorCogs
from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.cogs.gen_reply.recall import RecallContext
from discordbot.cogs.gen_reply.context import ReplyContext, ReplyContextBuilder
from discordbot.cogs.gen_reply.surface import INTERACTION_FOLLOWUP_LIMIT, TurnSurface
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.pipeline import ReplyPipeline
from discordbot.cogs.gen_reply.ask_store import load_ask_turns, record_ask_turn
from discordbot.cogs.gen_reply.streaming import TRUNCATED_NOTICE, ResponseStreamer, _cut_cleanly
from discordbot.cogs.gen_reply.ask_message import build_ask_message, rebuild_conversation

from tests.helpers.casting import as_bot
from tests.helpers.link_sources import hosting_off_planner

if TYPE_CHECKING:
    from nextcord.types.message import Attachment as AttachmentPayload

# A real Discord snowflake, so `Message.created_at` resolves to a real moment rather than 1970.
ASK_SNOWFLAKE = 1517561877973045349
BOT_USER_ID = 999
ASKER_ID = 4242
CHANNEL_ID = 777
GUILD_ID = 31337


class _FakeState:
    """The slice of `nextcord.ConnectionState` a synthesized message actually reads."""

    # Stored by every `Attachment` for its own download path, which nothing here follows.
    http = None

    def _get_guild(self, guild_id: int | None) -> None:
        """The bot is in no guild here, which is the whole premise of this route."""
        del guild_id

    def _get_guild_channel(self, data: object) -> tuple[None, None]:
        """Only reached for a resolved reference, which a `/ask` message never carries."""
        del data
        return None, None


class _FakeFollowup:
    """Records follow-up POSTs and hands back a message that can be edited."""

    def __init__(self) -> None:
        """Initializes the recorded sends."""
        self.sent: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 -- passthrough Discord payload
        """Records one follow-up and returns a stand-in for the message it created."""
        self.sent.append(kwargs)
        return SimpleNamespace(id=len(self.sent), content=kwargs.get("content"))


class _FakeAskInteraction:
    """The interaction surface of a user-installed `/ask`, with nothing the route never reads."""

    def __init__(
        self, guild_id: int | None = GUILD_ID, context: InteractionContextType | None = None
    ) -> None:
        """Initializes the invocation's identity, its channel and its response records."""
        self.id = ASK_SNOWFLAKE
        self.guild_id = guild_id
        self.context = context or (
            InteractionContextType.guild if guild_id is not None else InteractionContextType.bot_dm
        )
        self.channel_id = CHANNEL_ID
        self.user = SimpleNamespace(id=ASKER_ID, name="asker", display_name="Asker")
        self.data: dict[str, Any] | None = None
        self._state = _FakeState()
        self.channel = PartialMessageable(
            state=self._state,  # ty: ignore[invalid-argument-type] -- the state slice a partial channel reads
            id=CHANNEL_ID,
            type=ChannelType.text if guild_id is not None else ChannelType.private,
        )
        self.followup = _FakeFollowup()
        self.edits: list[dict[str, Any]] = []
        self.deferred = False
        # Settable so a test can move the token's window without inventing a snowflake; the real
        # property derives it from `id` the same way.
        self.created_at = utcnow()
        self.response = SimpleNamespace(defer=self._defer, is_done=lambda: self.deferred)
        self.client = SimpleNamespace(
            user=SimpleNamespace(id=BOT_USER_ID, name="pocat", discriminator="0")
        )

    @property
    def expires_at(self) -> datetime:
        """When Discord invalidates this token, mirroring `nextcord.Interaction.expires_at`.

        Fifteen minutes once the response is deferred and three seconds before it, which is the
        same split the real property makes. Modelled rather than stubbed to a fixed number,
        because that split is exactly what `delivery_budget_seconds` relies on being told.
        """
        return self.created_at + (
            timedelta(minutes=15) if self.response.is_done() else timedelta(seconds=3)
        )

    async def _defer(self) -> None:
        """Records that the three-second window was answered before any work started."""
        self.deferred = True

    async def edit_original_message(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 -- passthrough Discord payload
        """Records an edit of the deferred response and returns its message."""
        self.edits.append(kwargs)
        return SimpleNamespace(id=1, content=kwargs.get("content"))


# One attachment option's resolved payload, as Discord sends it back in `interaction.data`.
_CAT_PNG: dict[str, Any] = {
    "id": "9",
    "filename": "cat.png",
    "size": 1024,
    "url": "https://cdn.example/cat.png",
    "proxy_url": "https://cdn.example/cat.png",
    "content_type": "image/png",
}


def _interaction(**kwargs: Any) -> Any:  # noqa: ANN401 -- the fake stands in for a generic Interaction
    """Builds the fake invocation, untyped so it can stand in for `Interaction[Bot]`."""
    return _FakeAskInteraction(**kwargs)


def _ask_message(interaction: Any) -> Message:  # noqa: ANN401 -- see `_interaction`
    """The message the pipeline would answer for this invocation."""
    return build_ask_message(interaction=interaction, question="在幹嘛")


def test_the_synthesized_message_is_the_invocation_itself() -> None:
    """Id, author and text all come off the interaction, and no guild comes off the cache."""
    interaction = _interaction()

    message = _ask_message(interaction=interaction)

    assert message.id == ASK_SNOWFLAKE
    assert message.created_at.year == 2026
    assert message.author is interaction.user
    assert message.content == "在幹嘛"
    assert message.channel.id == CHANNEL_ID
    # Every one of these is read by the pipeline and every one is unset unless the payload
    # carries its key, because `Message` has `__slots__` and no defaults.
    assert message.guild is None
    assert message.mentions == []
    assert message.role_mentions == []
    assert message.attachments == []
    assert message.reference is None


def test_an_attached_file_reaches_the_message() -> None:
    """The option's payload is read back out of the interaction, not off the bound object."""
    interaction = _interaction()
    interaction.data = {"resolved": {"attachments": {"9": _CAT_PNG}}}

    message = _ask_message(interaction=interaction)

    assert [attachment.filename for attachment in message.attachments] == ["cat.png"]


@pytest.mark.parametrize(
    ("context", "guild_id", "expected_guild", "expected_direct"),
    [
        (InteractionContextType.guild, GUILD_ID, GUILD_ID, False),
        (InteractionContextType.bot_dm, None, None, True),
        (InteractionContextType.private_channel, None, None, False),
    ],
)
def test_only_a_dm_with_the_bot_counts_as_a_direct_message(
    context: InteractionContextType,
    guild_id: int | None,
    expected_guild: int | None,
    expected_direct: bool,
) -> None:
    """A group DM must not read as one, because `dm_partner_id` opens every compartment.

    `private_channel` covers a group DM and a DM between two other people alike, and the
    channel object is the same `PartialMessageable` for both plus for a real 1:1 DM, so the
    interaction's own context is the only thing that can tell them apart. Reading it wrong
    hands the asker's private tier and every server's facts to a channel full of strangers.
    """
    interaction = _interaction(guild_id=guild_id, context=context)

    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    assert surface.guild_id == expected_guild
    assert surface.is_direct_message is expected_direct


def _toolkit(interaction: Any) -> ReplyToolkit:  # noqa: ANN401 -- see `_interaction`
    """A toolkit whose clients no test here ever calls through."""
    return ReplyToolkit(bot=interaction.client, openai_client=SimpleNamespace(), gemini_api_key="")


@pytest.mark.usefixtures("memory_isolated_dir")
@pytest.mark.parametrize(
    ("context", "guild_id", "expected"),
    [
        (
            InteractionContextType.guild,
            GUILD_ID,
            RecallContext(guild_id=GUILD_ID, dm_partner_id=None),
        ),
        (
            InteractionContextType.bot_dm,
            None,
            RecallContext(guild_id=None, dm_partner_id=ASKER_ID),
        ),
        (
            InteractionContextType.private_channel,
            None,
            RecallContext(guild_id=None, dm_partner_id=None),
        ),
    ],
)
def test_an_ask_turn_reads_memory_scoped_to_where_it_happens(
    context: InteractionContextType, guild_id: int | None, expected: RecallContext
) -> None:
    """The recall plan takes both facts off the surface, since the message carries neither.

    `Message.guild` is None on this route whatever the interaction says, so a plan reading it
    would take a group DM for the asker's own DM, handing a channel full of strangers every
    compartment the asker has, and would drop a server turn's guild compartment.
    """
    interaction = _interaction(guild_id=guild_id, context=context)
    builder = ReplyContextBuilder(
        toolkit=_toolkit(interaction=interaction),
        surface=TurnSurface.for_interaction(interaction=interaction, question="在幹嘛"),
    )

    assert builder.plan_recall().recall_context == expected


@pytest.mark.parametrize(
    ("context", "guild_id", "expected_source"),
    [
        (InteractionContextType.guild, GUILD_ID, f"source: guild {GUILD_ID}"),
        (InteractionContextType.bot_dm, None, "source: dm"),
    ],
)
def test_an_ask_turn_stamps_its_memory_with_where_it_happens(
    monkeypatch: pytest.MonkeyPatch,
    context: InteractionContextType,
    guild_id: int | None,
    expected_source: str,
) -> None:
    """The read and the write must name the same compartment, or memory goes in and never out.

    `Message.guild` is None here whatever the interaction says, so a source stamp taken from
    the message would file a server turn's `source_only` observations under `dm/` while the next
    turn in the same server read only `global` and that guild's own.
    """
    scheduled: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.answer.schedule_memory_update",
        lambda **kwargs: scheduled.append(kwargs),
    )
    interaction = _interaction(guild_id=guild_id, context=context)
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")
    turn = AnswerTurn(
        config=LLMConfig.model_construct(),
        media_delivery=hosting_off_planner(),
        toolkit=_toolkit(interaction=interaction),
        surface=surface,
    )

    turn._schedule_memory_updates(
        context=ReplyContext(),
        full_reply="好",
        streamer=ResponseStreamer(
            message=surface.message, surface=surface, media_delivery=hosting_off_planner()
        ),
    )

    assert [update["subject"] for update in scheduled] == [
        f"target_user_id: {ASKER_ID}\n{expected_source}"
    ]


def test_a_rebuilt_conversation_gives_the_bot_its_own_turns() -> None:
    """The reply half has to carry the bot's id, or it reaches the model as another user line."""
    interaction = _interaction()

    messages = rebuild_conversation(
        turns=[
            ask_store.AskTurn(message_id=ASK_SNOWFLAKE, question="早", answer="早安"),
            ask_store.AskTurn(message_id=ASK_SNOWFLAKE + 2, question="午安", answer="午安啊"),
        ],
        interaction=interaction,
    )

    assert [m.content for m in messages] == ["早", "早安", "午安", "午安啊"]
    assert [m.author.id for m in messages] == [ASKER_ID, BOT_USER_ID, ASKER_ID, BOT_USER_ID]
    # Distinct ids, so one turn's log line and attachment-cache key never answer for another's.
    assert len({m.id for m in messages}) == len(messages)


@pytest.mark.usefixtures("ask_isolated_db")
async def test_the_store_replays_a_conversation_oldest_first() -> None:
    """What went in comes back in transcript order, scoped to one person in one channel."""
    for index, (question, answer) in enumerate([("一", "1"), ("二", "2"), ("三", "3")]):
        await record_ask_turn(
            channel_id=CHANNEL_ID,
            user_id=ASKER_ID,
            message_id=ASK_SNOWFLAKE + index * 2,
            question=question,
            answer=answer,
        )
    await record_ask_turn(
        channel_id=CHANNEL_ID,
        user_id=ASKER_ID + 1,
        message_id=ASK_SNOWFLAKE,
        question="別人的",
        answer="別人的回覆",
    )

    turns = await load_ask_turns(channel_id=CHANNEL_ID, user_id=ASKER_ID, limit=10)

    assert [turn.question for turn in turns] == ["一", "二", "三"]
    assert [turn.answer for turn in turns] == ["1", "2", "3"]


@pytest.mark.usefixtures("ask_isolated_db")
async def test_the_store_keeps_only_its_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    """A conversation someone keeps up for a year must not grow the table without limit."""
    monkeypatch.setattr(ask_store, "ASK_TURN_RETENTION", 2)
    for index in range(4):
        await record_ask_turn(
            channel_id=CHANNEL_ID,
            user_id=ASKER_ID,
            message_id=ASK_SNOWFLAKE + index * 2,
            question=str(index),
            answer=str(index),
        )

    turns = await load_ask_turns(channel_id=CHANNEL_ID, user_id=ASKER_ID, limit=10)

    assert [turn.question for turn in turns] == ["2", "3"]


async def test_the_first_send_edits_the_deferred_response_then_follows_up() -> None:
    """The deferred response is one free message; everything after it spends the budget."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")
    assert surface.answer_capacity(has_landed_reply=False) == INTERACTION_FOLLOWUP_LIMIT + 1

    await surface.send(content="first")
    await surface.send(content="second")

    assert [edit["content"] for edit in interaction.edits] == ["first"]
    assert [sent["content"] for sent in interaction.followup.sent] == ["second"]
    # One follow-up spent, and the answer sitting on the second one can still be edited.
    assert surface.answer_capacity(has_landed_reply=True) == INTERACTION_FOLLOWUP_LIMIT


async def test_a_follow_up_never_replies_into_a_channel_the_bot_is_not_in() -> None:
    """`previous.reply` is a plain channel send, which is a 403 here and loses the answer's tail."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    def _explode(**kwargs: object) -> None:
        """Fails if the chunk ever goes out as a reply to the previous message."""
        del kwargs
        raise AssertionError("a follow-up must not reply into the channel")

    await surface.follow_up(previous=SimpleNamespace(reply=_explode), content="tail")  # ty: ignore[invalid-argument-type] -- only `.reply` is reachable here

    assert [sent["content"] for sent in interaction.followup.sent] == ["tail"]


def test_an_answer_past_the_follow_up_budget_says_it_was_cut() -> None:
    """Silently losing the tail reads as the model being interrupted; the notice says otherwise."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    content = "字" * 9000

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=footer, max_messages=2
    )

    assert len(chunks) == 1
    assert len(parent) <= 2000
    assert len(chunks[0]) <= 2000
    assert TRUNCATED_NOTICE.strip() in f"{parent}{chunks[0]}"
    assert f"{parent}{chunks[0]}".endswith(footer)


@pytest.mark.parametrize("max_messages", [1, 2])
def test_only_an_answer_past_the_budget_is_cut(max_messages: int) -> None:
    """The notice's room is held back only once something must go, so a fitting answer is whole."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    fits = "字" * (max_messages * DISCORD_MESSAGE_LIMIT - len(footer))

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=fits, footer=footer, max_messages=max_messages
    )
    over_parent, over_chunks = ResponseStreamer._split_reply_for_discord(
        content=f"{fits}字", footer=footer, max_messages=max_messages
    )

    fit = [parent, *chunks]
    assert "".join(fit) == f"{fits}{footer}"
    assert len(fit) <= max_messages
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in fit)
    over = [over_parent, *over_chunks]
    assert len(over) == max_messages
    assert TRUNCATED_NOTICE.strip() in over[-1]


_SPLIT_FOOTER = "\n\n-# model · ⬆ 1 ⬇ 2 · $0.00000001"
_FAMILY_EMOJI = "\U0001f468\u200d\U0001f469\u200d\U0001f467"


def _code_lines(count: int) -> str:
    """Python source of `count` distinct lines, each with spaces a word break could take."""
    return "\n".join(
        f"value_{index:03d} = compute(index={index}, scale=2.5)" for index in range(count)
    )


def test_an_answer_that_fits_only_packed_full_is_not_cut_for_its_breaks() -> None:
    """Clean cuts leave room unused, which a capped surface cannot spare from an answer that fits."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    content = "\n\n".join(["字" * 1400, "字" * 1400, ""])
    content += "字" * (2 * DISCORD_MESSAGE_LIMIT - len(footer) - len(content))

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=footer, max_messages=2
    )
    _, uncapped_chunks = ResponseStreamer._split_reply_for_discord(content=content, footer=footer)

    assert len(uncapped_chunks) > 1, "clean cuts fit the cap anyway, so this test proves nothing"
    assert len(chunks) == 1
    assert f"{parent}{chunks[0]}" == f"{content}{footer}"
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in [parent, *chunks])


def test_an_answer_past_the_budget_is_still_cut_where_a_reader_would() -> None:
    """Running out of messages changes where the answer ends, not how its messages are cut."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    mention = "<@123456789012345678>"
    content = "b" * 1990 + f" {mention} thanks\n\n" + "\n\n".join(["word " * 100] * 25)

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=footer, max_messages=6
    )

    messages = [parent, *chunks]
    assert len(messages) == 6
    assert parent == "b" * 1990
    assert any(mention in message for message in messages)
    assert messages[-1].endswith(f"{TRUNCATED_NOTICE}{footer}")
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in messages)


def test_the_last_message_a_budget_allows_shows_what_it_can_of_a_code_block() -> None:
    """With no message after it, ending before the block would leave the room empty for nothing."""
    content = "prose " * 333 + f"\nHere it is:\n```python\n{_code_lines(count=150)}\n```"

    _parent, chunks = ResponseStreamer._split_reply_for_discord(
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

    parent, chunks = ResponseStreamer._split_reply_for_discord(content=content, footer=footer)

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
    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=_SPLIT_FOOTER
    )

    messages = [parent, *chunks]
    assert len(messages) > 1, "the answer did not split, so this test proves nothing"
    assert any(whole in message for message in messages)
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in messages)
    assert "".join(messages) == f"{content}{_SPLIT_FOOTER}"


def test_the_footer_still_follows_answer_text_after_a_cut() -> None:
    """A footer alone loses its blank line to Discord's trim, and history then fails to strip it."""
    content = "para " * 300 + "\n\n" + "z" * 1980

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=_SPLIT_FOOTER
    )

    assert parent == "para " * 300
    assert USAGE_FOOTER_RE.search(chunks[-1].strip()) is not None
    assert "".join([parent, *chunks]) == f"{content}{_SPLIT_FOOTER}"


def test_a_code_block_longer_than_a_message_renders_as_code_in_each() -> None:
    """The block starts its own message, and each cut through it closes and reopens the fence."""
    code = _code_lines(count=90)
    content = f"Here is the code:\n```python\n{code}\n```\nThat is all."

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=_SPLIT_FOOTER
    )

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


async def test_a_dropped_clip_is_written_where_it_cannot_be_reacted() -> None:
    """The ⏱️ / ⚠️ is the only trace a dropped clip leaves, so it must survive having no message."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    await surface.hint(emoji="⚠️")
    await surface.hint(emoji="⚠️")
    await surface.hint(emoji="⏱️")

    assert surface.take_hints() == ["⚠️", "⏱️"]
    assert surface.take_hints() == []


@pytest.mark.usefixtures("ask_isolated_db")
async def test_the_surface_records_a_turn_and_replays_it_next_time() -> None:
    """One turn's answer is the next turn's history, since Discord keeps none of it for us."""
    interaction = _interaction()
    first = TurnSurface.for_interaction(interaction=interaction, question="你叫什麼")
    assert await first.fetch_history(limit=500) == []

    await first.record_turn(answer="我叫破貓")

    later = TurnSurface.for_interaction(interaction=interaction, question="剛剛說了什麼")
    history = await later.fetch_history(limit=500)
    assert [message.content for message in history] == ["你叫什麼", "我叫破貓"]
    assert [message.author.id for message in history] == [ASKER_ID, BOT_USER_ID]


@pytest.mark.usefixtures("ask_isolated_db")
async def test_a_gateway_turn_records_nothing() -> None:
    """Discord's own channel history is the record there, so the store must stay empty."""
    interaction = _interaction()
    surface = TurnSurface.for_message(message=_ask_message(interaction=interaction))

    await surface.record_turn(answer="ignored")

    assert await load_ask_turns(channel_id=CHANNEL_ID, user_id=ASKER_ID, limit=10) == []


def _ask_cog(
    interaction: Any,  # noqa: ANN401 -- see `_interaction`
    monkeypatch: pytest.MonkeyPatch,
    run: Callable[[ReplyPipeline], Awaitable[None]],
) -> ReplyGeneratorCogs:
    """A cog whose `/ask` gets as far as handing the pipeline the turn, and no further."""
    cog = ReplyGeneratorCogs(
        bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=BOT_USER_ID, name="pocat")))
    )
    cog.__dict__["toolkit"] = _toolkit(interaction=interaction)
    monkeypatch.setattr(ReplyPipeline, "run", run)
    return cog


async def test_ask_defers_before_anything_slower_than_three_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The token dies after three seconds, and every phase of a turn is slower than that."""
    ran: list[tuple[TurnSurface, str]] = []

    async def _run(pipeline: ReplyPipeline) -> None:
        """Records the turn the command would have run."""
        ran.append((pipeline.surface, pipeline.user_prompt))

    interaction = _interaction()
    cog = _ask_cog(interaction=interaction, monkeypatch=monkeypatch, run=_run)

    await cog.ask(interaction, question="在幹嘛", attachment=None)

    assert interaction.deferred is True
    surface, user_prompt = ran[0]
    assert user_prompt == "在幹嘛"
    assert surface.interaction is interaction
    assert surface.guild_id == GUILD_ID
    assert surface.message.id == ASK_SNOWFLAKE


async def test_an_ask_turn_never_reacts_to_its_synthesized_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nobody posted that message, so every status reaction would be a REST call that 404s."""
    enabled: list[bool] = []

    async def _run(pipeline: ReplyPipeline) -> None:
        """Records whether the turn's status chain would react."""
        enabled.append(pipeline.reactions.enabled)

    interaction = _interaction()
    cog = _ask_cog(interaction=interaction, monkeypatch=monkeypatch, run=_run)

    await cog.ask(interaction, question="在幹嘛", attachment=None)

    assert enabled == [False]


async def test_ask_answers_a_blank_question_without_running_a_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whitespace-only option would otherwise route on nothing at all."""

    async def _never(pipeline: ReplyPipeline) -> None:
        """Fails if a blank question ever reaches the pipeline."""
        del pipeline
        raise AssertionError("a blank question must not run a turn")

    interaction = _interaction()
    cog = _ask_cog(interaction=interaction, monkeypatch=monkeypatch, run=_never)

    await cog.ask(interaction, question="   ", attachment=None)

    assert [edit["content"] for edit in interaction.edits] == ["?"]


async def test_ask_answers_an_attachment_sent_without_a_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file on its own is something to answer, not an empty prompt.

    The attachment reaches the turn only through the synthesized message, so that message is
    what the empty-prompt check has to read.
    """
    ran: list[tuple[TurnSurface, str]] = []

    async def _run(pipeline: ReplyPipeline) -> None:
        """Records the turn the command would have run."""
        ran.append((pipeline.surface, pipeline.user_prompt))

    interaction = _interaction()
    cog = _ask_cog(interaction=interaction, monkeypatch=monkeypatch, run=_run)
    interaction.data = {"resolved": {"attachments": {"9": _CAT_PNG}}}

    await cog.ask(
        interaction,
        question="   ",
        attachment=Attachment(data=cast("AttachmentPayload", _CAT_PNG), state=interaction._state),
    )

    assert interaction.edits == []
    ((surface, user_prompt),) = ran
    assert user_prompt == ""
    assert [attachment.filename for attachment in surface.message.attachments] == ["cat.png"]


async def test_ask_without_a_proxy_key_answers_its_deferred_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise the asker is left on Discord's "thinking" state with nothing ever landing."""
    # The SDK also accepts `OPENAI_ADMIN_KEY` from the environment, which would build the client.
    monkeypatch.delenv(name="OPENAI_ADMIN_KEY", raising=False)
    cog = ReplyGeneratorCogs(
        bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=BOT_USER_ID, name="pocat")))
    )
    cog.config = LLMConfig.model_construct()
    interaction = _interaction()

    await cog.ask(interaction, question="在幹嘛", attachment=None)

    (edit,) = interaction.edits
    assert edit["embed"].title == "Something went wrong"
    assert edit["embed"].footer.text == "OpenAIError"


class _EditableReply:
    """A landed reply that records what each edit wrote to it."""

    def __init__(self) -> None:
        """Initializes the recorded edits."""
        self.contents: list[str] = []

    async def edit(self, **kwargs: Any) -> None:  # noqa: ANN401 -- passthrough Discord payload
        """Records one content edit."""
        self.contents.append(kwargs["content"])


async def test_the_hint_line_lands_on_the_reply_but_not_in_the_transcript() -> None:
    """It has to show, and it has to stay out of what the bot is later told it said."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    reply = _EditableReply()
    streamer = ResponseStreamer(
        message=surface.message, surface=surface, reply=reply, media_delivery=hosting_off_planner()
    )
    streamer.stored_content = f"答案{footer}"
    streamer._usage_footer = footer
    await surface.hint(emoji="⚠️")

    await streamer._write_hint_line()

    assert reply.contents == [f"答案\n-# ⚠️{footer}"]
    assert streamer._without_added_lines(text=streamer.stored_content) == f"答案{footer}"


async def test_a_capped_surface_stops_chunking_where_its_budget_ends() -> None:
    """The budget arithmetic, not just the split: five follow-ups and no sixth attempt."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    reply = _EditableReply()
    streamer = ResponseStreamer(
        message=surface.message, surface=surface, reply=reply, media_delivery=hosting_off_planner()
    )

    await streamer._write_final_message(content="字" * 40000, footer=footer)

    assert len(interaction.followup.sent) == INTERACTION_FOLLOWUP_LIMIT
    written = reply.contents[-1] + "".join(sent["content"] for sent in interaction.followup.sent)
    assert TRUNCATED_NOTICE.strip() in written
    assert written.endswith(footer)


async def test_a_gateway_surface_still_reacts_instead_of_collecting() -> None:
    """Nothing about the `on_message` path changes: the hint is the reaction it always was."""
    added: list[str] = []

    class _Reactable:
        """A message that records the reactions added to it."""

        guild = None
        channel = object()

        async def add_reaction(self, emoji: str) -> None:
            """Records one reaction."""
            added.append(emoji)

    surface = TurnSurface.for_message(message=_Reactable())  # ty: ignore[invalid-argument-type] -- only `.add_reaction` is reachable here

    await surface.hint(emoji="⚠️")

    assert added == ["⚠️"]
    assert surface.take_hints() == []


def _deferred_surface(**kwargs: Any) -> TurnSurface:  # noqa: ANN401 -- forwards to the untyped fake
    """The surface `/ask` builds, after the defer that opens the fifteen-minute window."""
    interaction = _interaction(**kwargs)
    interaction.deferred = True
    return TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")


def test_a_gateway_turn_has_nothing_running_out() -> None:
    """`on_message` answers into a channel that is simply there, so no route is bounded by it."""
    message = _ask_message(interaction=_interaction())

    assert TurnSurface.for_message(message=message).delivery_budget_seconds() is None


def test_the_delivery_budget_holds_back_what_answering_will_cost() -> None:
    """A `/ask` turn may spend its token up to the margin the answer itself still needs."""
    budget = _deferred_surface().delivery_budget_seconds()

    assert budget is not None
    # Fifteen minutes of token, less the margin, less however long the test took to get here.
    usable = 15 * 60 - INTERACTION_DELIVERY_MARGIN_SECONDS
    assert usable - 5.0 < budget <= usable


def test_a_window_already_gone_leaves_nothing_to_spend() -> None:
    """Past the token the budget floors at zero, so a route starts nothing it cannot finish."""
    interaction = _interaction()
    interaction.deferred = True
    interaction.created_at = utcnow() - timedelta(minutes=30)
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    assert surface.delivery_budget_seconds() == 0.0


def test_an_undeferred_invocation_has_nothing_to_spend_either() -> None:
    """Three seconds is what an unanswered interaction really holds, so it may start nothing slow.

    `/ask` defers before it builds its surface, so nothing reaches this today. Pinned because the
    budget reads `Interaction.expires_at`, which tells the truth about both of Discord's windows
    rather than assuming the deferred one.
    """
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    assert surface.delivery_budget_seconds() == 0.0
