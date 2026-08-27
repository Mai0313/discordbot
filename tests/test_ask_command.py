"""`/ask`: the conversational entry point for the contexts a user install reaches.

The fakes here are local rather than pulled from `tests/helpers/discord_mocks.py` because this
route needs something that package deliberately does not model: real `nextcord.Message` objects,
built over a connection state, on a channel the bot is not a member of. Everything else about a
`/ask` turn is the ordinary pipeline, and `tests/test_gen_reply.py` already covers that.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from nextcord import Message, ChannelType, PartialMessageable
from nextcord.enums import InteractionContextType

from discordbot.cogs.gen_reply import ask_store
from discordbot.cogs.gen_reply.cog import ReplyGeneratorCogs
from discordbot.cogs.gen_reply.recall import build_recall_context, compartments_for_reading
from discordbot.services.memory.store import DM_COMPARTMENT, GLOBAL_COMPARTMENT, guild_compartment
from discordbot.cogs.gen_reply.surface import INTERACTION_FOLLOWUP_LIMIT, TurnSurface
from discordbot.services.memory.writer import subject_source_line
from discordbot.cogs.gen_reply.ask_store import load_ask_turns, record_ask_turn
from discordbot.cogs.gen_reply.streaming import TRUNCATED_NOTICE, ResponseStreamer
from discordbot.cogs.gen_reply.ask_message import (
    build_ask_message,
    interaction_channel,
    rebuild_conversation,
)

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
        self, *, guild_id: int | None = GUILD_ID, context: InteractionContextType | None = None
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
        self.response = SimpleNamespace(defer=self._defer)
        self.client = SimpleNamespace(
            user=SimpleNamespace(id=BOT_USER_ID, name="pocat", discriminator="0")
        )

    async def _defer(self) -> None:
        """Records that the three-second window was answered before any work started."""
        self.deferred = True

    async def edit_original_message(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 -- passthrough Discord payload
        """Records an edit of the deferred response and returns its message."""
        self.edits.append(kwargs)
        return SimpleNamespace(id=1, content=kwargs.get("content"))


async def _echo_prompt(content: str) -> str:
    """Stands in for the mention stripper, which has nothing to strip out of a command option."""
    return content.strip()


def _interaction(**kwargs: Any) -> Any:  # noqa: ANN401 -- the fake stands in for a generic Interaction
    """Builds the fake invocation, untyped so it can stand in for `Interaction[Bot]`."""
    return _FakeAskInteraction(**kwargs)


def _ask_message(*, interaction: Any, question: str = "在幹嘛") -> Message:  # noqa: ANN401 -- see `_interaction`
    """The message the pipeline would answer for this invocation."""
    return build_ask_message(
        interaction=interaction,
        question=question,
        channel=interaction_channel(interaction=interaction),
    )


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
    interaction.data = {
        "resolved": {
            "attachments": {
                "9": {
                    "id": "9",
                    "filename": "cat.png",
                    "size": 1024,
                    "url": "https://cdn.example/cat.png",
                    "proxy_url": "https://cdn.example/cat.png",
                    "content_type": "image/png",
                }
            }
        }
    }

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

    surface = TurnSurface.for_interaction(
        message=_ask_message(interaction=interaction), interaction=interaction
    )

    assert surface.guild_id == expected_guild
    assert surface.is_direct_message is expected_direct


def test_a_group_dm_reads_only_the_cross_server_compartment() -> None:
    """The consequence of the case above, spelled out where the boundary actually is."""
    interaction = _interaction(guild_id=None, context=InteractionContextType.private_channel)
    surface = TurnSurface.for_interaction(
        message=_ask_message(interaction=interaction), interaction=interaction
    )

    context = build_recall_context(
        author_id=ASKER_ID, guild_id=surface.guild_id, is_direct_message=surface.is_direct_message
    )

    assert compartments_for_reading(owner_id=ASKER_ID, context=context) == [GLOBAL_COMPARTMENT]


def test_a_guild_ask_reads_the_compartment_its_own_writes_land_in() -> None:
    """The read and the write must name the same compartment, or memory goes in and never out.

    `Message.guild` is None here whatever the interaction says, so a source stamp taken from
    the message would file this turn's `source_only` observations under `dm/` while the next
    turn in the same server read only `global` and that guild's own.
    """
    interaction = _interaction()
    surface = TurnSurface.for_interaction(
        message=_ask_message(interaction=interaction), interaction=interaction
    )

    context = build_recall_context(
        author_id=ASKER_ID, guild_id=surface.guild_id, is_direct_message=surface.is_direct_message
    )

    assert compartments_for_reading(owner_id=ASKER_ID, context=context) == [
        GLOBAL_COMPARTMENT,
        guild_compartment(guild_id=GUILD_ID),
    ]
    assert subject_source_line(guild_id=surface.guild_id) == f"source: guild {GUILD_ID}"
    assert DM_COMPARTMENT not in compartments_for_reading(owner_id=ASKER_ID, context=context)


def test_a_rebuilt_conversation_gives_the_bot_its_own_turns() -> None:
    """The reply half has to carry the bot's id, or it reaches the model as another user line."""
    interaction = _interaction()

    messages = rebuild_conversation(
        turns=[
            ask_store.AskTurn(message_id=ASK_SNOWFLAKE, question="早", answer="早安"),
            ask_store.AskTurn(message_id=ASK_SNOWFLAKE + 2, question="午安", answer="午安啊"),
        ],
        interaction=interaction,
        bot=interaction.client,
        channel=interaction_channel(interaction=interaction),
    )

    assert [m.content for m in messages] == ["早", "早安", "午安", "午安啊"]
    assert [m.author.id for m in messages] == [ASKER_ID, BOT_USER_ID, ASKER_ID, BOT_USER_ID]
    # Distinct ids, so one turn's log line and attachment-cache key never answer for another's.
    assert len({m.id for m in messages}) == len(messages)


async def test_the_store_replays_a_conversation_oldest_first(ask_isolated_db: None) -> None:
    """What went in comes back in transcript order, scoped to one person in one channel."""
    del ask_isolated_db
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


async def test_the_store_keeps_only_its_retention(
    ask_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversation someone keeps up for a year must not grow the table without limit."""
    del ask_isolated_db
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
    surface = TurnSurface.for_interaction(
        message=_ask_message(interaction=interaction), interaction=interaction
    )
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
    surface = TurnSurface.for_interaction(
        message=_ask_message(interaction=interaction), interaction=interaction
    )

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


def test_an_uncapped_surface_splits_the_whole_answer() -> None:
    """The gateway path has no budget, so nothing about the existing split changes."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    content = "字" * 9000

    parent, chunks = ResponseStreamer._split_reply_for_discord(content=content, footer=footer)

    assert TRUNCATED_NOTICE.strip() not in "".join([parent, *chunks])
    assert "".join([parent, *chunks]) == f"{content}{footer}"


async def test_a_dropped_clip_is_written_where_it_cannot_be_reacted() -> None:
    """The ⏱️ / ⚠️ is the only trace a dropped clip leaves, so it must survive having no message."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(
        message=_ask_message(interaction=interaction), interaction=interaction
    )

    await surface.hint(emoji="⚠️")
    await surface.hint(emoji="⚠️")
    await surface.hint(emoji="⏱️")

    assert surface.take_hints() == ["⚠️", "⏱️"]
    assert surface.take_hints() == []


async def test_the_surface_records_a_turn_and_replays_it_next_time(ask_isolated_db: None) -> None:
    """One turn's answer is the next turn's history, since Discord keeps none of it for us."""
    del ask_isolated_db
    interaction = _interaction()
    first = TurnSurface.for_interaction(
        message=_ask_message(interaction=interaction, question="你叫什麼"), interaction=interaction
    )
    assert await first.fetch_history(limit=500) == []

    await first.record_turn(answer="我叫破貓")

    later = TurnSurface.for_interaction(
        message=_ask_message(interaction=interaction, question="剛剛說了什麼"),
        interaction=interaction,
    )
    history = await later.fetch_history(limit=500)
    assert [message.content for message in history] == ["你叫什麼", "我叫破貓"]
    assert [message.author.id for message in history] == [ASKER_ID, BOT_USER_ID]


async def test_a_gateway_turn_records_nothing(ask_isolated_db: None) -> None:
    """Discord's own channel history is the record there, so the store must stay empty."""
    del ask_isolated_db
    interaction = _interaction()
    surface = TurnSurface.for_message(message=_ask_message(interaction=interaction))

    await surface.record_turn(answer="ignored")

    assert await load_ask_turns(channel_id=CHANNEL_ID, user_id=ASKER_ID, limit=10) == []


async def test_ask_defers_before_anything_slower_than_three_seconds() -> None:
    """The token dies after three seconds, and every phase of a turn is slower than that."""
    cog = ReplyGeneratorCogs.__new__(ReplyGeneratorCogs)
    cog.bot = SimpleNamespace(user=SimpleNamespace(id=BOT_USER_ID, name="pocat"))
    toolkit = SimpleNamespace(input_builder=SimpleNamespace(get_user_prompt=_echo_prompt))
    ran: list[tuple[TurnSurface, str]] = []

    async def _run_turn(*, surface: TurnSurface, user_prompt: str) -> None:
        """Records the turn the command would have run."""
        ran.append((surface, user_prompt))

    cog.toolkit = toolkit
    cog._run_turn = _run_turn
    interaction = _interaction()

    await cog.ask(interaction, question="在幹嘛", attachment=None)

    assert interaction.deferred is True
    surface, user_prompt = ran[0]
    assert user_prompt == "在幹嘛"
    assert surface.interaction is interaction
    assert surface.guild_id == GUILD_ID
    assert surface.message.id == ASK_SNOWFLAKE


async def test_ask_answers_a_blank_question_without_running_a_turn() -> None:
    """A whitespace-only option would otherwise route on nothing at all."""
    cog = ReplyGeneratorCogs.__new__(ReplyGeneratorCogs)
    cog.bot = SimpleNamespace(user=SimpleNamespace(id=BOT_USER_ID, name="pocat"))
    toolkit = SimpleNamespace(input_builder=SimpleNamespace(get_user_prompt=_echo_prompt))

    async def _never(*, surface: TurnSurface, user_prompt: str) -> None:
        """Fails if a blank question ever reaches the pipeline."""
        del surface, user_prompt
        raise AssertionError("a blank question must not run a turn")

    cog.toolkit = toolkit
    cog._run_turn = _never
    interaction = _interaction()

    await cog.ask(interaction, question="   ", attachment=None)

    assert [edit["content"] for edit in interaction.edits] == ["?"]


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
    surface = TurnSurface.for_interaction(
        message=_ask_message(interaction=interaction), interaction=interaction
    )
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    reply = _EditableReply()
    streamer = ResponseStreamer(message=surface.message, surface=surface, reply=reply)
    streamer.stored_content = f"答案{footer}"
    streamer._usage_footer = footer
    await surface.hint(emoji="⚠️")

    await streamer._write_hint_line()

    assert reply.contents == [f"答案\n-# ⚠️{footer}"]
    assert streamer._without_added_lines(text=streamer.stored_content) == f"答案{footer}"


async def test_a_capped_surface_stops_chunking_where_its_budget_ends() -> None:
    """The budget arithmetic, not just the split: five follow-ups and no sixth attempt."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(
        message=_ask_message(interaction=interaction), interaction=interaction
    )
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    reply = _EditableReply()
    streamer = ResponseStreamer(message=surface.message, surface=surface, reply=reply)

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
