"""Tests for the message `/ask` synthesizes from its interaction."""

from discordbot.cogs.gen_reply import ask_store
from discordbot.cogs.gen_reply.ask_message import rebuild_conversation

from tests.helpers.ask_interaction import (
    ASKER_ID,
    _CAT_PNG,
    CHANNEL_ID,
    BOT_USER_ID,
    ASK_SNOWFLAKE,
    _ask_message,
    _interaction,
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
    interaction.data = {"resolved": {"attachments": {"9": _CAT_PNG}}}

    message = _ask_message(interaction=interaction)

    assert [attachment.filename for attachment in message.attachments] == ["cat.png"]


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
