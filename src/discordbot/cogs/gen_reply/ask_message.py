"""Turning a `/ask` interaction, and the conversation before it, into `Message` objects.

A user-installed app is handed an `Interaction` where the gateway path is handed a `Message`, and
the reply pipeline is written entirely around the latter. Rather than widen every signature below
it, `/ask` builds the `Message` the pipeline expects: a real `nextcord.Message` over a payload
this module assembles, on the `PartialMessageable` the interaction resolved to.

Two things such a message cannot carry, and both are deliberate rather than missing:

- **No guild.** `Message.__init__` resolves `guild` out of the client's own cache, which misses for
  a server the bot was never added to, so `message.guild` is None however the payload is written.
  Where the conversation is really happening is carried beside the message by `TurnSurface`, and
  everything that must not be wrong about it (the location line at developer authority, the memory
  source stamp, the compartments a read may open) reads it from there.
- **No mentions.** Discord does not resolve user mentions inside a command option, so
  `message.mentions` stays empty and a `<@id>` typed into the question never widens the memory
  allowlist. That errs toward reading less, which is the safe direction.
"""

from typing import TYPE_CHECKING, Any, cast

from nextcord import User, Message, Interaction, PartialMessageable
from nextcord.ext import commands

from discordbot.cogs.gen_reply.ask_store import AskTurn

if TYPE_CHECKING:
    from nextcord.state import ConnectionState
    from nextcord.types.user import User as UserPayload
    from nextcord.types.message import Message as MessagePayload


def _base_payload(*, message_id: int, channel_id: int, content: str) -> dict[str, Any]:
    """The minimum `Message.__init__` reads, with every optional key left out.

    `timestamp` is absent because nextcord never reads it: `Message.created_at` is
    `snowflake_time(self.id)`, which is why the ids below have to be real snowflakes.
    """
    return {
        "id": str(message_id),
        "channel_id": str(channel_id),
        "content": content,
        "edited_timestamp": None,
        "type": 0,
        "pinned": False,
        "mention_everyone": False,
        "tts": False,
        "mentions": [],
        "mention_roles": [],
        "attachments": [],
        "embeds": [],
    }


def _build(
    *, state: "ConnectionState", channel: PartialMessageable, payload: dict[str, Any]
) -> Message:
    """Constructs the message over the connection state the interaction arrived on."""
    return Message(state=state, channel=channel, data=cast("MessagePayload", payload))


def interaction_channel(*, interaction: Interaction[commands.Bot]) -> PartialMessageable:
    """The channel a `/ask` turn happens in, as the partial nextcord could resolve.

    Always partial on the routes this command exists for: a guild the bot is not in resolves no
    channel, and Discord sends no data to complete a DM one. A guild channel the bot CAN see
    resolves to the real object, which is a `Messageable` all the same, so it is narrowed here
    rather than special-cased.
    """
    channel = interaction.channel
    if isinstance(channel, PartialMessageable):
        return channel
    if interaction.channel_id is None:
        raise RuntimeError("The interaction names no channel")
    return PartialMessageable(state=interaction._state, id=interaction.channel_id)  # noqa: SLF001 -- mirrors `Interaction.channel`, which builds its own the same way


def resolved_attachment_payloads(
    *, interaction: Interaction[commands.Bot]
) -> list[dict[str, Any]]:
    """The raw payloads of the attachment options this invocation carried, if any.

    Read back out of `interaction.data` rather than off the `nextcord.Attachment` the option
    binding produced, because `Message.__init__` builds its own `Attachment` objects from the
    payload and there is no round trip from one to the other.
    """
    data = interaction.data or {}
    resolved = cast("dict[str, Any]", data.get("resolved") or {})
    attachments = cast("dict[str, Any]", resolved.get("attachments") or {})
    return list(attachments.values())


def build_ask_message(
    *, interaction: Interaction[commands.Bot], question: str, channel: PartialMessageable
) -> Message:
    """Builds the message the pipeline answers, from one `/ask` invocation.

    The id is the interaction's own snowflake, so `Message.created_at` is the real moment the
    command was run: it feeds `REQUEST_TIME_CONTEXT_PROMPT`, and it is the `message_id` every
    record of this turn is correlated by.

    Args:
        interaction: The invocation being answered.
        question: The `question` option, verbatim.
        channel: The channel resolved by `interaction_channel`.

    Returns:
        A message carrying the question, its attachment, and the invoking user as its author.

    Raises:
        RuntimeError: The interaction names no user, which Discord never sends.
    """
    user = interaction.user
    if user is None:
        raise RuntimeError("The interaction names no user")
    payload = _base_payload(message_id=interaction.id, channel_id=channel.id, content=question)
    payload["attachments"] = resolved_attachment_payloads(interaction=interaction)
    message = _build(
        state=interaction._state,  # noqa: SLF001 -- the connection state the interaction arrived on
        channel=channel,
        payload=payload,
    )
    # Assigned rather than rendered into the payload: `interaction.user` is the real Member
    # nextcord already built, so the server nickname every identity line renders survives.
    message.author = user
    return message


def rebuild_conversation(
    *,
    turns: list[AskTurn],
    interaction: Interaction[commands.Bot],
    bot: commands.Bot,
    channel: PartialMessageable,
) -> list[Message]:
    """Rebuilds stored `/ask` turns as the history messages the renders expect, oldest first.

    Each turn becomes two messages, because that is what the answer model has to see: the
    question authored by the user, and the reply authored by the bot. The bot's own id is what
    makes the second one work — `MessageInputBuilder._assemble_input_message` decides `role` by
    comparing the author against `bot.user`, so a turn rebuilt under any other author would reach
    the model as one more user line with a `Bot (bot) [id: ...]:` prefix, which is exactly the
    shape that render exists to avoid.

    The reply's id is the question's plus one. Nothing resolves either against Discord, and the
    two only ever separate one log line and one attachment-cache key from another; taking the
    next value keeps the reply inside the same millisecond, so its `created_at` still reads as
    the moment it was written.

    Stored turns carry no attachments, so this history is text throughout and the media budget
    has nothing to cap. What someone attached to an earlier `/ask` is gone by the next one.

    Args:
        turns: The stored exchanges, oldest first.
        interaction: The current invocation, for the asker's own author object.
        bot: The bot, for its connection state and its own user.
        channel: The channel resolved by `interaction_channel`.

    Returns:
        The rebuilt messages in transcript order.
    """
    user = interaction.user
    bot_user = bot.user
    if user is None or bot_user is None:
        return []
    # A `User` rather than `bot.user` itself, which is a `ClientUser` and so not one of the two
    # types `Message.author` may hold. Only the id is ever read off it here.
    state = interaction._state  # noqa: SLF001 -- the connection state the interaction arrived on
    bot_author = User(
        state=state,
        data=cast(
            "UserPayload",
            {
                "id": str(bot_user.id),
                "username": bot_user.name,
                "discriminator": bot_user.discriminator,
                "avatar": None,
                "bot": True,
            },
        ),
    )
    messages: list[Message] = []
    for turn in turns:
        question = _build(
            state=state,
            channel=channel,
            payload=_base_payload(
                message_id=turn.message_id, channel_id=channel.id, content=turn.question
            ),
        )
        question.author = user
        answer = _build(
            state=state,
            channel=channel,
            payload=_base_payload(
                message_id=turn.message_id + 1, channel_id=channel.id, content=turn.answer
            ),
        )
        answer.author = bot_author
        messages.extend((question, answer))
    return messages
