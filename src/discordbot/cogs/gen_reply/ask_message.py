"""Turning a `/ask` interaction, and the conversation before it, into `Message` objects.

A user-installed app is handed an `Interaction` where the gateway path is handed a `Message`, and
the reply pipeline is written entirely around the latter. Rather than widen every signature below
it, `/ask` builds the `Message` the pipeline expects: a real `nextcord.Message` over a payload
this module assembles, on the `PartialMessageable` the interaction resolved to.

Two things such a message cannot carry, and both are deliberate rather than missing:

- **No guild.** The payload names no `guild_id` and a `PartialMessageable` carries no guild of its
  own, so `message.guild` is None in every server, one the bot is in included. Where the
  conversation is really happening is carried beside the message by `TurnSurface`, and
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
    from nextcord.types.message import Message as MessagePayload


def _base_payload(message_id: int, channel_id: int, content: str) -> dict[str, Any]:
    """Returns the minimal payload dict `Message.__init__` reads, with optional keys omitted.

    `Message.created_at` derives from `snowflake_time(self.id)`, so `message_id` must be a real
    snowflake.
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
    state: "ConnectionState", channel: PartialMessageable, payload: dict[str, Any]
) -> Message:
    """Constructs the message over the connection state the interaction arrived on."""
    return Message(state=state, channel=channel, data=cast("MessagePayload", payload))


def interaction_channel(interaction: Interaction[commands.Bot]) -> PartialMessageable:
    """Returns the `PartialMessageable` channel for a `/ask` turn.

    Args:
        interaction (Interaction[commands.Bot]): Invocation whose channel is resolved.

    Raises:
        RuntimeError: The interaction names no channel.
    """
    channel = interaction.channel
    if isinstance(channel, PartialMessageable):
        return channel
    if interaction.channel_id is None:
        raise RuntimeError("The interaction names no channel")
    return PartialMessageable(state=interaction._state, id=interaction.channel_id)  # noqa: SLF001 -- mirrors `Interaction.channel`, which builds its own the same way


def resolved_attachment_payloads(interaction: Interaction[commands.Bot]) -> list[dict[str, Any]]:
    """Returns the raw attachment payloads carried by `interaction.data`, if any.

    Read from `interaction.data` rather than the bound `nextcord.Attachment` because
    `Message.__init__` constructs its own `Attachment` instances from raw payloads.

    Args:
        interaction (Interaction[commands.Bot]): Invocation to inspect for resolved attachments.
    """
    data = interaction.data or {}
    resolved = cast("dict[str, Any]", data.get("resolved") or {})
    attachments = cast("dict[str, Any]", resolved.get("attachments") or {})
    return list(attachments.values())


def build_ask_message(interaction: Interaction[commands.Bot], question: str) -> Message:
    """Returns the synthesized `Message` the pipeline answers for one `/ask` invocation.

    Reuses the interaction's snowflake as `message.id` so `Message.created_at` reflects when the
    command ran and anchors every log record for the turn.

    Args:
        interaction (Interaction[commands.Bot]): The invocation being answered.
        question (str): The `question` option, verbatim.

    Raises:
        RuntimeError: The interaction names no channel or no user.
    """
    channel = interaction_channel(interaction=interaction)
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
    turns: list[AskTurn], interaction: Interaction[commands.Bot]
) -> list[Message]:
    """Returns stored `/ask` turns rebuilt as transcript `Message` objects, oldest first.

    Each turn expands into a user question message and a bot answer message (with id
    `question.id + 1` and the bot's user id so input rendering assigns `role="assistant"`).
    Stored turns carry text only, without attachments.

    Args:
        turns (list[AskTurn]): The stored exchanges, oldest first.
        interaction (Interaction[commands.Bot]): Current invocation supplying the user, bot user,
            and channel.

    Raises:
        RuntimeError: The interaction names no channel.
    """
    channel = interaction_channel(interaction=interaction)
    user = interaction.user
    bot_user = interaction.client.user
    if user is None or bot_user is None:
        return []
    # A `User` rather than `bot_user` itself, which is a `ClientUser` and so not one of the two
    # types `Message.author` may hold. Only the id is ever read off it here.
    state = interaction._state  # noqa: SLF001 -- the connection state the interaction arrived on
    bot_author = User(
        state=state,
        data={
            "id": str(bot_user.id),
            "username": bot_user.name,
            "discriminator": bot_user.discriminator,
            "avatar": None,
            "bot": True,
        },
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
