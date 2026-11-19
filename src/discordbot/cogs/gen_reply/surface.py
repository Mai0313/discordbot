"""Discord surface abstraction separating gateway message replies from `/ask` interactions.

`TurnSurface` encapsulates message sending, overflow follow-ups, history loading, reactions, and
guild/DM identity across both paths. Progress reactions are no-ops on `/ask`; `hint` queues
dropped-media emojis for `take_hints` instead. `send` and `send_unparented` stay separate so
callers can rebuild single-use `nextcord.File` payloads before retrying unparented.
"""

from typing import Any

import logfire
from nextcord import File, Embed, Message, DMChannel, ClientUser, Interaction, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.ext import commands
from nextcord.enums import InteractionContextType
from nextcord.utils import utcnow

from discordbot.utils.reactions import ReactionStatusChain, update_reaction
from discordbot.typings.timeouts import INTERACTION_DELIVERY_MARGIN_SECONDS
from discordbot.cogs.gen_reply.ask_store import load_ask_turns, record_ask_turn
from discordbot.cogs.gen_reply.ask_message import build_ask_message, rebuild_conversation

# Discord caps user-installed app follow-up POSTs at 5 per interaction outside bot guilds;
# editing the original response or existing follow-ups is uncapped.
INTERACTION_FOLLOWUP_LIMIT = 5

# Explicit message for /ask TimeoutError so extract_friendly_error avoids an empty code block.
WINDOW_EXPIRED_NOTICE = (
    "This took longer than Discord keeps a slash command open, "
    "so there was nowhere left to deliver it."
)


def _payload(
    content: str | None,
    embed: Embed | None,
    file: File | None,
    files: list[File] | None,
    allowed_mentions: AllowedMentions | None,
) -> dict[str, Any]:
    """Drops every unset argument, so nothing hands Discord a None it reads as "clear this"."""
    given = {
        "content": content,
        "embed": embed,
        "file": file,
        "files": files,
        "allowed_mentions": allowed_mentions,
    }
    return {key: value for key, value in given.items() if value is not None}


class TurnSurface(BaseModel):
    """One turn's Discord surface: how it answers, what it can read, and where it is."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message] = Field(
        ...,
        description=(
            "The turn's source message, real on the gateway path and synthesized on `/ask`."
        ),
    )
    interaction: SkipValidation[Interaction[commands.Bot] | None] = Field(
        default=None, description="The `/ask` invocation, or None for a gateway message."
    )
    guild_id: int | None = Field(
        default=None,
        description=(
            "The guild this conversation is happening in, even where `message.guild` is None."
        ),
    )
    is_direct_message: bool = Field(
        default=False, description="Whether this is a 1:1 DM between the author and the bot."
    )

    # First /ask send edits the deferred response; later sends consume follow-up slots.
    _original_consumed: bool = PrivateAttr(default=False)
    _followups_spent: int = PrivateAttr(default=0)
    _hints: list[str] = PrivateAttr(default_factory=list)

    def answer_capacity(self, has_landed_reply: bool) -> int | None:
        """Returns how many messages one answer may occupy on this surface, or None when unbounded.

        Counts the remaining interaction follow-up POSTs plus one for the original response when it
        has not been consumed yet or when `has_landed_reply` is already occupying it.

        Args:
            has_landed_reply (bool): Whether the current answer has already created its parent
                reply message.
        """
        if self.interaction is None:
            return None
        remaining = max(0, INTERACTION_FOLLOWUP_LIMIT - self._followups_spent)
        if has_landed_reply or not self._original_consumed:
            return remaining + 1
        return remaining

    def delivery_budget_seconds(self) -> float | None:
        """Returns seconds remaining to produce a deliverable before the surface expires, or None.

        Returns None on the gateway path. On `/ask`, returns time until `interaction.expires_at`
        minus `INTERACTION_DELIVERY_MARGIN_SECONDS`, clamped to `0.0`. Assumes the interaction is
        deferred.
        """
        if self.interaction is None:
            return None
        remaining = (self.interaction.expires_at - utcnow()).total_seconds()
        return max(0.0, remaining - INTERACTION_DELIVERY_MARGIN_SECONDS)

    @classmethod
    def for_message(cls, message: Message) -> "TurnSurface":
        """Returns a gateway `TurnSurface` for `message`.

        Args:
            message (Message): Triggering gateway message.
        """
        return cls(
            message=message,
            guild_id=message.guild.id if message.guild else None,
            is_direct_message=message.guild is None and isinstance(message.channel, DMChannel),
        )

    @classmethod
    def for_interaction(
        cls, interaction: Interaction[commands.Bot], question: str
    ) -> "TurnSurface":
        """Returns an `/ask` `TurnSurface` backed by `interaction` and a synthesized source message.

        Only `InteractionContextType.bot_dm` counts as a 1:1 DM; `private_channel` also covers
        group DMs and other users' DMs, which share the same `PartialMessageable` channel type.

        Args:
            interaction (Interaction[commands.Bot]): Deferred `/ask` slash-command interaction.
            question (str): User prompt text passed to `/ask`.

        Raises:
            RuntimeError: `interaction` has no channel or no user.
        """
        return cls(
            message=build_ask_message(interaction=interaction, question=question),
            interaction=interaction,
            guild_id=interaction.guild_id,
            is_direct_message=interaction.context is InteractionContextType.bot_dm,
        )

    async def send(
        self,
        content: str | None = None,
        embed: Embed | None = None,
        file: File | None = None,
        files: list[File] | None = None,
        allowed_mentions: AllowedMentions | None = None,
    ) -> Message:
        """Returns the sent message, replying to `message` on the gateway or editing/following up on `/ask`.

        Args:
            content (str | None): Message text content.
            embed (Embed | None): Single embed to attach.
            file (File | None): Single file attachment.
            files (list[File] | None): Multiple file attachments.
            allowed_mentions (AllowedMentions | None): Mention policy for the send.
        """
        payload = _payload(
            content=content, embed=embed, file=file, files=files, allowed_mentions=allowed_mentions
        )
        if self.interaction is None:
            return await self.message.reply(**payload)
        if not self._original_consumed:
            self._original_consumed = True
            return await self.interaction.edit_original_message(**payload)
        self._followups_spent += 1
        return await self.interaction.followup.send(**payload, wait=True)

    async def send_unparented(
        self,
        content: str | None = None,
        embed: Embed | None = None,
        file: File | None = None,
        files: list[File] | None = None,
        allowed_mentions: AllowedMentions | None = None,
    ) -> Message:
        """Returns a message sent to the channel without a reply reference when the source was deleted.

        On `/ask`, delegates to `send`.

        Args:
            content (str | None): Message text content.
            embed (Embed | None): Single embed to attach.
            file (File | None): Single file attachment.
            files (list[File] | None): Multiple file attachments.
            allowed_mentions (AllowedMentions | None): Mention policy for the send.
        """
        payload = _payload(
            content=content, embed=embed, file=file, files=files, allowed_mentions=allowed_mentions
        )
        if self.interaction is not None:
            return await self.send(**payload)
        return await self.message.channel.send(**payload)

    async def follow_up(
        self, previous: Message, content: str, allowed_mentions: AllowedMentions | None = None
    ) -> Message:
        """Returns the message sent as the next overflow chunk after `previous`.

        Replies to `previous` on the gateway path, and sends an interaction follow-up on `/ask`
        where `previous.channel` is a `PartialMessageable` the bot cannot post into directly.

        Args:
            previous (Message): Preceding chunk's message.
            content (str): Text content for this overflow chunk.
            allowed_mentions (AllowedMentions | None): Mention policy for the send.
        """
        payload = _payload(
            content=content, embed=None, file=None, files=None, allowed_mentions=allowed_mentions
        )
        if self.interaction is None:
            return await previous.reply(**payload)
        self._followups_spent += 1
        return await self.interaction.followup.send(**payload, wait=True)

    async def fetch_history(self, limit: int) -> list[Message]:
        """Returns up to `limit` conversation messages before this turn, oldest first.

        Reads channel history on the gateway path, and reconstructs stored `/ask` turns for the
        current `(channel_id, user_id)` pair on the interaction path.

        Args:
            limit (int): Maximum number of historical messages to return.
        """
        if self.interaction is None:
            # Fetch newest first and reverse: oldest_first=True reverses each 100-message page.
            history = [
                m async for m in self.message.channel.history(limit=limit, before=self.message)
            ]
            history.reverse()
            return history
        turns = await load_ask_turns(
            channel_id=self.message.channel.id, user_id=self.message.author.id, limit=limit // 2
        )
        return rebuild_conversation(turns=turns, interaction=self.interaction)

    def status_chain(self, bot_user: ClientUser | None) -> ReactionStatusChain:
        """Returns the turn's status-reaction chain, disabled when running on `/ask`.

        Args:
            bot_user (ClientUser | None): The bot's own user for removing prior status reactions.
        """
        return ReactionStatusChain(
            message=self.message, bot_user=bot_user, enabled=self.interaction is None
        )

    async def mark(self, emoji: str) -> None:
        """Adds a status or provenance reaction `emoji` to `message` on the gateway path.

        No-op on `/ask`, where the synthesized message does not exist on Discord.

        Args:
            emoji (str): Reaction emoji to add.
        """
        if self.interaction is not None:
            return
        await update_reaction(message=self.message, bot_user=None, emoji=emoji)

    async def hint(self, emoji: str) -> None:
        """Records a best-effort dropped-media hint `emoji` as a reaction or queued `/ask` subtext.

        Reacts on `message` on the gateway path, and queues `emoji` for `take_hints` on `/ask`.

        Args:
            emoji (str): Hint emoji to record.
        """
        if self.interaction is None:
            await update_reaction(message=self.message, bot_user=None, emoji=emoji)
            return
        if emoji not in self._hints:
            self._hints.append(emoji)

    def take_hints(self) -> list[str]:
        """Returns and clears the queued `/ask` failure-hint emojis."""
        hints = list(self._hints)
        self._hints.clear()
        return hints

    async def record_turn(self, answer: str) -> None:
        """Persists the question and `answer` to the `/ask` store, or no-ops on the gateway path.

        Best-effort: logs and swallows database errors.

        Args:
            answer (str): Finalized reply text to store for future `/ask` turns.
        """
        if self.interaction is None:
            return
        try:
            await record_ask_turn(
                channel_id=self.message.channel.id,
                user_id=self.message.author.id,
                message_id=self.message.id,
                question=self.message.content,
                answer=answer,
            )
        except Exception as exc:
            # Broad: best-effort history write must never fail a delivered /ask reply.
            logfire.warn(
                "Failed to record the /ask turn; the next one starts without it",
                message_id=self.message.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
