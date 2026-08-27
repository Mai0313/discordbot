"""Where one turn's messages go, and what conversation it continues.

Every reply the pipeline writes used to be a reply to a message in a channel the bot is a member
of. `/ask` has neither half: a user-installed app is not in the channel it was invoked in, so it
cannot send there, cannot read the history, and cannot react. All it holds is an interaction
token, which buys unlimited edits of one original response plus a small number of follow-ups.

`TurnSurface` is that difference, and only that difference. It rides beside `message` through the
classes that already hold one, and `for_message` reproduces today's gateway behaviour exactly, so
nothing on the `on_message` path changes shape.

Three notes on what it deliberately does not paper over:

- **Progress signals are dropped rather than replaced.** `update_reaction` already suppresses its
  own failures, so the pipeline's status chain and the per-marker "working on it" emoji are silent
  no-ops on the interaction path — which is right, because Discord shows its own thinking state
  while the response is deferred and the reply then streams in place. Only the best-effort FAILURE
  hints (a dropped clip's ⏱️ / ⚠️) have nothing left to say them, which is what `hint` collects
  and the streamer writes onto the reply as one line.
- **`send` and `send_unparented` are two methods, not one with a fallback.** A `nextcord.File` is
  read once, so a caller that wants a second attempt has to rebuild its payload first; the failure
  notice in `cog.py` already does exactly that with its embed spacer.
- **`guild_id` is carried rather than read.** `Message.guild` resolves out of the client's own
  cache and misses for a server the bot was never added to, so the synthesized message's is always
  None. Everything that must not be wrong about where a turn happened — the location line at
  developer authority, the memory source stamp, the compartments a memory read may open — takes it
  from here instead.
"""

from typing import Any

import logfire
from nextcord import File, Embed, Message, DMChannel, ClientUser, Interaction, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.ext import commands
from nextcord.enums import InteractionContextType

from discordbot.utils.reactions import update_reaction
from discordbot.cogs.gen_reply.ask_store import load_ask_turns, record_ask_turn
from discordbot.cogs.gen_reply.ask_message import interaction_channel, rebuild_conversation

# How many follow-up messages Discord lets a user-installed app POST per interaction while it is
# not a member of the server (`interactions/receiving-and-responding.mdx:474`, read 2026-08-26).
# The cap is on creating messages only: editing the original response, or an existing follow-up,
# is documented nowhere as bounded and is what the whole streaming preview rides on.
INTERACTION_FOLLOWUP_LIMIT = 5


def _payload(
    *,
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
    """One turn's Discord surface: how it answers, what it can read, and where it is.

    Attributes:
        message: The turn's source message, real on the gateway path and synthesized on `/ask`.
        interaction: The `/ask` invocation, or None when the turn came off the gateway.
        guild_id: The guild this conversation is happening in, even where `message.guild` is None.
        is_direct_message: Whether this is a 1:1 DM between the author and the bot.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message] = Field(..., description="The turn's source message.")
    interaction: SkipValidation[Interaction[commands.Bot] | None] = Field(
        default=None, description="The `/ask` invocation, or None for a gateway message."
    )
    guild_id: int | None = Field(
        default=None, description="The guild this conversation is happening in."
    )
    is_direct_message: bool = Field(
        default=False, description="Whether this is a 1:1 DM between the author and the bot."
    )

    # Discord treats the deferred response as already sent, so the first send edits it into place
    # and only later ones spend the follow-up budget.
    _original_consumed: bool = PrivateAttr(default=False)
    _followups_spent: int = PrivateAttr(default=0)
    # The failure hints reactions would have carried, in the order they happened.
    _hints: list[str] = PrivateAttr(default_factory=list)

    def answer_capacity(self, *, has_landed_reply: bool) -> int | None:
        """How many messages one answer may occupy here, or None when nothing caps it.

        The follow-up POSTs Discord still allows, plus one for the message the answer is written
        onto — the original response whether or not it has been spent, since editing it is free.
        Asked as one question rather than answered by the caller adding its own message on:
        a landed reply always means the original was spent, so counting both separately would
        hand the answer one message more than it can have.
        """
        if self.interaction is None:
            return None
        remaining = max(0, INTERACTION_FOLLOWUP_LIMIT - self._followups_spent)
        if has_landed_reply or not self._original_consumed:
            return remaining + 1
        return remaining

    @classmethod
    def for_message(cls, *, message: Message) -> "TurnSurface":
        """The gateway surface: reply into the channel, read its history, react on the message."""
        return cls(
            message=message,
            guild_id=message.guild.id if message.guild else None,
            is_direct_message=message.guild is None and isinstance(message.channel, DMChannel),
        )

    @classmethod
    def for_interaction(
        cls, *, message: Message, interaction: Interaction[commands.Bot]
    ) -> "TurnSurface":
        """The `/ask` surface: answer through the interaction, read the conversation store.

        `bot_dm` is the one interaction context that is a 1:1 DM with the bot; `private_channel`
        covers both a group DM and a DM between two other people, neither of which is one, and
        the channel object cannot tell them apart (it is a `PartialMessageable` for all three).
        """
        return cls(
            message=message,
            interaction=interaction,
            guild_id=interaction.guild_id,
            is_direct_message=interaction.context is InteractionContextType.bot_dm,
        )

    async def send(
        self,
        *,
        content: str | None = None,
        embed: Embed | None = None,
        file: File | None = None,
        files: list[File] | None = None,
        allowed_mentions: AllowedMentions | None = None,
    ) -> Message:
        """Puts a new message on screen, parented to the source message where there is one.

        Returns:
            The message that landed, editable by every caller that keeps the handle.
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
        *,
        content: str | None = None,
        embed: Embed | None = None,
        file: File | None = None,
        files: list[File] | None = None,
        allowed_mentions: AllowedMentions | None = None,
    ) -> Message:
        """Puts a message on screen without parenting it, for a source that was deleted.

        Identical to `send` on the interaction path, which has no parent to lose in the first
        place; the caller reaches it only after `send` raised, and the retry is what matters.

        Returns:
            The message that landed.
        """
        if self.interaction is not None:
            return await self.send(
                content=content,
                embed=embed,
                file=file,
                files=files,
                allowed_mentions=allowed_mentions,
            )
        return await self.message.channel.send(
            **_payload(
                content=content,
                embed=embed,
                file=file,
                files=files,
                allowed_mentions=allowed_mentions,
            )
        )

    async def follow_up(
        self, *, previous: Message, content: str, allowed_mentions: AllowedMentions | None = None
    ) -> Message:
        """Continues past what one Discord message can hold.

        `previous.reply` is what the gateway path wants and what the interaction path must never
        do: a follow-up carries a `PartialMessageable` channel, so replying to it would send a
        plain bot message into a channel the bot is not in, and every answer over the message
        limit would lose its tail.

        Returns:
            The message carrying this chunk.
        """
        if self.interaction is None:
            return await previous.reply(
                **_payload(
                    content=content,
                    embed=None,
                    file=None,
                    files=None,
                    allowed_mentions=allowed_mentions,
                )
            )
        self._followups_spent += 1
        return await self.interaction.followup.send(
            **_payload(
                content=content,
                embed=None,
                file=None,
                files=None,
                allowed_mentions=allowed_mentions,
            ),
            wait=True,
        )

    async def fetch_history(self, *, limit: int) -> list[Message]:
        """The conversation before this turn, oldest first and at most `limit` messages.

        The gateway path walks the channel. `/ask` cannot: the bot is not a member and holds no
        `READ_MESSAGE_HISTORY`, so what it replays is the conversation it kept itself, one stored
        turn rebuilding into the question and the answer that followed it.
        """
        if self.interaction is None:
            return [
                m
                async for m in self.message.channel.history(
                    limit=limit, before=self.message, oldest_first=True
                )
            ]
        user = self.interaction.user
        if user is None or self.interaction.channel_id is None:
            return []
        turns = await load_ask_turns(
            channel_id=self.interaction.channel_id, user_id=user.id, limit=limit // 2
        )
        return rebuild_conversation(
            turns=turns,
            interaction=self.interaction,
            bot=self.interaction.client,
            channel=interaction_channel(interaction=self.interaction),
        )

    async def mark(self, *, emoji: str, bot_user: ClientUser | None = None) -> None:
        """Puts a status or provenance reaction on the source message, where there is one.

        Refused rather than attempted on `/ask`: the synthesized message names nothing Discord
        holds, so every add is a REST round trip that 404s and is then swallowed, and two of
        these are awaited before the answer is even dispatched. What they carried is carried
        there by the deferred response's own thinking state and by the reply streaming in place.
        """
        if self.interaction is not None:
            return
        await update_reaction(message=self.message, bot_user=bot_user, emoji=emoji)

    async def hint(self, *, emoji: str) -> None:
        """Records that something best-effort was dropped, so it is never silent.

        On the gateway path this is the independent reaction it has always been. On `/ask` there
        is nothing to react to, so the emoji is held for `take_hints` and the streamer writes it
        onto the reply instead. This is the one signal that cannot simply be dropped the way
        `mark` drops a progress marker: per the best-effort convention it is the ONLY trace a
        dropped clip, image, song or video ever leaves.
        """
        if self.interaction is None:
            await update_reaction(message=self.message, bot_user=None, emoji=emoji)
            return
        if emoji not in self._hints:
            self._hints.append(emoji)

    def take_hints(self) -> list[str]:
        """Hands over the collected hints and clears them, so one line is written once."""
        hints = list(self._hints)
        self._hints.clear()
        return hints

    async def record_turn(self, *, answer: str) -> None:
        """Appends this exchange to the conversation, so the next `/ask` has one to continue.

        A no-op on the gateway path, where Discord's own channel history is the record.

        Best-effort like everything else that runs after the answer is on screen: the reply is
        delivered by the time this is called, so a busy database must cost the next turn its
        context rather than posting `Something went wrong` under a complete answer and taking
        the turn's memory notes down with it.
        """
        if self.interaction is None:
            return
        user = self.interaction.user
        if user is None or self.interaction.channel_id is None:
            return
        try:
            await record_ask_turn(
                channel_id=self.interaction.channel_id,
                user_id=user.id,
                message_id=self.message.id,
                question=self.message.content,
                answer=answer,
            )
        except Exception as exc:
            # Broad on purpose, per the docstring: nothing above this can recover, and the
            # only cost is that the next `/ask` in this channel starts a turn short.
            logfire.warn(
                "Failed to record the /ask turn; the next one starts without it",
                message_id=self.message.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
