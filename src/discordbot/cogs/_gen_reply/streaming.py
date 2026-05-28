"""Streams a Responses API reply onto a Discord message."""

import re

from openai import AsyncStream
from nextcord import Message
from pydantic import BaseModel, ConfigDict, SkipValidation
from openai.types.responses import ResponseStreamEvent

from discordbot.utils.avatars import guild_avatar_url
from discordbot.utils.reactions import update_reaction
from discordbot.utils.model_pricing import get_token_rates
from discordbot.cogs._economy.database import credit_with_repayment
from discordbot.cogs._economy.presentation import currency_text

# Gemini occasionally wraps Discord mention syntax in backticks (inline code),
# which stops Discord from rendering the actual mention. Strip those wrappers
# before sending; matches user (<@id>, <@!id>), role (<@&id>) and channel (<#id>) mentions.
CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")
DISCORD_MESSAGE_LIMIT = 2000


class ResponseStreamer(BaseModel):
    """Renders a streaming Responses API reply onto the originating message.

    Attributes:
        message: The Discord message being answered and replied to.
        responses: The streaming Responses API events to render.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message]
    responses: SkipValidation[AsyncStream[ResponseStreamEvent]]

    @staticmethod
    def _split_reply_for_discord(content: str, footer: str) -> tuple[str, list[str]]:
        """Splits a completed reply into one parent message plus follow-up chunks."""
        if len(f"{content}{footer}") <= DISCORD_MESSAGE_LIMIT:
            return f"{content}{footer}", []

        tail_capacity = DISCORD_MESSAGE_LIMIT - len(footer)
        if tail_capacity <= 0:
            raise ValueError("Usage footer is too long for Discord message content")

        parent_content = content[:DISCORD_MESSAGE_LIMIT]
        remaining = content[DISCORD_MESSAGE_LIMIT:]
        follow_up_chunks: list[str] = []

        while len(remaining) > DISCORD_MESSAGE_LIMIT:
            follow_up_chunks.append(remaining[:DISCORD_MESSAGE_LIMIT])
            remaining = remaining[DISCORD_MESSAGE_LIMIT:]

        if len(remaining) <= tail_capacity:
            follow_up_chunks.append(f"{remaining}{footer}")
        else:
            follow_up_chunks.append(remaining[:tail_capacity])
            follow_up_chunks.append(f"{remaining[tail_capacity:]}{footer}")
        return parent_content, follow_up_chunks

    async def _write_preview(
        self, reply: Message | None, content: str, displayed_content: str
    ) -> tuple[Message | None, str]:
        """Writes at most one Discord message worth of streaming preview text."""
        preview = content[:DISCORD_MESSAGE_LIMIT]
        if preview == displayed_content:
            return reply, displayed_content
        if reply is None:
            reply = await self.message.reply(content=preview)
        else:
            await reply.edit(content=preview)
        return reply, preview

    async def _finalize(self, reply: Message | None, content: str, footer: str) -> Message:
        """Writes the final reply, continuing overflow as follow-up replies in the same channel."""
        parent_content, follow_up_chunks = self._split_reply_for_discord(
            content=content, footer=footer
        )
        if reply is None:
            reply = await self.message.reply(content=parent_content)
        else:
            await reply.edit(content=parent_content)
        previous = reply
        for chunk in follow_up_chunks:
            previous = await previous.reply(content=chunk)
        return reply

    async def stream(self) -> str:  # noqa: C901 -- dispatches on multiple Responses API stream event types
        """Renders the streamed reply and returns the full content with usage footer."""
        stored_content = ""
        counted_content = 0
        reply: Message | None = None
        displayed_content = ""
        content_started = False
        model_name = ""
        input_tokens = 0
        output_tokens = 0
        used_web_search = False

        async for response in self.responses:
            if response.type == "response.completed":
                model_name = response.response.model
                if response.response.usage:
                    input_tokens = response.response.usage.input_tokens
                    output_tokens = response.response.usage.output_tokens
            elif response.type in {
                "response.web_search_call.in_progress",
                "response.web_search_call.searching",
                "response.web_search_call.completed",
                "response.output_text.annotation.added",
            }:
                used_web_search = True
            elif response.type == "response.output_text.delta":
                delta = response.delta
                if not content_started:
                    delta = delta.lstrip("\n")
                    if not delta:
                        continue
                    content_started = True
                stored_content += delta
                counted_content += len(delta)

                if counted_content >= 30:
                    reply, displayed_content = await self._write_preview(
                        reply=reply, content=stored_content, displayed_content=displayed_content
                    )
                    counted_content = 0

        input_rate, output_rate = get_token_rates(model_name=model_name)
        cost = input_rate * input_tokens + output_rate * output_tokens

        # Award chat points equal to total tokens used. We await this (rather than fire-and-forget)
        # so the resulting balance can land in the footer.
        # On DB failure, it returns None and the footer falls back to the delta-only format.
        total_tokens = input_tokens + output_tokens
        avatar_url = await guild_avatar_url(
            user=self.message.author, guild=getattr(self.message, "guild", None)
        )
        result = await credit_with_repayment(
            user_id=self.message.author.id,
            name=self.message.author.name,
            avatar_url=avatar_url,
            amount=total_tokens,
        )

        stored_content = CODED_MENTION_RE.sub(r"\1", stored_content)
        if result.new_balance is not None:
            balance_text = f"{currency_text(amount=result.new_balance, compact=True)} ({currency_text(amount=total_tokens, signed=True, compact=True)})"
        else:
            balance_text = currency_text(amount=total_tokens, signed=True, compact=True)
        # Footer format must stay matchable by `input.USAGE_FOOTER_RE`; the ⬆/⬇ icons are its anchor.
        usage_footer = f"\n\n-# {model_name} · ⬆ {input_tokens:,} ⬇ {output_tokens:,} · ${cost:.8f} · {balance_text}"

        # Final update to ensure complete message is displayed.
        await self._finalize(reply=reply, content=stored_content, footer=usage_footer)
        stored_content += usage_footer

        if used_web_search:
            await update_reaction(message=self.message, bot_user=None, emoji="🌐")

        return stored_content
