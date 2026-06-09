"""Streams a Responses API reply onto a Discord message."""

import re
from typing import Protocol

from openai import AsyncOpenAI, AsyncStream
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response import Response
from openai.types.responses.tool_param import ToolParam
from openai.types.shared_params.reasoning import Reasoning
from openai.types.responses.response_input_param import FunctionCallOutput, ResponseInputParam
from openai.types.responses.response_function_tool_call_param import ResponseFunctionToolCallParam

from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.economy import CHAT_REWARD_MAX_PER_REPLY, CHAT_REWARD_TOKEN_DIVISOR
from discordbot.utils.reactions import update_reaction
from discordbot.utils.model_pricing import get_token_rates
from discordbot.cogs._economy.database import credit_with_repayment
from discordbot.cogs._economy.presentation import currency_text

# Gemini occasionally wraps Discord mention syntax in backticks (inline code),
# which stops Discord from rendering the actual mention. Strip those wrappers
# before sending; matches user (<@id>, <@!id>), role (<@&id>) and channel (<#id>) mentions.
CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")
DISCORD_MESSAGE_LIMIT = 2000
MAX_TOOL_ROUNDS = 3
MAX_TOOL_CALLS = 8


class ResponseUsageSummary(BaseModel):
    """Aggregated token usage across one visible reply pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reward_input_tokens: int = 0
    reward_output_tokens: int = 0
    cost: float = 0.0
    used_web_search: bool = False

    def add_response(self, response: Response, count_reward_tokens: bool = True) -> None:
        """Accumulates usage from one completed Responses API response."""
        self.model_name = response.model or self.model_name
        if response.usage is None:
            return
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_rate, output_rate = get_token_rates(model_name=response.model)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if count_reward_tokens:
            self.reward_input_tokens += input_tokens
            self.reward_output_tokens += output_tokens
        self.cost += input_rate * input_tokens + output_rate * output_tokens

    def merge(self, other: "ResponseUsageSummary") -> None:
        """Adds another summary into this one."""
        self.model_name = other.model_name or self.model_name
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.reward_input_tokens += other.reward_input_tokens
        self.reward_output_tokens += other.reward_output_tokens
        self.cost += other.cost
        self.used_web_search = self.used_web_search or other.used_web_search


class ResponseToolCall(BaseModel):
    """A completed function call emitted by a streamed Responses API round."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: str
    call_id: str
    item_id: str | None = None

    def to_input_item(self) -> ResponseFunctionToolCallParam:
        """Converts the captured call into a Responses input item."""
        item = ResponseFunctionToolCallParam(
            type="function_call", name=self.name, arguments=self.arguments, call_id=self.call_id
        )
        if self.item_id is not None:
            item["id"] = self.item_id
        item["status"] = "completed"
        return item


class HiddenStreamResult(BaseModel):
    """Collected output from one non-visible streamed tool round."""

    usage: ResponseUsageSummary = Field(default_factory=ResponseUsageSummary)
    function_calls: list[ResponseToolCall] = Field(default_factory=list)


class ToolExecutor(Protocol):
    """Callable that executes a captured Responses API function call."""

    def __call__(self, call: ResponseToolCall) -> str:
        """Returns the function_call_output string for a call."""


class ResponseStreamer(BaseModel):
    """Renders a streaming Responses API reply onto the originating message.

    Attributes:
        message: The Discord message being answered and replied to.
        responses: The streaming Responses API events to render.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message]
    responses: SkipValidation[AsyncStream[ResponseStreamEvent]]
    prior_usage: ResponseUsageSummary = Field(default_factory=ResponseUsageSummary)

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

    async def _usage_footer(self, usage: ResponseUsageSummary) -> str:
        """Credits chat reward and formats the usage footer."""
        total_tokens = usage.reward_input_tokens + usage.reward_output_tokens
        reward = min(total_tokens // CHAT_REWARD_TOKEN_DIVISOR, CHAT_REWARD_MAX_PER_REPLY)
        avatar_url = await guild_avatar_url(
            user=self.message.author, guild=getattr(self.message, "guild", None)
        )
        result = await credit_with_repayment(
            user_id=self.message.author.id,
            name=self.message.author.name,
            avatar_url=avatar_url,
            amount=reward,
        )

        if result.new_balance is not None:
            balance_text = f"{currency_text(amount=result.new_balance, compact=True)} ({currency_text(amount=reward, signed=True, compact=True)})"
        else:
            balance_text = currency_text(amount=reward, signed=True, compact=True)
        # Footer format must stay matchable by `input.USAGE_FOOTER_RE`; the ⬆/⬇ icons are its anchor.
        return f"\n\n-# {usage.model_name} · ⬆ {usage.input_tokens:,} ⬇ {usage.output_tokens:,} · ${usage.cost:.8f} · {balance_text}"

    async def _complete_reply(
        self, reply: Message | None, content: str, usage: ResponseUsageSummary
    ) -> str:
        """Writes final content, footer, and side effects."""
        stored_content = CODED_MENTION_RE.sub(r"\1", content)
        usage_footer = await self._usage_footer(usage=usage)
        await self._finalize(reply=reply, content=stored_content, footer=usage_footer)
        stored_content += usage_footer
        if usage.used_web_search:
            await update_reaction(message=self.message, bot_user=None, emoji="🌐")
        return stored_content

    async def stream(self) -> str:
        """Renders the streamed reply and returns the full content with usage footer."""
        stored_content = ""
        counted_content = 0
        reply: Message | None = None
        displayed_content = ""
        content_started = False
        usage = self.prior_usage.model_copy(deep=True)

        async for response in self.responses:
            if response.type == "response.created":
                # Capture the model on `created` too so the usage footer never
                # falls back to an empty model name (and $0.00000000) when a
                # stream ends without a clean `completed` event. Usage only
                # arrives on `completed`.
                usage.model_name = response.response.model or usage.model_name
            elif response.type == "response.completed":
                usage.add_response(response=response.response)
            elif response.type in {
                "response.web_search_call.in_progress",
                "response.web_search_call.searching",
                "response.web_search_call.completed",
                "response.output_text.annotation.added",
            }:
                usage.used_web_search = True
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

        return await self._complete_reply(reply=reply, content=stored_content, usage=usage)


async def collect_hidden_response_stream(
    responses: AsyncStream[ResponseStreamEvent],
) -> HiddenStreamResult:
    """Consumes a streamed Responses API round without writing text to Discord."""
    result = HiddenStreamResult()
    async for response in responses:
        if response.type == "response.created":
            result.usage.model_name = response.response.model or result.usage.model_name
        elif response.type == "response.completed":
            result.usage.add_response(response=response.response, count_reward_tokens=False)
        elif response.type in {
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.output_text.annotation.added",
        }:
            result.usage.used_web_search = True
        elif response.type == "response.output_item.done":
            item = response.item
            if item.type != "function_call":
                continue
            result.function_calls.append(
                ResponseToolCall(
                    name=item.name, arguments=item.arguments, call_id=item.call_id, item_id=item.id
                )
            )
    return result


async def stream_response_with_tool_loop(  # noqa: PLR0913 -- mirrors the Responses API request surface
    client: AsyncOpenAI,
    message: Message,
    model: str,
    instructions: str,
    input_items: ResponseInputParam,
    reasoning: Reasoning,
    tool_loop_tools: list[ToolParam],
    final_tools: list[ToolParam],
    tool_executor: ToolExecutor,
    extra_headers: dict[str, str],
    extra_body: dict[str, bool],
) -> str:
    """Runs hidden function-call rounds, then streams the final visible reply."""
    conversation = list(input_items)
    usage = ResponseUsageSummary()
    tool_call_count = 0

    for _ in range(MAX_TOOL_ROUNDS):
        responses = await client.responses.create(
            model=model,
            instructions=instructions,
            input=list(conversation),
            reasoning=reasoning,
            tools=tool_loop_tools,
            stream=True,
            service_tier="auto",
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        hidden = await collect_hidden_response_stream(responses=responses)
        usage.merge(other=hidden.usage)
        if not hidden.function_calls:
            break

        for call in hidden.function_calls:
            if tool_call_count >= MAX_TOOL_CALLS:
                break
            conversation.append(call.to_input_item())
            conversation.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=call.call_id,
                    output=tool_executor(call=call),
                )
            )
            tool_call_count += 1
        if tool_call_count >= MAX_TOOL_CALLS:
            break

    final_responses = await client.responses.create(
        model=model,
        instructions=instructions,
        input=list(conversation),
        reasoning=reasoning,
        tools=final_tools,
        stream=True,
        service_tier="auto",
        extra_headers=extra_headers,
        extra_body=extra_body,
    )
    return await ResponseStreamer(
        message=message, responses=final_responses, prior_usage=usage
    ).stream()
