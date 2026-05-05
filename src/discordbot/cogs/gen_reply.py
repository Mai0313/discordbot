from io import BytesIO
import re
import ast
import json
import base64
from typing import TYPE_CHECKING, Literal, cast
import asyncio
from datetime import UTC, datetime
from functools import cached_property
from mimetypes import guess_type
import contextlib

from PIL import Image
from openai import AsyncOpenAI, AsyncStream
import logfire
from nextcord import File, Embed, Message, Attachment, StickerItem
from pydantic import BaseModel, ValidationError
from nextcord.ext import commands
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.tool_param import ToolParam
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.utils.images import get_pil_image, get_image_data, convert_base64_to_data_uri
from discordbot.utils.model_pricing import get_token_rates

from ._gen_reply.prompts import BELIEF, IMAGE_PROMPT, REPLY_PROMPT, ROUTE_PROMPT, SUMMARY_PROMPT

if TYPE_CHECKING:
    from collections.abc import Awaitable

DEFAULT_FAST_MODEL = "gemini-flash-latest"
DEFAULT_SLOW_MODEL = "gemini-pro-latest"
PEAK_SLOW_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_VIDEO_MODEL = "veo-3.1-fast-generate-preview"

# Gemini occasionally wraps Discord mention syntax in backticks (inline code),
# which stops Discord from rendering the actual mention. Strip those wrappers
# before sending; matches user (<@id>, <@!id>), role (<@&id>) and channel (<#id>) mentions.
_CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")

# LiteLLM surfaces upstream provider errors as a chain like
# `litellm.X: litellm.Y: VertexException - b'{"error": {"message": "..."}}'`,
# where the provider's actual JSON body is embedded as a Python bytes literal.
_BYTES_LITERAL_RE = re.compile(pattern=r"b'((?:[^'\\]|\\.)*)'", flags=re.DOTALL)


def _extract_friendly_error(exc: BaseException) -> str:
    """Surface the innermost provider error message from a LiteLLM-wrapped APIError.

    OpenAI's streaming layer constructs `APIError(message=error["message"], ...)`
    from the upstream SSE event; when LiteLLM is the upstream, that `message` is
    the wrapped exception chain with the provider response stuffed inside as a
    `b'...'` Python literal. Walk every embedded bytes literal, parse it as
    JSON, and return `error.message` (or top-level `message`). Fall back to
    `str(exc)` when nothing parses, so we never lose the original signal.
    """
    raw = str(exc)
    for match in _BYTES_LITERAL_RE.finditer(string=raw):
        try:
            decoded = ast.literal_eval(node_or_string=match.group(0)).decode(
                encoding="utf-8", errors="replace"
            )
            data = json.loads(s=decoded)
        except (SyntaxError, ValueError, TypeError, AttributeError):
            continue
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                inner = error.get("message")
                if isinstance(inner, str) and inner:
                    return inner
            top = data.get("message")
            if isinstance(top, str) and top:
                return top
    return raw


def get_tools(model: str) -> list[ToolParam]:
    """Returns the tools available for the specified model.

    Args:
        model: The name of the model.

    Returns:
        A list of tools supported by the model.
    """
    if "gemini" in model:
        return [{"googleSearch": {}}, {"urlContext": {}}]
    if "claude" in model:
        return [
            {"type": "web_search_20260209", "name": "web_search"},
            {"type": "web_fetch_20260209", "name": "web_fetch"},
        ]
    return [{"type": "web_search"}]


class RouteDecision(BaseModel):
    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"]


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the ReplyGeneratorCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The cached AsyncOpenAI client instance."""
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return client

    async def _get_user_prompt(self, content: str) -> str:
        """Removes the bot mention from the content and strips whitespace."""
        if self.bot.user:
            content = content.replace(f"<@{self.bot.user.id}>", "")
        return content.strip()

    async def _get_cleaned_content(self, message: Message) -> str:
        """Gets cleaned message content, including embed details if text is empty, and adds author prefix."""
        content = await self._get_user_prompt(content=message.content)
        if not content and message.embeds:
            embed_parts: list[str] = []
            for embed in message.embeds:
                parts: list[str] = []
                if embed.author and embed.author.name:
                    parts.append(f"Author: {embed.author.name}")
                if embed.title:
                    parts.append(f"Title: {embed.title}")
                if embed.description:
                    parts.append(embed.description)
                for field in embed.fields:
                    parts.append(f"{field.name}: {field.value}")
                if embed.footer and embed.footer.text:
                    parts.append(f"Footer: {embed.footer.text}")
                if parts:
                    embed_parts.append("\n".join(parts))
            content = "\n\n".join(embed_parts)
        # TODO(deferred root-cause fix): weaker models sometimes echo the
        # `display_name (username) [id: USER_ID]: ` prefix at the start of their
        # own reply, because they see the same prefix attached to their previous
        # replies in chat history and imitate the pattern. The thorough fix is
        # to skip the prefix for the bot's own messages — role=assistant already
        # marks them, so the prefix is redundant on those rows. Sketch:
        #     is_bot = bool(self.bot.user and message.author.id == self.bot.user.id)
        #     if is_bot:
        #         return content
        #     content = f"{message.author.display_name} ({message.author.name}) [id: {message.author.id}]: {content}"
        #     return content
        # Currently relying on the prompt-level guard in REPLY_PROMPT /
        # SUMMARY_PROMPT (do-not-reproduce-prefix rule); revisit if that proves
        # insufficient.
        content = f"{message.author.display_name} ({message.author.name}) [id: {message.author.id}]: {content}"
        return content

    async def _image_to_part(
        self, source: Attachment | StickerItem | str
    ) -> ResponseInputImageParam | None:
        """Converts an image source to a content part for the API."""
        url = source if isinstance(source, str) else source.url
        try:
            if isinstance(source, str):
                downloaded = get_pil_image(image_file=source)
            else:
                downloaded = Image.open(BytesIO(await source.read()))
            downloaded.thumbnail(size=(1568, 1568))
            if downloaded.mode != "RGB":
                downloaded = downloaded.convert("RGB")
            buffer = BytesIO()
            downloaded.save(buffer, format="JPEG", quality=85, optimize=True)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            converted = convert_base64_to_data_uri(b64)
            return ResponseInputImageParam(image_url=converted, detail="auto", type="input_image")
        except Exception:
            logfire.warn(f"Failed to convert image, keeping original URL: {url}")
            return None

    async def _attachment_to_part(self, attachment: Attachment) -> ResponseInputFileParam | None:
        """Converts a file attachment to a content part for the API."""
        try:
            file_bytes = await attachment.read()
            b64_data = base64.b64encode(file_bytes).decode()
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            mime_type = content_type.split(";")[0].strip()
            if not mime_type:
                logfire.warn(
                    f"Skipping attachment with unknown MIME type: {attachment.filename} ({attachment.url})"
                )
                return None
            data_uri = f"data:{mime_type};base64,{b64_data}"
            return ResponseInputFileParam(
                filename=attachment.filename, file_data=data_uri, type="input_file"
            )
        except Exception:
            logfire.warn(f"Failed to download attachment: {attachment.url}")
            return None

    async def _get_attachment_parts(
        self, message: Message
    ) -> list[ResponseInputImageParam | ResponseInputFileParam]:
        """Extracts attachment content parts from a message."""
        _content_parts: list[ResponseInputImageParam | ResponseInputFileParam | None] = []

        for attachment in message.attachments:
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            if content_type.startswith("image/"):
                _content_parts.append(await self._image_to_part(source=attachment))
            else:
                # video/*, application/pdf, text/plain, application/json, etc.
                _content_parts.append(await self._attachment_to_part(attachment=attachment))

        for sticker in message.stickers:
            _content_parts.append(await self._image_to_part(source=sticker))

        # Prefer Discord's proxy_url (media.discordapp.net) over the original URL,
        # since sources like Threads CDN expire and reject requests without specific headers.
        for embed in message.embeds:
            if embed.image and (url := embed.image.proxy_url or embed.image.url):
                _content_parts.append(await self._image_to_part(source=url))
            if embed.thumbnail and (url := embed.thumbnail.proxy_url or embed.thumbnail.url):
                _content_parts.append(await self._image_to_part(source=url))

        content_parts: list[ResponseInputImageParam | ResponseInputFileParam] = [
            part for part in _content_parts if part is not None
        ]
        return content_parts

    async def _process_single_message(self, message: Message) -> EasyInputMessageParam:
        """Processes a single Discord message into a Responses API input message."""
        try:
            content = await self._get_cleaned_content(message=message)
            attachment_parts = await self._get_attachment_parts(message=message)
            is_bot = bool(self.bot.user and message.author.id == self.bot.user.id)

            # No attachments → use EasyInputMessageParam's string-content shorthand.
            # The SDK serializes it as `output_text` for role=assistant and as
            # `input_text` for role=user, which satisfies GPT-5.4's strict rule
            # (role=assistant rejects an explicit `type: input_text` content part).
            # This preserves the assistant-role weighting for bot text replies.
            if not attachment_parts:
                return EasyInputMessageParam(
                    role="assistant" if is_bot else "user", content=content
                )

            # Has attachments → must use a content list with input_text/input_image.
            # role=assistant cannot carry `input_image` (only output_text/refusal),
            # so bot replies that include generated images (from _handle_image_reply)
            # fall back to role=user. The author identity prefix already in `content`
            # preserves bot-vs-user distinction for the model.
            return EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(text=content, type="input_text"),
                    *attachment_parts,
                ],
            )
        except Exception as e:
            logfire.warn(f"Failed to process message {message.id}: {e}")
            return EasyInputMessageParam(role="user", content="")

    async def _get_history_message(
        self, message: Message, limit: int
    ) -> list[EasyInputMessageParam]:
        """Retrieves and processes channel history as context."""
        messages: list[EasyInputMessageParam] = []
        hist_messages: list[Message] = []
        async for m in message.channel.history(limit=limit, before=message, oldest_first=True):
            hist_messages.append(m)

        if hist_messages:
            tasks: list[Awaitable[EasyInputMessageParam]] = []
            for hist_msg in hist_messages:
                task = self._process_single_message(message=hist_msg)
                tasks.append(task)
            processed: list[EasyInputMessageParam] = await asyncio.gather(*tasks)

            messages.append(
                EasyInputMessageParam(
                    role="system",
                    content=[
                        ResponseInputTextParam(
                            text="==== Chat History that might be helpful for answering. ====",
                            type="input_text",
                        )
                    ],
                )
            )
            messages.extend(processed)

        return messages

    async def _get_reference_message(self, message: Message) -> list[EasyInputMessageParam]:
        """Retrieves and processes the referenced message if it exists."""
        messages: list[EasyInputMessageParam] = []
        if message.reference and isinstance(message.reference.resolved, Message):
            messages.append(
                EasyInputMessageParam(
                    role="system",
                    content=[
                        ResponseInputTextParam(
                            text=f"==== Reference Message from {message.author.display_name} ({message.author.name}) [id: {message.author.id}] that might be helpful for answering. ====",
                            type="input_text",
                        )
                    ],
                )
            )
            reference_msg = await self._process_single_message(message=message.reference.resolved)
            messages.append(reference_msg)
        return messages

    async def _get_current_message(self, message: Message) -> list[EasyInputMessageParam]:
        """Processes the current message that needs to be answered."""
        messages: list[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role="system",
                content=[
                    ResponseInputTextParam(
                        text=f"==== Current Message that needs to be answered from {message.author.display_name} ({message.author.name}) [id: {message.author.id}]. ====",
                        type="input_text",
                    )
                ],
            )
        ]
        current_msg = await self._process_single_message(message=message)
        messages.append(current_msg)
        return messages

    async def _handle_video_generation(self, message: Message, user_prompt: str) -> None:
        """Handles video generation requests."""
        video = await self.client.videos.create(
            model=DEFAULT_VIDEO_MODEL,
            prompt=user_prompt,
            extra_headers={"x-litellm-end-user-id": message.author.name},
        )
        while video.status not in ("completed", "failed"):
            await asyncio.sleep(5)
            video = await self.client.videos.retrieve(
                video_id=video.id, extra_headers={"x-litellm-end-user-id": message.author.name}
            )
        if video.status != "completed":
            raise RuntimeError(f"Video generation failed: {video.error}")
        video_content = await self.client.videos.download_content(
            video_id=video.id, extra_headers={"x-litellm-end-user-id": message.author.name}
        )
        video_file = File(fp=BytesIO(video_content.content), filename="generated.mp4")
        await message.reply(content=f"{message.author.mention}", file=video_file)

    async def _handle_image_reply(self, message: Message, user_prompt: str) -> None:
        """Handles image generation or editing requests."""
        if message.reference and isinstance(message.reference.resolved, Message):
            own_parts, ref_parts = await asyncio.gather(
                self._get_attachment_parts(message=message),
                self._get_attachment_parts(message=message.reference.resolved),
            )
            attachment_parts = own_parts + ref_parts
        else:
            attachment_parts = await self._get_attachment_parts(message=message)

        data_uris: list[str] = []
        for part in attachment_parts:
            if part.get("type") == "input_image" and (image_url := part.get("image_url")):
                data_uris.append(image_url)

        if data_uris:
            image_bytes_list: list[bytes] = []
            for uri in data_uris:
                image_data = get_image_data(image_file=uri, use_b64=False)
                image_bytes_list.append(image_data)
            result = await self.client.images.edit(
                image=image_bytes_list,
                prompt=user_prompt,
                model=DEFAULT_IMAGE_MODEL,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )
        else:
            result = await self.client.images.generate(
                prompt=user_prompt,
                model=DEFAULT_IMAGE_MODEL,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )

        if not result.data:
            raise ValueError("Image operation returned no results")
        image_b64 = result.data[0].b64_json
        if image_b64 is None:
            raise ValueError("Image operation returned no b64_json")
        image_url = convert_base64_to_data_uri(image_b64)
        image_description_input: list[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(
                        text="Describe this generated image briefly for the Discord reply.",
                        type="input_text",
                    ),
                    ResponseInputImageParam(
                        image_url=image_url, detail="auto", type="input_image"
                    ),
                ],
            )
        ]
        image_responses = await self.client.responses.create(
            model=DEFAULT_FAST_MODEL,
            instructions=IMAGE_PROMPT,
            input=cast("ResponseInputParam", image_description_input),
            reasoning={"effort": "none", "summary": "auto"},
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"mock_testing_fallbacks": False},
        )
        image_description = (image_responses.output_text or "").strip()
        image_bytes = BytesIO(base64.b64decode(image_b64))
        image_file = File(fp=image_bytes, filename="generated.png")

        await message.reply(
            content=f"{message.author.mention} {image_description}", file=image_file
        )

    async def _handle_reaction(
        self, message: Message, emoji: str, previous_emoji: str | None = None
    ) -> None:
        """Handles adding and removing reactions on a message."""
        if previous_emoji and self.bot.user:
            with contextlib.suppress(Exception):
                await message.remove_reaction(emoji=previous_emoji, member=self.bot.user)
        with contextlib.suppress(Exception):
            await message.add_reaction(emoji=emoji)

    async def _route_message(self, message: Message) -> Literal["IMAGE", "QA", "SUMMARY", "VIDEO"]:
        """Routes the message to the appropriate handler."""
        message_list: list[EasyInputMessageParam] = []

        reference_messages, current_message = await asyncio.gather(
            self._get_reference_message(message=message),
            self._get_current_message(message=message),
        )
        message_list.extend(reference_messages)
        message_list.extend(current_message)

        try:
            responses = await self.client.responses.parse(
                model=DEFAULT_FAST_MODEL,
                instructions=ROUTE_PROMPT,
                input=cast("ResponseInputParam", message_list),
                text_format=RouteDecision,
                reasoning={"effort": "none", "summary": "auto"},
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
                extra_body={"mock_testing_fallbacks": False},
            )
            if responses.output_parsed is None:
                return "QA"
            return responses.output_parsed.decision
        except ValidationError:
            # The model returned no text output (e.g. safety filter, empty response);
            # model_validate_json(None) raises ValidationError before we can inspect output_parsed.
            logfire.warn("RouteDecision parse failed — model returned no text; defaulting to QA")
            return "QA"

    @staticmethod
    def _calculate_cost(
        model_name: str, input_tokens: int, output_tokens: int, reasoning_tokens: int
    ) -> float:
        """Calculates the cost of a model response based on token usage."""
        input_rate, output_rate = get_token_rates(model_name=model_name)
        total_output_tokens = output_tokens + reasoning_tokens
        return input_rate * input_tokens + output_rate * total_output_tokens

    async def _handle_streaming(  # noqa: C901, PLR0912 -- dispatches on multiple Responses API stream event types
        self, responses: AsyncStream[ResponseStreamEvent], message: Message
    ) -> str:
        """Handles streaming responses from the API and updates the Discord message."""
        stored_content = ""
        counted_content = 0
        reply: Message | None = None
        content_started = False
        model_name = ""
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        used_web_search = False

        async for response in responses:
            if response.type == "response.completed":
                model_name = response.response.model
                if response.response.usage:
                    input_tokens = response.response.usage.input_tokens
                    output_tokens = response.response.usage.output_tokens
                    reasoning_tokens = (
                        response.response.usage.output_tokens_details.reasoning_tokens
                    )
            elif response.type == "response.output_text.annotation.added":
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
                    if reply is None:
                        reply = await message.reply(content=stored_content)
                    else:
                        await reply.edit(content=stored_content)
                    counted_content = 0

        cost = self._calculate_cost(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        stored_content = _CODED_MENTION_RE.sub(r"\1", stored_content)
        usage_footer = f"\n> **{model_name}** ⬆ {input_tokens:,} ⬇ {output_tokens:,} ${cost:.8f}"
        stored_content += usage_footer

        # Final update to ensure complete message is displayed
        if reply is None:
            await message.reply(content=stored_content)
        else:
            with contextlib.suppress(Exception):
                await reply.edit(content=stored_content)

        if used_web_search:
            await self._handle_reaction(message=message, emoji="🌐")

        return stored_content

    async def _handle_message_reply(
        self, message: Message, system_prompt: str, history_limit: int, context_prompt: str = ""
    ) -> None:
        """Handles generating text replies using history and context."""
        message_list: list[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role="developer",
                content=[ResponseInputTextParam(text=context_prompt, type="input_text")],
            )
        ]

        hist_messages, reference_messages, current_message = await asyncio.gather(
            self._get_history_message(message=message, limit=history_limit),
            self._get_reference_message(message=message),
            self._get_current_message(message=message),
        )
        message_list.extend(hist_messages)
        message_list.extend(reference_messages)
        message_list.extend(current_message)

        # Workaround: gemini-pro-latest is overloaded during UTC 10:00-18:00 on weekdays, swap to the lite model.
        now = datetime.now(UTC)
        is_peak = now.weekday() < 5 and 10 <= now.hour < 17
        model = PEAK_SLOW_MODEL if is_peak else DEFAULT_SLOW_MODEL
        tools = get_tools(model=model)
        responses = await self.client.responses.create(
            model=model,
            instructions=system_prompt,
            input=cast("ResponseInputParam", message_list),
            reasoning={"effort": "high", "summary": "auto"},
            tools=tools,
            stream=True,
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"mock_testing_fallbacks": False},
        )

        await self._handle_streaming(responses=responses, message=message)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and handles AI reply generation.

        Args:
            message: The message that was sent.
        """
        # Ignore messages from bots.
        if message.author.bot:
            return

        # In DMs, always respond. In guilds, only respond when explicitly mentioned.
        # A Discord reply-notification puts the bot in message.mentions without
        # writing <@ID> into the content, so we check content to avoid
        # triggering on messages that merely reply to a functional bot post
        # (e.g. a Threads embed or a video download result).
        is_dm = message.guild is None
        if not is_dm and (not self.bot.user or f"<@{self.bot.user.id}>" not in message.content):
            return

        current_emoji = "🤔"
        await self._handle_reaction(message=message, emoji=current_emoji)
        user_prompt = await self._get_user_prompt(content=message.content)
        has_attachment = bool(message.attachments or message.stickers)

        if not user_prompt and not has_attachment:
            await self._handle_reaction(message=message, emoji="🆗", previous_emoji=current_emoji)
            await message.reply(content="?")
            return

        try:
            await self._handle_reaction(message=message, emoji="🔀", previous_emoji=current_emoji)
            current_emoji = "🔀"
            route = await self._route_message(message=message)
            if route == "IMAGE":
                await self._handle_reaction(
                    message=message, emoji="🎨", previous_emoji=current_emoji
                )
                current_emoji = "🎨"
                await self._handle_image_reply(message=message, user_prompt=user_prompt)
            elif route == "VIDEO":
                await self._handle_reaction(
                    message=message, emoji="🎬", previous_emoji=current_emoji
                )
                current_emoji = "🎬"
                await self._handle_video_generation(message=message, user_prompt=user_prompt)
            elif route == "SUMMARY":
                await self._handle_reaction(
                    message=message, emoji="📖", previous_emoji=current_emoji
                )
                current_emoji = "📖"
                await self._handle_message_reply(
                    message=message,
                    system_prompt=SUMMARY_PROMPT,
                    history_limit=100,
                    context_prompt=BELIEF,
                )
            else:
                await self._handle_reaction(
                    message=message, emoji="❓", previous_emoji=current_emoji
                )
                current_emoji = "❓"
                await self._handle_message_reply(
                    message=message,
                    system_prompt=REPLY_PROMPT,
                    history_limit=30,
                    context_prompt=BELIEF,
                )
            await self._handle_reaction(message=message, emoji="🆗", previous_emoji=current_emoji)
        except Exception as e:
            logfire.error("Failed to generate reply", _exc_info=True)
            with contextlib.suppress(Exception):
                await self._handle_reaction(
                    message=message, emoji="❌", previous_emoji=current_emoji
                )
                error_embed = Embed(
                    title="Something went wrong",
                    description=f"```\n{_extract_friendly_error(exc=e)}\n```",
                    color=0xED4245,
                )
                error_embed.set_footer(text=type(e).__name__)
                await message.reply(content=None, embed=error_embed)


async def setup(bot: commands.Bot) -> None:
    """Adds the ReplyGeneratorCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
