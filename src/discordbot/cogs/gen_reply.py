from io import BytesIO
import re
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
from pydantic import ValidationError
from nextcord.ext import commands
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.utils.images import get_pil_image, get_image_data, convert_base64_to_data_uri
from discordbot.typings.models import ModelSettings, RouteDecision
from discordbot.utils.model_pricing import get_token_rates
from discordbot.cogs._gen_reply.prompts import (
    BELIEF,
    IMAGE_PROMPT,
    REPLY_PROMPT,
    ROUTE_PROMPT,
    SUMMARY_PROMPT,
)
from discordbot.cogs._gen_reply.exceptions import extract_friendly_error

if TYPE_CHECKING:
    from collections.abc import Awaitable

# Gemini occasionally wraps Discord mention syntax in backticks (inline code),
# which stops Discord from rendering the actual mention. Strip those wrappers
# before sending; matches user (<@id>, <@!id>), role (<@&id>) and channel (<#id>) mentions.
_CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")


class ReplyGeneratorCogs(commands.Cog):
    """Generates AI replies for Discord messages.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: The LLM client configuration loaded for reply generation.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the ReplyGeneratorCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The cached AsyncOpenAI client instance.

        Returns:
            A configured AsyncOpenAI client reused across reply requests.
        """
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return client

    @property
    def image_model(self) -> ModelSettings:
        """The image generation/edit model used by `images.generate` and `images.edit`."""
        image_model = ModelSettings(name="gemini-3.1-flash-image-preview", effort=None)
        return image_model

    @property
    def video_model(self) -> ModelSettings:
        """The video generation model used by `videos.create`."""
        video_model = ModelSettings(name="veo-3.1-fast-generate-preview", effort=None)
        return video_model

    @property
    def fast_model(self) -> ModelSettings:
        """The fast model used for routing decisions and image captioning."""
        fast_model = ModelSettings(name="gemini-flash-latest", effort="none")
        return fast_model

    @property
    def slow_model(self) -> ModelSettings:
        """Selects the slow model based on time of day to avoid overload periods; `gemini-pro-latest` is overloaded during UTC 10:00-17:00 on weekdays, swap to the lite model."""
        now = datetime.now(UTC)
        is_peak = now.weekday() < 5 and 9 <= now.hour < 17
        if is_peak:
            return ModelSettings(name="azure/gpt-5.4", effort="high")
        return ModelSettings(name="gemini-pro-latest", effort="high")

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
        """Extracts attachment content parts from a message.

        Video attachments are skipped when the slow model (the one that
        actually consumes the heaviest payload) does not accept them; routing
        and image-edit paths inherit the same gate, which is fine because the
        former classifies intent without inspecting frames and the latter
        downstream-filters to ``input_image`` anyway.
        """
        _content_parts: list[ResponseInputImageParam | ResponseInputFileParam | None] = []

        for attachment in message.attachments:
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            if content_type.startswith("image/"):
                _content_parts.append(await self._image_to_part(source=attachment))
            elif content_type.startswith("video/"):
                if "video" not in self.slow_model.input_modalities:
                    logfire.warn(
                        f"Skipping video attachment for {self.slow_model.name}: {attachment.filename}"
                    )
                    continue
                _content_parts.append(await self._attachment_to_part(attachment=attachment))
            else:
                # application/pdf, text/plain, application/json, etc.
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
            model=self.video_model.name,
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
                model=self.image_model.name,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )
        else:
            result = await self.client.images.generate(
                prompt=user_prompt,
                model=self.image_model.name,
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
            model=self.fast_model.name,
            instructions=IMAGE_PROMPT,
            input=cast("ResponseInputParam", image_description_input),
            reasoning=self.fast_model.reasoning,
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
        self, message: Message, emoji: str, previous: str | None = None
    ) -> None:
        """Handles adding and removing reactions on a message."""
        if previous and self.bot.user:
            with contextlib.suppress(Exception):
                await message.remove_reaction(emoji=previous, member=self.bot.user)
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
                model=self.fast_model.name,
                instructions=ROUTE_PROMPT,
                input=cast("ResponseInputParam", message_list),
                text_format=RouteDecision,
                reasoning=self.fast_model.reasoning,
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
    def _calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculates the cost of a model response based on token usage."""
        input_rate, output_rate = get_token_rates(model_name=model_name)
        return input_rate * input_tokens + output_rate * output_tokens

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
        used_web_search = False

        async for response in responses:
            if response.type == "response.completed":
                model_name = response.response.model
                if response.response.usage:
                    input_tokens = response.response.usage.input_tokens
                    output_tokens = response.response.usage.output_tokens
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
            model_name=model_name, input_tokens=input_tokens, output_tokens=output_tokens
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
        self, message: Message, system_prompt: str, context_prompt: str, history_limit: int
    ) -> None:
        """Handles generating text replies using history and context."""
        message_list: list[EasyInputMessageParam] = [
            # Temp skip since this belief is too strong in responses and causes refusal to answer; revisit after prompt tuning.
            # EasyInputMessageParam(
            #     role="user",
            #     content=[ResponseInputTextParam(text=context_prompt, type="input_text")],
            # )
        ]

        hist_messages, reference_messages, current_message = await asyncio.gather(
            self._get_history_message(message=message, limit=history_limit),
            self._get_reference_message(message=message),
            self._get_current_message(message=message),
        )
        message_list.extend(hist_messages)
        message_list.extend(reference_messages)
        message_list.extend(current_message)

        responses = await self.client.responses.create(
            model=self.slow_model.name,
            instructions=system_prompt,
            input=cast("ResponseInputParam", message_list),
            reasoning=self.slow_model.reasoning,
            tools=self.slow_model.tools,
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
            await self._handle_reaction(message=message, emoji="🆗", previous=current_emoji)
            await message.reply(content="?")
            return

        try:
            await self._handle_reaction(message=message, emoji="🔀", previous=current_emoji)
            current_emoji = "🔀"
            route = await self._route_message(message=message)
            if route == "IMAGE":
                await self._handle_reaction(message=message, emoji="🎨", previous=current_emoji)
                current_emoji = "🎨"
                await self._handle_image_reply(message=message, user_prompt=user_prompt)
            elif route == "VIDEO":
                await self._handle_reaction(message=message, emoji="🎬", previous=current_emoji)
                current_emoji = "🎬"
                await self._handle_video_generation(message=message, user_prompt=user_prompt)
            elif route == "SUMMARY":
                await self._handle_reaction(message=message, emoji="📖", previous=current_emoji)
                current_emoji = "📖"
                await self._handle_message_reply(
                    message=message,
                    system_prompt=SUMMARY_PROMPT,
                    context_prompt=BELIEF,
                    history_limit=200,
                )
            else:
                await self._handle_reaction(message=message, emoji="❓", previous=current_emoji)
                current_emoji = "❓"
                await self._handle_message_reply(
                    message=message,
                    system_prompt=REPLY_PROMPT,
                    context_prompt=BELIEF,
                    history_limit=30,
                )
            await self._handle_reaction(message=message, emoji="🆗", previous=current_emoji)
        except Exception as e:
            logfire.error("Failed to generate reply", _exc_info=True)
            with contextlib.suppress(Exception):
                await self._handle_reaction(message=message, emoji="❌", previous=current_emoji)
                error_embed = Embed(
                    title="Something went wrong",
                    description=f"```\n{extract_friendly_error(exc=e)}\n```",
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
