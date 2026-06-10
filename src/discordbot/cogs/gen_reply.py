"""Cog that routes Discord messages through the AI reply pipeline."""

from io import BytesIO
import re
import base64
from typing import TYPE_CHECKING, Literal, cast
import asyncio
from functools import cached_property
import contextlib

from openai import AsyncOpenAI
import logfire
from nextcord import File, Embed, Message
from pydantic import ValidationError
from nextcord.ext import commands
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseInputParam,
    EasyInputMessageParam,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_function_tool_call_param import ResponseFunctionToolCallParam

from discordbot.utils.llm import create_litellm_client
from discordbot.typings.llm import LLMConfig
from discordbot.utils.images import get_image_data, convert_base64_to_data_uri
from discordbot.typings.models import RouteDecision, RuntimeModelCatalog
from discordbot.utils.reactions import update_reaction
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.cogs._gen_reply.input import (
    MessageInputBuilder,
    sanitize_identity,
    render_author_identity,
)
from discordbot.cogs._memory.pipeline import schedule_memory_update
from discordbot.cogs._gen_reply.prompts import (
    IMAGE_PROMPT,
    REPLY_PROMPT,
    ROUTE_PROMPT,
    SUMMARY_PROMPT,
)
from discordbot.cogs._memory.extraction import MemoryExtractorAI, target_centered_memory_messages
from discordbot.cogs._gen_reply.streaming import ResponseStreamer
from discordbot.cogs._gen_reply.exceptions import extract_friendly_error
from discordbot.cogs._gen_reply.memory_tool import (
    GET_USER_MEMORY_TOOL,
    dump_user_memories,
    parse_user_id_list,
    resolve_user_memories,
    build_memory_allowlist,
    render_callable_users_block,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable


_MESSAGE_URL_RE = re.compile(pattern=r"(?i)\b(?:https?://|www\.)\S+")


def _message_has_url(content: str) -> bool:
    """Returns whether the current message carries an explicit URL."""
    return _MESSAGE_URL_RE.search(string=content) is not None


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
        self.runtime_models = RuntimeModelCatalog()

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The cached AsyncOpenAI client instance.

        Returns:
            A configured AsyncOpenAI client reused across reply requests.
        """
        return create_litellm_client(config=self.config)

    @cached_property
    def memory_extractor(self) -> MemoryExtractorAI:
        """The cached per-user memory extraction service.

        Returns:
            An extractor bound to this cog's client and the phase-1/phase-2
            memory models.
        """
        return MemoryExtractorAI(
            client=self.client,
            extract_model=self.runtime_models.extract_model,
            evaluate_model=self.runtime_models.memory_evaluator_model,
            consolidate_model=self.runtime_models.memories_model,
        )

    @cached_property
    def input_builder(self) -> MessageInputBuilder:
        """The cached Discord-message-to-Responses-API input builder.

        Returns:
            A builder bound to this bot and runtime model catalog.
        """
        return MessageInputBuilder(bot=self.bot, runtime_models=self.runtime_models)

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
                task = self.input_builder.process_single_message(message=hist_msg)
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
        """Walks the reference chain up to depth 3 and renders each link as context."""
        chain: list[Message] = []
        visited: set[int] = {message.id}
        current = message
        while (
            len(chain) < 3
            and current.reference
            and isinstance(current.reference.resolved, Message)
            and current.reference.resolved.id not in visited
        ):
            ref = current.reference.resolved
            visited.add(ref.id)
            chain.append(ref)
            current = ref

        if not chain:
            return []

        tasks: list[Awaitable[EasyInputMessageParam]] = []
        for ref in chain:
            task = self.input_builder.process_single_message(message=ref)
            tasks.append(task)
        processed: list[EasyInputMessageParam] = await asyncio.gather(*tasks)

        messages: list[EasyInputMessageParam] = []
        for ref, processed_ref in zip(reversed(chain), reversed(processed), strict=True):
            messages.append(
                EasyInputMessageParam(
                    role="system",
                    content=[
                        ResponseInputTextParam(
                            text=(
                                f"==== Reference Message from {sanitize_identity(value=ref.author.display_name)} "
                                f"({sanitize_identity(value=ref.author.name)}) [id: {ref.author.id}] that might be helpful "
                                "for answering. ===="
                            ),
                            type="input_text",
                        )
                    ],
                )
            )
            messages.append(processed_ref)
        return messages

    async def _get_current_message(self, message: Message) -> list[EasyInputMessageParam]:
        """Processes the current message that needs to be answered."""
        messages: list[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role="system",
                content=[
                    ResponseInputTextParam(
                        text=f"==== Current Message that needs to be answered from {sanitize_identity(value=message.author.display_name)} ({sanitize_identity(value=message.author.name)}) [id: {message.author.id}]. ====",
                        type="input_text",
                    )
                ],
            )
        ]
        current_msg = await self.input_builder.process_single_message(message=message)
        messages.append(current_msg)
        return messages

    async def _collect_participant_messages(
        self, message: Message, history_limit: int
    ) -> list[Message]:
        """Collects the raw current, reference, and history messages for the memory allowlist.

        The rendered Responses input drops the underlying `Message` objects, but the
        allowlist needs each message's `.author` and `.mentions`. This re-walks the same
        reference chain (depth 3) and channel history the `_get_*` helpers render.
        """
        messages: list[Message] = [message]

        visited: set[int] = {message.id}
        current = message
        while (
            len(messages) < 4
            and current.reference
            and isinstance(current.reference.resolved, Message)
            and current.reference.resolved.id not in visited
        ):
            ref = current.reference.resolved
            visited.add(ref.id)
            messages.append(ref)
            current = ref

        async for hist_msg in message.channel.history(
            limit=history_limit, before=message, oldest_first=True
        ):
            messages.append(hist_msg)

        return messages

    async def _handle_video_reply(self, message: Message, user_prompt: str) -> None:
        """Handles video generation requests."""
        video_model = self.runtime_models.video_model
        video = await self.client.videos.create(
            model=video_model.name,
            prompt=user_prompt or "請依照訊息內容生成一段影片。",
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
        image_model = self.runtime_models.image_model
        if message.reference and isinstance(message.reference.resolved, Message):
            own_parts, ref_parts = await asyncio.gather(
                self.input_builder.get_attachment_parts(message=message),
                self.input_builder.get_attachment_parts(message=message.reference.resolved),
            )
            attachment_parts = own_parts + ref_parts
        else:
            attachment_parts = await self.input_builder.get_attachment_parts(message=message)

        data_uris: list[str] = []
        for part in attachment_parts:
            if part.get("type") == "input_image" and (image_url := part.get("image_url")):
                data_uris.append(image_url)

        if data_uris:
            tasks = []
            for uri in data_uris:
                tasks.append(asyncio.to_thread(get_image_data, image_file=uri, use_b64=False))
            image_bytes_list: list[bytes] = list(await asyncio.gather(*tasks))
            result = await self.client.images.edit(
                image=image_bytes_list,
                prompt=user_prompt or "請依照附件內容進行編輯或優化。",
                model=image_model.name,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )
        else:
            result = await self.client.images.generate(
                prompt=user_prompt or "請生成一張圖片。",
                model=image_model.name,
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
        fast_model = self.runtime_models.fast_model
        image_responses = await self.client.responses.create(
            model=fast_model.name,
            instructions=IMAGE_PROMPT,
            input=cast("ResponseInputParam", image_description_input),
            reasoning=fast_model.reasoning,
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"mock_testing_fallbacks": False},
        )
        image_description = (image_responses.output_text or "").strip()
        image_bytes = BytesIO(base64.b64decode(image_b64))
        image_file = File(fp=image_bytes, filename="generated.png")
        final_content = f"{message.author.mention} {image_description}"
        await message.reply(content=final_content, file=image_file)

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
            fast_model = self.runtime_models.fast_model
            responses = await self.client.responses.parse(
                model=fast_model.name,
                instructions=ROUTE_PROMPT,
                input=cast("ResponseInputParam", message_list),
                text_format=RouteDecision,
                reasoning=fast_model.reasoning,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
                extra_body={"mock_testing_fallbacks": False},
            )
            if responses.output_parsed is None:
                return "QA"
            decision = responses.output_parsed.decision
            if decision == "SUMMARY" and _message_has_url(content=message.content):
                return "QA"
            return decision
        except ValidationError:
            # The model returned no text output (e.g. safety filter, empty response);
            # model_validate_json(None) raises ValidationError before we can inspect output_parsed.
            logfire.warn("RouteDecision parse failed, model returned no text; defaulting to QA")
            return "QA"

    async def _handle_message_reply(
        self, message: Message, system_prompt: str, history_limit: int, memory_enabled: bool = True
    ) -> None:
        """Handles generating text replies using history and context."""
        hist_messages, reference_messages, current_message = await asyncio.gather(
            self._get_history_message(message=message, limit=history_limit),
            self._get_reference_message(message=message),
            self._get_current_message(message=message),
        )
        message_list: list[EasyInputMessageParam] = [
            *hist_messages,
            *reference_messages,
            *current_message,
        ]

        # Long-term memory is no longer injected: the slow model decides whether and
        # whose memory to read by calling the get_user_memory tool. The allowlist
        # (conversation authors + mentioned users, minus the bot) is the permission
        # boundary; `message_list` stays memory-free for phase-1 extraction.
        slow_model = self.runtime_models.slow_model
        base_tools = list(slow_model.tools)
        tools = base_tools
        running_input: ResponseInputParam = [*message_list]
        allowed: dict[int, str] = {}
        if memory_enabled and self.bot.user:
            # Only the allowlist needs the raw Message objects, so the SUMMARY route
            # (memory_enabled=False) skips this second history fetch entirely.
            participant_messages = await self._collect_participant_messages(
                message=message, history_limit=history_limit
            )
            allowed = build_memory_allowlist(
                messages=participant_messages, bot_user_id=self.bot.user.id
            )
            if allowed:
                tools = [*base_tools, GET_USER_MEMORY_TOOL]
                running_input.append(render_callable_users_block(allowed=allowed))

        streamer = ResponseStreamer(message=message)
        for turn in range(3):
            # The final turn drops the memory tool so a tool-looping model is always
            # forced to produce a text answer instead of exhausting the turn cap.
            turn_tools = tools if turn < 2 else base_tools
            responses = await self.client.responses.create(
                model=slow_model.name,
                instructions=system_prompt,
                input=running_input,
                reasoning=slow_model.reasoning,
                tools=turn_tools,
                stream=True,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
                extra_body={"mock_testing_fallbacks": False},
            )
            function_calls = await streamer.consume_turn(responses=responses)
            if not function_calls:
                break
            for call in function_calls:
                memories = resolve_user_memories(
                    user_id_list=parse_user_id_list(arguments=call.arguments), allowed=allowed
                )
                running_input.append(
                    ResponseFunctionToolCallParam(
                        type="function_call",
                        call_id=call.call_id,
                        name=call.name,
                        arguments=call.arguments,
                    )
                )
                running_input.append(
                    FunctionCallOutput(
                        type="function_call_output",
                        call_id=call.call_id,
                        output=dump_user_memories(memories=memories),
                    )
                )

        full_reply = await streamer.finalize()
        if memory_enabled:
            memory_message_list = target_centered_memory_messages(
                hist_messages=hist_messages,
                reference_messages=reference_messages,
                current_message=current_message,
                target_user_id=message.author.id,
            )
            schedule_memory_update(
                user_id=message.author.id,
                message_list=memory_message_list,
                full_reply=full_reply,
                extractor=self.memory_extractor,
                identity=render_author_identity(
                    display_name=message.author.display_name,
                    username=message.author.name,
                    user_id=message.author.id,
                ),
            )

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and handles AI reply generation.

        Args:
            message: The message that was sent.
        """
        # Ignore messages from bots.
        if message.author.bot:
            return

        # Match <@ID> in content, not message.mentions: reply notifications add
        # the bot to mentions and would trigger on replies to functional bot
        # posts (e.g. Threads embeds, video downloads).
        is_dm = message.guild is None
        if not is_dm and not self.input_builder.has_bot_mention(content=message.content):
            return

        user_prompt = await self.input_builder.get_user_prompt(content=message.content)
        has_attachment = bool(message.attachments or message.stickers)

        if not user_prompt and not has_attachment:
            await update_reaction(message=message, bot_user=self.bot.user, emoji="❓")
            await message.reply(content="?")
            return

        try:
            current_emoji = await update_reaction(
                message=message, bot_user=self.bot.user, emoji="🔀"
            )
            route = await self._route_message(message=message)
            if route == "IMAGE":
                current_emoji = await update_reaction(
                    message=message, bot_user=self.bot.user, emoji="🎨", previous=current_emoji
                )
                await self._handle_image_reply(message=message, user_prompt=user_prompt)
            elif route == "VIDEO":
                current_emoji = await update_reaction(
                    message=message, bot_user=self.bot.user, emoji="🎬", previous=current_emoji
                )
                await self._handle_video_reply(message=message, user_prompt=user_prompt)
            elif route == "SUMMARY":
                current_emoji = await update_reaction(
                    message=message, bot_user=self.bot.user, emoji="📖", previous=current_emoji
                )
                # Summaries digest ~100 channel messages: skip per-user memory
                # so it neither biases the digest nor floods extraction.
                await self._handle_message_reply(
                    message=message,
                    system_prompt=SUMMARY_PROMPT,
                    history_limit=100,
                    memory_enabled=False,
                )
            else:
                current_emoji = await update_reaction(
                    message=message, bot_user=self.bot.user, emoji="❓", previous=current_emoji
                )
                await self._handle_message_reply(
                    message=message, system_prompt=REPLY_PROMPT, history_limit=30
                )
            await update_reaction(
                message=message, bot_user=self.bot.user, emoji="🆗", previous=current_emoji
            )
        except Exception as e:
            logfire.error("Failed to generate reply", user_id=message.author.name, _exc_info=True)
            with contextlib.suppress(Exception):
                await update_reaction(message=message, bot_user=self.bot.user, emoji="❌")
                error_embed = Embed(
                    title="Something went wrong",
                    description=f"```\n{extract_friendly_error(exc=e)}\n```",
                    color=0xED4245,
                )
                error_embed.set_footer(text=type(e).__name__)
                await message.reply(
                    content=None,
                    embed=error_embed,
                    **embed_spacer_payload(embeds=[error_embed], is_edit=False, target=message),
                )


def setup(bot: commands.Bot) -> None:
    """Adds the ReplyGeneratorCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
