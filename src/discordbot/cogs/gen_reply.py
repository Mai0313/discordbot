from io import BytesIO
import base64
import datetime

from openai import AsyncStream, BadRequestError
import logfire
import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.tool_param import ImageGeneration
from openai.types.responses.web_search_tool_param import WebSearchToolParam

from discordbot.sdk.llm import LLMSDK

available_models = [
    "openai/gpt-4o",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "claude-3-5-haiku-20241022",
]
MODEL_CHOICES = {available_model: available_model for available_model in available_models}

_TOOLS = [
    WebSearchToolParam(type="web_search_preview"),
    ImageGeneration(type="image_generation"),  # 圖片可能很貴 看情況解決
]


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the ReplyGeneratorCogs.

        Args:
            bot (commands.Bot): The bot instance.
        """
        self.bot = bot
        # 儲存每個用戶的上一個 response ID，用於對話記憶
        self.user_last_response_id: dict[int, str] = {}

    async def _get_attachment_list(
        self, messages: list[nextcord.Message] | None = None
    ) -> list[str]:
        """Retrieve all attachments from a message.

        This function extracts image attachment URLs, embed descriptions, and converts sticker images to base64 encoded strings. If the message is None, an empty list is returned.

        Args:
            messages (Optional[list[nextcord.Message]]): The message from which to extract attachments.

        Returns:
            List[str]: A list containing the attachment URLs and base64 encoded sticker images.
        """
        if messages is None:
            messages = []
        attachments: list[str] = []
        for message in messages:
            if message.attachments:
                _attach = [attachment.url for attachment in message.attachments if attachment.url]
                if _attach:
                    attachments.extend(_attach)
            if message.embeds:
                _attach = [embed.description for embed in message.embeds if embed.description]
                if _attach:
                    attachments.extend(_attach)
            if message.stickers:
                _attach = [sticker.url for sticker in message.stickers]
                if _attach:
                    attachments.extend(_attach)
        return attachments

    @nextcord.slash_command(
        name="oai",
        description="I can reply from hints, search the web, or draw.",
        name_localizations={Locale.zh_TW: "生成", Locale.ja: "生成"},
        description_localizations={
            Locale.zh_TW: "我可以回答問題, 上網搜尋, 也可以畫圖",
            Locale.ja: "提示に基づいて返答を生成し、検索や描画もできます。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def oai(
        self,
        interaction: Interaction,
        prompt: str = SlashOption(
            description="Enter your prompt.",
            description_localizations={
                Locale.zh_TW: "請輸入提示詞。",
                Locale.ja: "プロンプトを入力してください。",
            },
        ),
        model: str = SlashOption(
            description="Choose a model (default: GPT-5).",
            description_localizations={
                Locale.zh_TW: "選擇模型 (預設為 GPT-5)",
                Locale.ja: "モデルを選択してください（デフォルトは GPT-5）",
            },
            choices=MODEL_CHOICES,
            required=False,
            default=available_models[0],
        ),
        image: nextcord.Attachment | None = SlashOption(  # noqa: B008
            description="(Optional) Upload an image.",
            description_localizations={
                Locale.zh_TW: "（可選）上傳一張圖片。",
                Locale.ja: "（オプション）画像をアップロードしてください。",
            },
            required=False,
        ),
    ) -> None:
        """Generate a reply based on the user's prompt.

        If the model 'o1' is selected along with an image, an error message is returned since 'o1' does not support image input.
        The function can either generate a complete response or stream the response in real-time based on the stream parameter.

        Args:
            interaction (Interaction): The interaction object for the command.
            prompt (str): The prompt text provided by the user.
            model (str): The selected model, defaults to "gpt-5" if not specified.
            image (Optional[nextcord.Attachment]): An optional image attachment uploaded by the user.
        """
        await interaction.response.defer()
        attachments = []
        if model not in ["o1", "o1-mini"] and image:
            attachments.append(image.url)

        # 初始狀態訊息
        await interaction.followup.send(content="🤔 思考中...")

        try:
            llm_sdk = LLMSDK(model=model)
            content = await llm_sdk.prepare_response_content(
                prompt=prompt, attachments=attachments
            )
            # 準備 streaming 請求
            try:
                # 獲取用戶的最新 response ID
                previous_response_id = self.user_last_response_id.get(interaction.user.id, None)
                stream = await llm_sdk.client.responses.create(
                    model=model,
                    tools=_TOOLS,
                    input=[{"role": "user", "content": content}],
                    stream=True,
                    previous_response_id=previous_response_id,
                )
            except BadRequestError:
                # 如果 API 回傳錯誤（response ID 無效），清理該用戶記錄並重新嘗試
                self.user_last_response_id.pop(interaction.user.id, None)
                stream = await llm_sdk.client.responses.create(
                    model=model,
                    tools=_TOOLS,
                    input=[{"role": "user", "content": content}],
                    stream=True,
                )

            await self._handle_streaming_response(
                interaction=interaction, stream=stream, prompt=prompt, update_per_words=10
            )

        except Exception as e:
            await interaction.edit_original_message(
                content=f"{interaction.user.mention}\n❌ 錯誤:\n{e}"
            )
            logfire.error("Error in oai", _exc_info=True)

    async def _handle_streaming_response(
        self,
        interaction: Interaction,
        stream: AsyncStream[ResponseStreamEvent],
        prompt: str,
        update_per_words: int = 10,
    ) -> None:
        """處理 streaming 回應，每 10 個字更新一次訊息。"""
        accumulated_text = ""
        accumulated_image = ""

        char_count = 0
        async for event in stream:
            # 處理完成事件，獲取 response ID
            if event.type == "response.completed":
                self.user_last_response_id[interaction.user.id] = event.response.id
                continue

            # 處理文字串流
            if event.type == "response.output_text.delta":
                accumulated_text += event.delta
                char_count += len(event.delta)
                # 每 X 個字更新一次訊息
                if char_count >= update_per_words:
                    await interaction.edit_original_message(
                        content=f"{interaction.user.mention}\n{accumulated_text}"
                    )
                    char_count = 0

            # 處理圖片生成串流
            if event.type == "response.image_generation_call.partial_image":
                accumulated_image += event.partial_image_b64
                await self._display_image(
                    interaction=interaction,
                    text=accumulated_text,
                    image_base64=event.partial_image_b64,
                    prompt=prompt,
                )

        await interaction.edit_original_message(
            content=f"{interaction.user.mention}\n{accumulated_text}"
        )
        # 文字一定會有 圖片不一定
        if accumulated_image:
            await self._display_image(
                interaction=interaction,
                text=accumulated_text,
                image_base64=accumulated_image,
                prompt=prompt,
            )

    async def _display_image(
        self, interaction: Interaction, text: str, image_base64: str, prompt: str
    ) -> None:
        """顯示最終完整圖片。"""
        try:
            image_bytes = base64.b64decode(image_base64)
            filename = "generated_image.png"
            file_obj = nextcord.File(BytesIO(image_bytes), filename=filename)
            embed_obj = nextcord.Embed(
                color=nextcord.Color.green(),
                title="🖼️ 生成完成",
                description=f"提示詞: {prompt}",
                timestamp=datetime.datetime.now(),
            )
            embed_obj.set_image(url=f"attachment://{filename}")
            embed_obj.set_footer(text="Images generated via Responses API")
            await interaction.edit_original_message(
                content=f"{interaction.user.mention}\n{text}", file=file_obj, embed=embed_obj
            )
        except Exception as e:
            logfire.warning(f"Failed to display final image: {e}")

    @nextcord.slash_command(
        name="clear_memory",
        description="Clear your conversation memory with the bot.",
        name_localizations={Locale.zh_TW: "清除記憶", Locale.ja: "メモリをクリア"},
        description_localizations={
            Locale.zh_TW: "清除你與機器人的對話記憶。",
            Locale.ja: "ボットとの会話メモリをクリアします。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def clear_memory(self, interaction: Interaction) -> None:
        """清除用戶的對話記憶。

        Args:
            interaction (Interaction): The interaction object for the command.
        """
        user_id = interaction.user.id
        had_memory = self.user_last_response_id.pop(user_id, None) is not None

        if had_memory:
            await interaction.response.send_message(
                content="對話記憶已清除! 下次對話將重新開始。", ephemeral=True
            )
        else:
            await interaction.response.send_message(
                content="你目前沒有對話記憶需要清除。", ephemeral=True
            )


async def setup(bot: commands.Bot) -> None:
    """Register the reply generation cog with the bot.

    Args:
        bot (commands.Bot): The bot instance to which the cog will be added.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
