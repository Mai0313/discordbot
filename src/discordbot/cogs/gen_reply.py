from typing import Any
import contextlib

from openai import AsyncStream
import logfire
import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands
from openai.types.chat import ChatCompletionChunk

from discordbot.sdk.llm import LLMSDK

available_models = ["openrouter/x-ai/grok-4.1-fast"]
MODEL_CHOICES = {"grok-4.1-fast": "openrouter/x-ai/grok-4.1-fast"}


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the ReplyGeneratorCogs.

        Args:
            bot (commands.Bot): The bot instance.
        """
        self.bot = bot
        # 儲存每個用戶的對話紀錄
        # key: user_id, value: list of message dicts
        self.user_memory: dict[int, list[dict[str, Any]]] = {}

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
        description="I can reply from hints, search the web.",
        name_localizations={Locale.zh_TW: "生成", Locale.ja: "生成"},
        description_localizations={
            Locale.zh_TW: "我可以回答問題, 上網搜尋",
            Locale.ja: "提示に基づいて返答を生成し、検索もできます。",
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

        Args:
            interaction (Interaction): The interaction object for the command.
            prompt (str): The prompt text provided by the user.
            model (str): The selected model, defaults to "gpt-5" if not specified.
            image (Optional[nextcord.Attachment]): An optional image attachment uploaded by the user.
        """
        await interaction.response.defer()
        attachments = []
        if image:
            attachments.append(image.url)

        # 初始狀態訊息
        await interaction.followup.send(content="🤔 思考中...")

        try:
            llm_sdk = LLMSDK(model=model)
            # 使用 completion content 格式 (ChatCompletion)
            content = await llm_sdk.prepare_completion_content(
                prompt=prompt, attachments=attachments
            )
            content = f"You are not allowed to use Simplified Chinese in your response.\n{content}"

            user_id = interaction.user.id
            if user_id not in self.user_memory:
                self.user_memory[user_id] = []

            # 將用戶訊息加入記憶
            self.user_memory[user_id].append({"role": "user", "content": content})

            try:
                stream = await llm_sdk.client.chat.completions.create(
                    model=model, messages=self.user_memory[user_id], stream=True
                )
            except Exception as e:
                # 若發生錯誤，可能是 content filter 或其他問題，不清除記憶但報錯
                # 或是如果 memory 太長導致 context length exceeded，可能需要清理
                # 這裡簡單報錯
                logfire.error("Error creating chat completion", _exc_info=True)
                raise e

            response_text = await self._handle_streaming_response(
                interaction=interaction, stream=stream, update_per_words=10
            )

            # 將 AI 回應加入記憶
            if response_text:
                self.user_memory[user_id].append({"role": "assistant", "content": response_text})

        except Exception as e:
            await interaction.edit_original_message(
                content=f"{interaction.user.mention}\n❌ 錯誤:\n{e}"
            )
            logfire.error("Error in oai", _exc_info=True)

    async def _handle_streaming_response(
        self,
        interaction: Interaction,
        stream: AsyncStream[ChatCompletionChunk],
        update_per_words: int = 10,
    ) -> str:
        """處理 streaming 回應，每 10 個字更新一次訊息。

        Returns:
            str: 完整的生成文字
        """
        accumulated_text = ""
        char_count = 0

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta.content:
                accumulated_text += delta.content
                char_count += len(delta.content)

                # 每 X 個字更新一次訊息
                if char_count >= update_per_words:
                    try:
                        await interaction.edit_original_message(
                            content=f"{interaction.user.mention}\n{accumulated_text}"
                        )
                        char_count = 0
                    except nextcord.errors.NotFound:
                        # 訊息可能被刪除
                        break
                    except Exception as e:
                        logfire.warning(f"Failed to update message: {e}")

        # 最終更新確保顯示完整訊息
        with contextlib.suppress(Exception):
            await interaction.edit_original_message(
                content=f"{interaction.user.mention}\n{accumulated_text}"
            )

        return accumulated_text

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
        had_memory = self.user_memory.pop(user_id, None) is not None

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
