"""Slash commands for viewing and clearing per-user long-term memory."""

import nextcord
from nextcord import Embed, Locale, Interaction
from nextcord.ext import commands

from discordbot.cogs._memory.store import read_main_memory, clear_user_memory, count_raw_entries

_MEMORY_EMBED_COLOR = 0x5865F2
_CLEAR_EMBED_COLOR = 0x57F287


class MemoryCogs(commands.Cog):
    """Provides the personal long-term memory management commands.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the memory cog.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot

    @nextcord.slash_command(
        name="memory",
        description="Manage what the bot remembers about you.",
        name_localizations={Locale.zh_TW: "記憶", Locale.ja: "メモリー"},
        description_localizations={
            Locale.zh_TW: "管理 bot 對你的長期記憶",
            Locale.ja: "ボットがあなたについて記憶している内容を管理します。",
        },
        nsfw=False,
    )
    async def memory(self, interaction: Interaction) -> None:
        """Slash command group for per-user memory management."""

    @memory.subcommand(
        name="show",
        description="Show what the bot remembers about you.",
        name_localizations={Locale.zh_TW: "查看", Locale.ja: "表示"},
        description_localizations={
            Locale.zh_TW: "查看 bot 對你的長期記憶",
            Locale.ja: "ボットがあなたについて記憶している内容を表示します。",
        },
    )
    async def memory_show(self, interaction: Interaction) -> None:
        """Shows the caller's consolidated memory and pending observations."""
        if interaction.user is None:
            return
        memory_text = read_main_memory(user_id=interaction.user.id)
        pending_count = count_raw_entries(user_id=interaction.user.id)
        if memory_text:
            # The `v1` first line is the pipeline's format marker, not content.
            display_text = memory_text.removeprefix("v1").strip()
            embed = Embed(
                title="🧠 我對你的記憶", description=display_text, color=_MEMORY_EMBED_COLOR
            )
            if pending_count:
                embed.set_footer(text=f"另有 {pending_count} 筆新觀察待整理，會在背景慢慢併入")
            else:
                embed.set_footer(text="記憶會在你與我對話後於背景慢慢更新")
        elif pending_count:
            # Extraction has produced raw observations but the first
            # consolidation has not run yet; saying "no memory" here would
            # contradict what the user just experienced in chat.
            embed = Embed(
                title="🧠 我對你的記憶",
                description=(
                    f"我已經記下 {pending_count} 筆對你的觀察，正在整理成長期記憶，"
                    "再多聊幾次就會在這裡看到完整內容。"
                ),
                color=_MEMORY_EMBED_COLOR,
            )
        else:
            embed = Embed(
                title="🧠 我對你的記憶",
                description="目前還沒有任何記憶，多跟我聊聊，我會慢慢認識你。",
                color=_MEMORY_EMBED_COLOR,
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @memory.subcommand(
        name="clear",
        description="Clear everything the bot remembers about you.",
        name_localizations={Locale.zh_TW: "清除", Locale.ja: "消去"},
        description_localizations={
            Locale.zh_TW: "清除 bot 對你的所有長期記憶",
            Locale.ja: "ボットがあなたについて記憶している内容をすべて消去します。",
        },
    )
    async def memory_clear(self, interaction: Interaction) -> None:
        """Deletes the caller's memory files and aborts in-flight updates.

        Deliberately lock-free: a background memory update can hold the
        per-user lock across LLM calls far past Discord's interaction ack
        window. `clear_user_memory` flags `mark_cleared`, and in-flight
        updates re-check `cleared_since` before every write, so the clear
        cannot be resurrected by a slower background task.
        """
        if interaction.user is None:
            return
        removed = clear_user_memory(user_id=interaction.user.id)
        description = "已清除我對你的所有記憶。" if removed else "本來就沒有任何記憶，無事發生。"
        embed = Embed(title="🧹 記憶清除", description=description, color=_CLEAR_EMBED_COLOR)
        await interaction.response.send_message(embed=embed, ephemeral=True)


def setup(bot: commands.Bot) -> None:
    """Adds the MemoryCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(MemoryCogs(bot), override=True)
