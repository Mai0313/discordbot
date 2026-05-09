"""View components attached to AI replies."""

from typing import TYPE_CHECKING
import contextlib

from nextcord import Message, ButtonStyle, Interaction, ui

if TYPE_CHECKING:
    from discordbot.cogs.gen_reply import ReplyGeneratorCogs


class RegenerateView(ui.View):
    """Single-button view that re-runs reply generation for the original user message.

    Attributes:
        cog: The ReplyGeneratorCogs instance owning the dispatch flow.
        original_message: The user's message that triggered the original reply.
    """

    def __init__(self, cog: "ReplyGeneratorCogs", original_message: Message) -> None:
        """Initializes the regenerate view.

        Args:
            cog: The owning ReplyGeneratorCogs (used to re-enter on_message).
            original_message: The user's message to regenerate a reply for.
        """
        super().__init__(timeout=600)
        self.cog = cog
        self.original_message = original_message

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Restricts the button to the original message author."""
        if interaction.user is None or interaction.user.id != self.original_message.author.id:
            await interaction.response.send_message(
                content="只有原作者可以重新生成喔", ephemeral=True
            )
            return False
        return True

    @ui.button(label="重新生成", emoji="🔄", style=ButtonStyle.secondary)
    async def regenerate(self, button: ui.Button, interaction: Interaction) -> None:
        """Deletes the old AI reply and re-runs the dispatch flow on the user's message.

        Args:
            button: The clicked button (provided by nextcord).
            interaction: The interaction representing the click.
        """
        await interaction.response.defer()
        # Stop accepting clicks; on_message will create a fresh view on the new reply.
        self.stop()

        if interaction.message:
            with contextlib.suppress(Exception):
                await interaction.message.delete()

        # Clear the bot's reactions on the user's original message so the new
        # 🤔 → 🔀 → ... sequence doesn't pile on top of a stale 🆗.
        bot_user = self.cog.bot.user
        if bot_user:
            for reaction in list(self.original_message.reactions):
                if reaction.me:
                    with contextlib.suppress(Exception):
                        await self.original_message.remove_reaction(
                            emoji=reaction.emoji, member=bot_user
                        )

        await self.cog.on_message(message=self.original_message)
