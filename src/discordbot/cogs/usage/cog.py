"""Records every slash-command invocation to the usage log.

Listens on `on_interaction` rather than `on_application_command_completion` so invocations
rejected by permission checks, targeting unresolvable commands, or raising inside the callback
are still recorded. Implemented as a cog listener because overriding `on_interaction` on
`DiscordBot` would replace `Client.on_interaction` and prevent slash commands from running.
"""

from typing import TYPE_CHECKING, cast

from nextcord import Interaction, InteractionType, ApplicationCommandOptionType
from nextcord.ext import commands

from discordbot.utils.usage_log import UsageRecorder

if TYPE_CHECKING:
    from nextcord.types.interactions import (
        ApplicationCommandInteractionData,
        ApplicationCommandInteractionDataOption,
    )


def command_path(data: "ApplicationCommandInteractionData") -> str:
    """Returns the full invoked command path (such as `"memory server show"`) from `data`.

    Reads the payload's option tree directly because `interaction.application_command` is not
    populated until `invoke_callback_with_hooks` runs later.

    Args:
        data (ApplicationCommandInteractionData): Raw application-command interaction payload.
    """
    name = data.get("name")
    if not name:
        return ""
    subcommand_types = (
        ApplicationCommandOptionType.sub_command.value,
        ApplicationCommandOptionType.sub_command_group.value,
    )
    parts = [name]
    options = cast("list[ApplicationCommandInteractionDataOption]", data.get("options") or [])
    while options:
        nested = next((option for option in options if option["type"] in subcommand_types), None)
        if nested is None:
            break
        parts.append(nested["name"])
        options = cast(
            "list[ApplicationCommandInteractionDataOption]", nested.get("options") or []
        )
    return " ".join(parts)


class UsageCogs(commands.Cog):
    """Records each application-command invocation to the usage file.

    Attributes:
        bot (commands.Bot): The Discord bot instance that owns this cog.
        usage_recorder (UsageRecorder): Writer for the append-only usage records.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the cog with its own usage recorder.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot
        self.usage_recorder = UsageRecorder()

    @commands.Cog.listener()
    async def on_interaction(self, interaction: Interaction[commands.Bot]) -> None:
        """Records one slash-command invocation, ignoring components, modals, and autocomplete.

        Args:
            interaction (Interaction[commands.Bot]): The interaction Discord delivered.
        """
        if interaction.type is not InteractionType.application_command:
            return
        data = interaction.data
        user = interaction.user
        if data is None or user is None:
            return
        name = command_path(data=cast("ApplicationCommandInteractionData", data))
        if not name:
            return
        await self.usage_recorder.record(
            kind="slash",
            name=name,
            user_id=user.id,
            user_name=user.name,
            guild_id=interaction.guild_id,
            channel_id=interaction.channel_id,
        )


def setup(bot: commands.Bot) -> None:
    """Adds `UsageCogs` to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(UsageCogs(bot), override=True)
