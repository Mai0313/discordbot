"""Records every slash-command invocation so unused commands can be found later.

The whole Discord surface is one `on_interaction` listener: this cog registers no command, sends
nothing, and is invisible in the client. Every application-command interaction becomes one `slash`
record through `utils/usage_log.py` — the full invoked path plus the invoker, guild and channel
ids, and nothing else — gated only by that module's `USAGE_LOG_ENABLED`. The other half of the
picture, one record per AI reply turn, is written by `gen_reply/cog.py`; nothing here reads a
record back.

The listener is on `on_interaction`, not `on_application_command_completion`. The gateway
dispatches `interaction` for every INTERACTION_CREATE (`nextcord/state.py`) before
`process_application_commands` runs a single check or callback, so this one point sees an
invocation a permission check rejects, one whose command no longer resolves, and one whose
body raises — none of which reach the completion event. Someone pressing the button is the
demand signal this file exists to capture, so it must not be lost because the command is
broken; "failed or succeeded" is the runtime log's job.

It has to be a cog listener: overriding `on_interaction` on `DiscordBot` would replace
`Client.on_interaction` and stop slash commands executing altogether.
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

# The only two option types that continue the path. Every other option is an argument the user
# typed, so the walk stops there rather than folding a supplied value into the recorded name.
_SUBCOMMAND_TYPES = (
    ApplicationCommandOptionType.sub_command.value,
    ApplicationCommandOptionType.sub_command_group.value,
)


def command_path(data: "ApplicationCommandInteractionData") -> str:
    """The full invoked path (`memory server show`) read out of the interaction payload.

    `interaction.application_command` is still unset here — nextcord fills it in
    `invoke_callback_with_hooks`, which runs later — so the name comes from the payload's
    own option tree, where a subcommand (or a group holding one) is an option carrying the
    nested options below it. Recording the whole path keeps a subcommand its own unit;
    folding one back into its parent group is a `split` at analysis time, while the
    reverse is not recoverable.

    Args:
        data (ApplicationCommandInteractionData): The raw command payload off the interaction.

    Returns:
        The space-joined path, or "" for a payload that names no command.
    """
    name = data.get("name")
    if not name:
        return ""
    parts = [name]
    options = cast("list[ApplicationCommandInteractionDataOption]", data.get("options") or [])
    while options:
        nested = next((option for option in options if option["type"] in _SUBCOMMAND_TYPES), None)
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
        bot: The Discord bot instance that owns this cog.
        usage_recorder: Writer for the append-only usage records.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Builds the cog with its own usage recorder.

        A recorder of its own rather than one shared with `gen_reply`: it holds no state between
        records, and one cog may not reach into another.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot
        self.usage_recorder = UsageRecorder()

    @commands.Cog.listener()
    async def on_interaction(self, interaction: Interaction[commands.Bot]) -> None:
        """Records one invocation, ignoring components, modals and autocomplete.

        A payload naming no command, or one carrying no invoker, is dropped rather than recorded
        under an empty name: there would be nothing to attribute the use to. The write itself is
        best-effort inside `UsageRecorder.record`, so a recording failure never reaches the
        invocation being recorded.

        Args:
            interaction (Interaction[commands.Bot]): The interaction Discord delivered.
        """
        if interaction.type is not InteractionType.application_command:
            return
        data = interaction.data
        user = interaction.user
        if data is None or user is None:
            return
        # `Interaction.data` is the union of all three payload shapes; the type check above is
        # what makes narrowing it to the command one sound.
        name = command_path(data=cast("ApplicationCommandInteractionData", data))
        if not name:
            return
        await self.usage_recorder.record(
            kind="slash",
            name=name,
            user_id=user.id,
            guild_id=interaction.guild_id,
            channel_id=interaction.channel_id,
        )


def setup(bot: commands.Bot) -> None:
    """Adds the UsageCogs to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(UsageCogs(bot), override=True)
