"""Casino-style games (`/games blackjack`, `/games dragon_gate`) wagering economy points."""

from random import SystemRandom
from functools import partial, cached_property
from collections.abc import Callable

from openai import AsyncOpenAI
import nextcord
from nextcord import Embed, Locale, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.games import (
    DealerIdentity,
    GameParticipant,
    GameParticipantIdentity,
    RefreshParticipantsResult,
    ParticipantPreparationResult,
)
from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs._games.dealer import DealerAI
from discordbot.cogs._games.wagers import WagerMode, build_wager_participant
from discordbot.utils.message_cleanup import (
    track_public_message,
    delete_tracked_public_messages,
    schedule_public_message_delete,
)
from discordbot.cogs._economy.database import get_balance
from discordbot.cogs._games.dragon_gate import ANTE
from discordbot.cogs._games.presentation import ERROR_COLOR
from discordbot.cogs._economy.presentation import CURRENCY_NAME, bold_currency
from discordbot.cogs._games.blackjack_views import (
    MAX_BLACKJACK_PLAYERS,
    BlackjackLobbyView,
    build_blackjack_lobby_embed,
)
from discordbot.cogs._games.dragon_gate_views import (
    DragonGateLobbyView,
    build_dragon_gate_lobby_embed,
    fetch_dragon_gate_jackpot_snapshot,
)


class GamesCogs(commands.Cog):
    """Slash commands for multiplayer casino games against an AI dealer.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: The LLM client configuration loaded for dealer banter.
        rng: System randomness used for card draws.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initialises the GamesCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()
        self.runtime_models = RuntimeModelCatalog()
        self.rng = SystemRandom()
        self._startup_cleanup_done = False

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The cached OpenAI-compatible client used for dealer banter.

        Returns:
            A configured client reused by the AI dealer.
        """
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return client

    @cached_property
    def dealer(self) -> DealerAI:
        """The cached AI dealer reused across game commands.

        Returns:
            A DealerAI built from the cached client and dealer model settings.
        """
        return DealerAI(client=self.client, model=self.runtime_models.fast_model)

    async def _dealer_identity(self, guild: nextcord.Guild | None = None) -> DealerIdentity:
        """Returns the dealer identity from the bot user.

        Slash commands only fire after the gateway has connected, so
        `self.bot.user` is guaranteed non-None at call time. We still fall
        back to a synthetic id / "莊家" name to keep type-narrowing clean and
        to avoid blowing up the round if Discord briefly returns no client
        user (e.g. mid-reconnect).
        """
        if self.bot.user is None:
            return DealerIdentity(dealer_id=0, dealer_name="莊家", dealer_avatar_url="")
        avatar_url = await guild_avatar_url(user=self.bot.user, guild=guild)
        return DealerIdentity(
            dealer_id=self.bot.user.id,
            dealer_name=self.bot.user.display_name,
            dealer_avatar_url=avatar_url,
        )

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Deletes stale public messages left by a previous bot process."""
        if self._startup_cleanup_done:
            return
        self._startup_cleanup_done = True
        await delete_tracked_public_messages(bot=self.bot)

    @staticmethod
    async def _identity_from_user(
        user: nextcord.User | nextcord.Member, guild: nextcord.Guild | None = None
    ) -> GameParticipantIdentity:
        """Builds the shared game identity for a Discord user."""
        avatar_url = await guild_avatar_url(user=user, guild=guild)
        return GameParticipantIdentity(
            user_id=user.id,
            account_name=user.name,
            display_name=user.display_name,
            avatar_url=avatar_url,
        )

    async def _participant_from_user(
        self,
        user: nextcord.User | nextcord.Member,
        wager: int,
        mode: WagerMode,
        guild: nextcord.Guild | None = None,
    ) -> ParticipantPreparationResult:
        """Builds a lobby participant under the requested wager and mode."""
        balance = await get_balance(user_id=user.id)
        return ParticipantPreparationResult(
            participant=build_wager_participant(
                identity=await self._identity_from_user(user=user, guild=guild),
                balance=balance,
                wager=wager,
                mode=mode,
            ),
            balance=balance,
        )

    async def _all_in_participant_from_user(
        self, user: nextcord.User | nextcord.Member, guild: nextcord.Guild | None = None
    ) -> ParticipantPreparationResult:
        """Builds a clamp-mode participant using the user's current full balance."""
        balance = await get_balance(user_id=user.id)
        return ParticipantPreparationResult(
            participant=build_wager_participant(
                identity=await self._identity_from_user(user=user, guild=guild),
                balance=balance,
                wager=balance,
                mode="clamp",
            ),
            balance=balance,
        )

    async def _prepare_participant(
        self,
        interaction: Interaction,
        wager: int,
        mode: WagerMode,
        insufficient_embed_builder: Callable[[int], Embed],
    ) -> GameParticipant | None:
        """Prepares a user who pressed a lobby Join button.

        Sends the supplied insufficient-balance embed (game-specific copy) when
        the user cannot cover the wager. Returns the participant otherwise.
        """
        if interaction.user is None:
            return None
        result = await self._participant_from_user(
            user=interaction.user,
            wager=wager,
            mode=mode,
            guild=getattr(interaction, "guild", None),
        )
        if result.participant is None:
            await interaction.followup.send(
                embed=insufficient_embed_builder(result.balance), ephemeral=True
            )
        return result.participant

    async def _refresh_participants(
        self, participants: list[GameParticipant], wager: int, mode: WagerMode
    ) -> RefreshParticipantsResult:
        """Re-checks balances when the lobby owner starts the table."""
        refreshed: list[GameParticipant] = []
        dropped: list[str] = []
        for participant in participants:
            balance = await get_balance(user_id=participant.user_id)
            refreshed_participant = build_wager_participant(
                identity=GameParticipantIdentity(
                    user_id=participant.user_id,
                    account_name=participant.account_name,
                    display_name=participant.display_name,
                    avatar_url=participant.avatar_url,
                ),
                balance=balance,
                wager=wager,
                mode=mode,
            )
            if refreshed_participant is None:
                dropped.append(participant.display_name)
                continue
            refreshed.append(refreshed_participant)
        return RefreshParticipantsResult(participants=refreshed, dropped_names=dropped)

    def _insufficient_balance_embed(self, balance: int) -> Embed:
        """Builds the shared insufficient-balance embed for clamp-mode tables."""
        return Embed(
            title="餘額不足",
            description=(
                f"### {bold_currency(amount=balance)}\n"
                f"沒有可下注的{CURRENCY_NAME}\n"
                f"-# 跟機器人聊天可以累積{CURRENCY_NAME}"
            ),
            color=ERROR_COLOR,
        )

    def _dragon_gate_insufficient_balance_embed(self, balance: int) -> Embed:
        """Builds the insufficient-balance embed for 射龍門 ante checks."""
        return Embed(
            title="餘額不足",
            description=(
                f"### {bold_currency(amount=balance)}\n"
                f"射龍門入場費固定 {bold_currency(amount=ANTE)} 進彩金池\n"
                f"-# 跟機器人聊天可以累積{CURRENCY_NAME}"
            ),
            color=ERROR_COLOR,
        )

    def _blackjack_missing_bet_embed(self) -> Embed:
        """Builds the validation embed for missing Blackjack wager options."""
        return Embed(
            title="缺少下注",
            description="請輸入 `bet`, 或把 `all_in` 設成 `true` 直接用目前餘額開桌",
            color=ERROR_COLOR,
        )

    @nextcord.slash_command(
        name="games",
        description="Game commands.",
        name_localizations={Locale.zh_TW: "小遊戲", Locale.ja: "ゲーム"},
        description_localizations={Locale.zh_TW: "小遊戲指令", Locale.ja: "ゲームコマンド。"},
        nsfw=False,
    )
    async def games(self, interaction: Interaction) -> None:
        """Slash command group for casino games."""

    @games.subcommand(
        name="blackjack",
        description="Open a 21 lobby against the dealer.",
        name_localizations={Locale.zh_TW: "二十一點", Locale.ja: "ブラックジャック"},
        description_localizations={
            Locale.zh_TW: "開一桌 21 點 lobby",
            Locale.ja: "21（ブラックジャック）の lobby を開きます。",
        },
    )
    async def blackjack(
        self,
        interaction: Interaction,
        bet: int | None = SlashOption(
            name="bet",
            description=f"Table stake in {CURRENCY_NAME}; omit when all_in is true.",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: f"這桌的基本下注{CURRENCY_NAME}; all_in=true 時可省略",
                Locale.ja: f"Table の基本賭け金{CURRENCY_NAME}; all_in=true の場合は省略できます。",
            },
            required=False,
            default=None,
            min_value=1,
        ),
        all_in: bool = SlashOption(
            name="all_in",
            description="Use your current balance as the Blackjack table stake.",
            name_localizations={Locale.zh_TW: "all_in", Locale.ja: "オールイン"},
            description_localizations={
                Locale.zh_TW: "不用輸入大額數字, 直接用目前餘額開桌",
                Locale.ja: "大きな数値を入力せず、現在の残高で table を開きます。",
            },
            required=False,
            default=False,
        ),
    ) -> None:
        """Opens a Blackjack lobby. The owner starts the table from the lobby.

        Args:
            interaction: The interaction that triggered the command.
            bet: How many points to wager, unless `all_in` is selected.
            all_in: Whether to use the owner's current balance as the wager.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return

        bet_value = bet if isinstance(bet, int) else None
        all_in_requested = all_in is True
        guild = getattr(interaction, "guild", None)
        if all_in_requested:
            participant_result = await self._all_in_participant_from_user(
                user=interaction.user, guild=guild
            )
        elif bet_value is not None:
            participant_result = await self._participant_from_user(
                user=interaction.user, wager=bet_value, mode="clamp", guild=guild
            )
        else:
            message = await interaction.followup.send(
                embed=self._blackjack_missing_bet_embed(), wait=True
            )
            schedule_public_message_delete(message=message, user_name=interaction.user.name)
            return
        owner = participant_result.participant
        if owner is None:
            message = await interaction.followup.send(
                embed=self._insufficient_balance_embed(balance=participant_result.balance),
                wait=True,
            )
            schedule_public_message_delete(message=message, user_name=interaction.user.name)
            return

        table_bet = owner.bet
        dealer_identity = await self._dealer_identity(guild=getattr(interaction, "guild", None))
        view = BlackjackLobbyView(
            owner=owner,
            requested_bet=table_bet,
            rng=self.rng,
            dealer=self.dealer,
            dealer_id=dealer_identity.dealer_id,
            dealer_name=dealer_identity.dealer_name,
            dealer_avatar_url=dealer_identity.dealer_avatar_url,
            prepare_participant=partial(
                self._prepare_participant,
                wager=table_bet,
                mode="clamp",
                insufficient_embed_builder=self._insufficient_balance_embed,
            ),
            refresh_participants=partial(
                self._refresh_participants, wager=table_bet, mode="clamp"
            ),
        )
        embed = build_blackjack_lobby_embed(
            owner=owner,
            participants=view.participants,
            requested_bet=table_bet,
            max_players=MAX_BLACKJACK_PLAYERS,
        )
        message = await interaction.followup.send(embed=embed, view=view, wait=True)
        await track_public_message(message=message, user_name=owner.account_name)
        view.message = message

    @games.subcommand(
        name="dragon_gate",
        description="Open an In-Between table with a shared jackpot pool.",
        name_localizations={Locale.zh_TW: "射龍門", Locale.ja: "インビトウィーン"},
        description_localizations={
            Locale.zh_TW: "開一桌共享全域彩金池的射龍門",
            Locale.ja: "共有ジャックポットのインビトウィーン table を開きます。",
        },
    )
    async def dragon_gate(self, interaction: Interaction) -> None:
        """Opens a 射龍門 lobby. The owner starts the table from the lobby.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return

        participant_result = await self._participant_from_user(
            user=interaction.user,
            wager=ANTE,
            mode="exact",
            guild=getattr(interaction, "guild", None),
        )
        owner = participant_result.participant
        if owner is None:
            message = await interaction.followup.send(
                embed=self._dragon_gate_insufficient_balance_embed(
                    balance=participant_result.balance
                ),
                wait=True,
            )
            schedule_public_message_delete(message=message, user_name=interaction.user.name)
            return

        dealer_identity = await self._dealer_identity(guild=getattr(interaction, "guild", None))
        initial_jackpot = await fetch_dragon_gate_jackpot_snapshot()
        view = DragonGateLobbyView(
            owner=owner,
            rng=self.rng,
            dealer=self.dealer,
            dealer_name=dealer_identity.dealer_name,
            dealer_avatar_url=dealer_identity.dealer_avatar_url,
            prepare_participant=partial(
                self._prepare_participant,
                wager=ANTE,
                mode="exact",
                insufficient_embed_builder=self._dragon_gate_insufficient_balance_embed,
            ),
            refresh_participants=partial(self._refresh_participants, wager=ANTE, mode="exact"),
            initial_jackpot=initial_jackpot.balance,
            initial_jackpot_generation=initial_jackpot.generation,
        )
        embed = build_dragon_gate_lobby_embed(
            owner=owner, participants=view.participants, jackpot=initial_jackpot.balance
        )
        message = await interaction.followup.send(embed=embed, view=view, wait=True)
        await track_public_message(message=message, user_name=owner.account_name)
        view.message = message


def setup(bot: commands.Bot) -> None:
    """Adds the GamesCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(GamesCogs(bot), override=True)
