"""Casino-style games (`/games blackjack`, `/games dragon_gate`) wagering economy points."""

from random import SystemRandom
from functools import partial, cached_property
from collections.abc import Callable

from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import Embed, Locale, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.games import (
    SystemIdentity,
    GameParticipant,
    BotFinancialContext,
    GameParticipantIdentity,
    RefreshParticipantsResult,
    ParticipantPreparationResult,
)
from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs._games.dealer import SystemNarrator
from discordbot.cogs._games.wagers import WagerMode, parse_wager_amount, build_wager_participant
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import (
    track_public_message,
    delete_tracked_public_messages,
    schedule_public_message_delete,
)
from discordbot.cogs._economy.database import get_account, get_balance, get_casino_daily_stats
from discordbot.cogs._games.bot_player import BotPlayerAI, fallback_bet
from discordbot.cogs._games.dragon_gate import ANTE
from discordbot.cogs._games.presentation import ERROR_COLOR, SYSTEM_NARRATOR_NAME
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
    """Slash commands for multiplayer casino games against the casino system.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: The LLM client configuration loaded for the system narrator.
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
        """The cached OpenAI-compatible client used for the system narrator and bot AI."""
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return client

    @cached_property
    def narrator(self) -> SystemNarrator:
        """The cached casino system narrator reused across game commands."""
        return SystemNarrator(client=self.client, model=self.runtime_models.fast_model)

    @cached_property
    def bot_player_ai(self) -> BotPlayerAI:
        """The cached bot-player decision AI shared across Blackjack tables.

        Uses `player_model` (pinned to `gemini-flash-latest`) so bot turns
        between human players stay snappy and do not inherit any future
        promotion of `slow_model` to a heavier Pro tier.
        """
        return BotPlayerAI(client=self.client, model=self.runtime_models.player_model)

    async def _system_identity(self, guild: nextcord.Guild | None = None) -> SystemIdentity:
        """Returns the casino system identity used for narrator embeds.

        Slash commands only fire after the gateway has connected, so
        `self.bot.user` is guaranteed non-None at call time. We still fall back
        to a synthetic id / SYSTEM_NARRATOR_NAME to keep type narrowing clean
        and to avoid blowing up the round if Discord briefly returns no client
        user (e.g. mid-reconnect).
        """
        if self.bot.user is None:
            return SystemIdentity(
                system_id=0, system_name=SYSTEM_NARRATOR_NAME, system_avatar_url=""
            )
        avatar_url = await guild_avatar_url(user=self.bot.user, guild=guild)
        return SystemIdentity(
            system_id=self.bot.user.id,
            system_name=SYSTEM_NARRATOR_NAME,
            system_avatar_url=avatar_url,
        )

    async def _bot_blackjack_participant(
        self,
        *,
        guild: nextcord.Guild | None,
        table_bet: int,
        other_player_bets: list[tuple[str, int]],
    ) -> GameParticipant | None:
        """Returns a Blackjack participant for the bot player, or None if it cannot join."""
        bot_user = self.bot.user
        if bot_user is None:
            return None
        account = await get_account(user_id=bot_user.id)
        balance = account.balance if account is not None else 0
        if balance <= 0:
            logfire.info(
                "Bot player skipped Blackjack lobby; wallet is empty", user_id=bot_user.id
            )
            return None
        daily_stats = await get_casino_daily_stats(user_id=bot_user.id)
        finance = BotFinancialContext(
            balance=balance,
            total_earned=account.total_earned if account is not None else 0,
            total_spent=account.total_spent if account is not None else 0,
            daily_loss=daily_stats.daily_loss,
            daily_win=daily_stats.daily_win,
            daily_net=daily_stats.daily_net,
        )
        try:
            decided_bet = await self.bot_player_ai.decide_bot_bet(
                finance=finance, table_bet=table_bet, other_player_bets=other_player_bets
            )
        except Exception:
            logfire.warn("Bot bet decision raised; using deterministic fallback", _exc_info=True)
            decided_bet = fallback_bet(balance=balance, table_bet=table_bet)
        avatar_url = await guild_avatar_url(user=bot_user, guild=guild)
        identity = GameParticipantIdentity(
            user_id=bot_user.id,
            account_name=bot_user.name,
            display_name=bot_user.display_name,
            avatar_url=avatar_url,
        )
        return build_wager_participant(
            identity=identity, balance=balance, wager=decided_bet, mode="clamp"
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
        self, participants: list[GameParticipant], mode: WagerMode
    ) -> RefreshParticipantsResult:
        """Re-checks balances against each queued participant wager."""
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
                wager=participant.bet,
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
                f"### {bold_currency(amount=balance, compact=True)}\n"
                f"沒有可下注的{CURRENCY_NAME}\n"
                f"-# 跟機器人聊天可以累積{CURRENCY_NAME}"
            ),
            color=ERROR_COLOR,
        )

    @staticmethod
    def _invalid_bet_embed() -> Embed:
        """Builds the validation embed for malformed bet input."""
        return Embed(
            title="下注格式錯誤",
            description="請輸入非負整數，可以加逗號，例如 `1,000`；輸入 `0` 會 all in。",
            color=ERROR_COLOR,
        )

    def _dragon_gate_insufficient_balance_embed(self, balance: int) -> Embed:
        """Builds the insufficient-balance embed for 射龍門 ante checks."""
        return Embed(
            title="餘額不足",
            description=(
                f"### {bold_currency(amount=balance, compact=True)}\n"
                f"射龍門入場費固定 {bold_currency(amount=ANTE, compact=True)} 進彩金池\n"
                f"-# 跟機器人聊天可以累積{CURRENCY_NAME}"
            ),
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
        description="Open a 21 lobby; the casino is the dealer and the bot joins as a player.",
        name_localizations={Locale.zh_TW: "二十一點", Locale.ja: "ブラックジャック"},
        description_localizations={
            Locale.zh_TW: "開一桌 21 點 lobby",
            Locale.ja: "21（ブラックジャック）の lobby を開きます。",
        },
    )
    async def blackjack(
        self,
        interaction: Interaction,
        bet: str = SlashOption(
            name="bet",
            description=f"Table stake in {CURRENCY_NAME}; enter 0 to go all in. Commas are allowed.",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: f"這桌的基本下注{CURRENCY_NAME}; 可加逗號，輸入 0 會直接 all in",
                Locale.ja: f"Table の基本賭け金{CURRENCY_NAME}; カンマ可、0 で all in。",
            },
            required=True,
            min_length=1,
        ),
    ) -> None:
        """Opens a Blackjack lobby. The owner starts the table from the lobby.

        Args:
            interaction: The interaction that triggered the command.
            bet: Raw wager text. Zero uses the owner's current balance.
        """
        if interaction.user is None:
            return
        wager = parse_wager_amount(raw_amount=bet)
        if wager is None:
            await interaction.response.send_message(
                embed=self._invalid_bet_embed(), ephemeral=True
            )
            return

        await interaction.response.defer()

        guild = getattr(interaction, "guild", None)
        if wager == 0:
            participant_result = await self._all_in_participant_from_user(
                user=interaction.user, guild=guild
            )
        else:
            participant_result = await self._participant_from_user(
                user=interaction.user, wager=wager, mode="clamp", guild=guild
            )
        owner = participant_result.participant
        if owner is None:
            message = await interaction.followup.send(
                embed=self._insufficient_balance_embed(balance=participant_result.balance),
                wait=True,
            )
            schedule_public_message_delete(message=message, user_name=interaction.user.name)
            return

        table_bet = owner.bet
        system_identity = await self._system_identity(guild=guild)
        bot_participant = await self._bot_blackjack_participant(
            guild=guild, table_bet=table_bet, other_player_bets=[(owner.display_name, owner.bet)]
        )
        extra_initial_participants: list[GameParticipant] = (
            [bot_participant] if bot_participant is not None else []
        )
        view = BlackjackLobbyView(
            owner=owner,
            requested_bet=table_bet,
            rng=self.rng,
            narrator=self.narrator,
            system_name=system_identity.system_name,
            system_avatar_url=system_identity.system_avatar_url,
            prepare_participant=partial(
                self._prepare_participant,
                wager=table_bet,
                mode="clamp",
                insufficient_embed_builder=self._insufficient_balance_embed,
            ),
            refresh_participants=partial(self._refresh_participants, mode="clamp"),
            bot_player_ai=self.bot_player_ai,
            bot_user_id=bot_participant.user_id if bot_participant is not None else None,
            extra_initial_participants=extra_initial_participants,
        )
        embed = build_blackjack_lobby_embed(
            owner=owner,
            participants=view.participants,
            requested_bet=table_bet,
            max_players=MAX_BLACKJACK_PLAYERS,
        )
        message = await interaction.followup.send(
            embed=embed,
            view=view,
            wait=True,
            **embed_spacer_payload(embeds=[embed], is_edit=False),
        )
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

        system_identity = await self._system_identity(guild=getattr(interaction, "guild", None))
        initial_jackpot = await fetch_dragon_gate_jackpot_snapshot()
        view = DragonGateLobbyView(
            owner=owner,
            rng=self.rng,
            narrator=self.narrator,
            system_name=system_identity.system_name,
            system_avatar_url=system_identity.system_avatar_url,
            prepare_participant=partial(
                self._prepare_participant,
                wager=ANTE,
                mode="exact",
                insufficient_embed_builder=self._dragon_gate_insufficient_balance_embed,
            ),
            refresh_participants=partial(self._refresh_participants, mode="exact"),
            initial_jackpot=initial_jackpot.balance,
            initial_jackpot_generation=initial_jackpot.generation,
        )
        embed = build_dragon_gate_lobby_embed(
            owner=owner, participants=view.participants, jackpot=initial_jackpot.balance
        )
        message = await interaction.followup.send(
            embed=embed,
            view=view,
            wait=True,
            **embed_spacer_payload(embeds=[embed], is_edit=False),
        )
        await track_public_message(message=message, user_name=owner.account_name)
        view.message = message


def setup(bot: commands.Bot) -> None:
    """Adds the GamesCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(GamesCogs(bot), override=True)
