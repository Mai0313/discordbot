"""Casino-style games (`/blackjack`, `/dragon_gate`) wagering economy points."""

from random import SystemRandom
from functools import cached_property

from openai import AsyncOpenAI
import nextcord
from nextcord import Embed, Locale, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.games import GameParticipant, GameParticipantIdentity
from discordbot.typings.models import ModelSettings
from discordbot.cogs._games.views import (
    MAX_BLACKJACK_PLAYERS,
    BlackjackLobbyView,
    build_blackjack_lobby_embed,
)
from discordbot.cogs._games.dealer import DealerAI
from discordbot.cogs._games.wagers import build_wager_participant
from discordbot.cogs._games.cleanup import (
    track_game_message,
    delete_tracked_game_messages,
    schedule_game_message_delete,
)
from discordbot.cogs._economy.database import get_balance
from discordbot.cogs._games.dragon_gate import ANTE
from discordbot.cogs._games.presentation import ERROR_COLOR
from discordbot.cogs._economy.presentation import CURRENCY_NAME, bold_currency
from discordbot.cogs._games.dragon_gate_views import (
    DragonGateLobbyView,
    fetch_dragon_gate_jackpot,
    build_dragon_gate_lobby_embed,
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

    @property
    def dealer_model(self) -> ModelSettings:
        """The model settings used by the AI dealer.

        Returns:
            Fast model settings with reasoning disabled for dealer banter.
        """
        return ModelSettings(name="gemini-flash-latest", effort="none")

    @cached_property
    def dealer(self) -> DealerAI:
        """The cached AI dealer reused across game commands.

        Returns:
            A DealerAI built from the cached client and dealer model settings.
        """
        return DealerAI(client=self.client, model=self.dealer_model)

    def _dealer_identity(self) -> tuple[int, str, str]:
        """Returns ``(dealer_id, dealer_name, dealer_avatar_url)`` from the bot user.

        Slash commands only fire after the gateway has connected, so
        ``self.bot.user`` is guaranteed non-None at call time. We still fall
        back to a synthetic id / "莊家" name to keep type-narrowing clean and
        to avoid blowing up the round if Discord briefly returns no client
        user (e.g. mid-reconnect).
        """
        if self.bot.user is None:
            return (0, "莊家", "")
        return (self.bot.user.id, self.bot.user.display_name, self.bot.user.display_avatar.url)

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Deletes stale game messages left by a previous bot process."""
        if self._startup_cleanup_done:
            return
        self._startup_cleanup_done = True
        await delete_tracked_game_messages(bot=self.bot)

    @staticmethod
    def _identity_from_user(user: nextcord.User | nextcord.Member) -> GameParticipantIdentity:
        """Builds the shared game identity for a Discord user."""
        return GameParticipantIdentity(
            user_id=user.id,
            account_name=user.name,
            display_name=user.display_name,
            avatar_url=user.display_avatar.url,
        )

    async def _participant_from_user(
        self, user: nextcord.User | nextcord.Member, requested_bet: int
    ) -> tuple[GameParticipant | None, int]:
        """Builds a lobby participant after clamping the requested bet."""
        balance = await get_balance(user_id=user.id)
        return (
            build_wager_participant(
                identity=self._identity_from_user(user=user),
                balance=balance,
                wager=requested_bet,
                mode="clamp",
            ),
            balance,
        )

    async def _prepare_lobby_participant(
        self, interaction: Interaction, requested_bet: int
    ) -> GameParticipant | None:
        """Prepares a user who pressed the lobby Join button."""
        if interaction.user is None:
            return None
        participant, balance = await self._participant_from_user(
            user=interaction.user, requested_bet=requested_bet
        )
        if participant is None:
            await interaction.followup.send(
                embed=self._insufficient_balance_embed(balance=balance), ephemeral=True
            )
        return participant

    async def _dragon_gate_participant_from_user(
        self, user: nextcord.User | nextcord.Member
    ) -> tuple[GameParticipant | None, int]:
        """Builds a 射龍門 lobby participant after checking the fixed ante."""
        balance = await get_balance(user_id=user.id)
        return (
            build_wager_participant(
                identity=self._identity_from_user(user=user),
                balance=balance,
                wager=ANTE,
                mode="exact",
            ),
            balance,
        )

    async def _prepare_dragon_gate_participant(
        self, interaction: Interaction
    ) -> GameParticipant | None:
        """Prepares a user who pressed the 射龍門 lobby Join button."""
        if interaction.user is None:
            return None
        participant, balance = await self._dragon_gate_participant_from_user(user=interaction.user)
        if participant is None:
            await interaction.followup.send(
                embed=self._dragon_gate_insufficient_balance_embed(balance=balance), ephemeral=True
            )
        return participant

    async def _refresh_dragon_gate_participants(
        self, participants: list[GameParticipant]
    ) -> tuple[list[GameParticipant], list[str]]:
        """Re-checks 射龍門 ante balances when the lobby owner starts the table."""
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
                wager=ANTE,
                mode="exact",
            )
            if refreshed_participant is None:
                dropped.append(participant.display_name)
                continue
            refreshed.append(refreshed_participant)
        return refreshed, dropped

    async def _refresh_lobby_participants(
        self, participants: list[GameParticipant], requested_bet: int
    ) -> tuple[list[GameParticipant], list[str]]:
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
                wager=requested_bet,
                mode="clamp",
            )
            if refreshed_participant is None:
                dropped.append(participant.display_name)
                continue
            refreshed.append(refreshed_participant)
        return refreshed, dropped

    def _insufficient_balance_embed(self, balance: int) -> Embed:
        """Builds the shared insufficient-balance embed for casino games."""
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

    @nextcord.slash_command(
        name="blackjack",
        description="Open a 21 lobby against the dealer.",
        name_localizations={Locale.zh_TW: "二十一點", Locale.ja: "ブラックジャック"},
        description_localizations={
            Locale.zh_TW: "開一桌 21 點 lobby",
            Locale.ja: "21（ブラックジャック）の lobby を開きます。",
        },
        nsfw=False,
    )
    async def blackjack(
        self,
        interaction: Interaction,
        bet: int = SlashOption(
            name="bet",
            description=f"Table stake in {CURRENCY_NAME}; over-betting opens at your balance.",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: f"這桌的基本下注{CURRENCY_NAME} (超過餘額會用你的餘額開桌)",
                Locale.ja: f"Table の基本賭け金{CURRENCY_NAME} (残高超過なら残高で開きます)。",
            },
            required=True,
            min_value=1,
        ),
    ) -> None:
        """Opens a Blackjack lobby. The owner starts the table from the lobby.

        Args:
            interaction: The interaction that triggered the command.
            bet: How many points to wager.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return

        owner, balance = await self._participant_from_user(
            user=interaction.user, requested_bet=bet
        )
        if owner is None:
            message = await interaction.followup.send(
                embed=self._insufficient_balance_embed(balance=balance), wait=True
            )
            schedule_game_message_delete(message=message)
            return

        table_bet = owner.bet
        dealer_id, dealer_name, dealer_avatar_url = self._dealer_identity()
        view = BlackjackLobbyView(
            owner=owner,
            requested_bet=table_bet,
            rng=self.rng,
            dealer=self.dealer,
            dealer_id=dealer_id,
            dealer_name=dealer_name,
            dealer_avatar_url=dealer_avatar_url,
            prepare_participant=self._prepare_lobby_participant,
            refresh_participants=self._refresh_lobby_participants,
        )
        embed = build_blackjack_lobby_embed(
            owner=owner,
            participants=view.participants,
            requested_bet=table_bet,
            max_players=MAX_BLACKJACK_PLAYERS,
        )
        message = await interaction.followup.send(embed=embed, view=view, wait=True)
        await track_game_message(message=message)
        view.message = message

    @nextcord.slash_command(
        name="dragon_gate",
        description="Open an In-Between table with a shared jackpot pool.",
        name_localizations={Locale.zh_TW: "射龍門", Locale.ja: "インビトウィーン"},
        description_localizations={
            Locale.zh_TW: "開一桌共享全域彩金池的射龍門",
            Locale.ja: "共有ジャックポットのインビトウィーン table を開きます。",
        },
        nsfw=False,
    )
    async def dragon_gate(self, interaction: Interaction) -> None:
        """Opens a 射龍門 lobby. The owner starts the table from the lobby.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return

        owner, balance = await self._dragon_gate_participant_from_user(user=interaction.user)
        if owner is None:
            message = await interaction.followup.send(
                embed=self._dragon_gate_insufficient_balance_embed(balance=balance), wait=True
            )
            schedule_game_message_delete(message=message)
            return

        _dealer_id, dealer_name, dealer_avatar_url = self._dealer_identity()
        initial_jackpot = await fetch_dragon_gate_jackpot()
        view = DragonGateLobbyView(
            owner=owner,
            rng=self.rng,
            dealer=self.dealer,
            dealer_name=dealer_name,
            dealer_avatar_url=dealer_avatar_url,
            prepare_participant=self._prepare_dragon_gate_participant,
            refresh_participants=self._refresh_dragon_gate_participants,
            initial_jackpot=initial_jackpot,
        )
        embed = build_dragon_gate_lobby_embed(
            owner=owner, participants=view.participants, jackpot=initial_jackpot
        )
        message = await interaction.followup.send(embed=embed, view=view, wait=True)
        await track_game_message(message=message)
        view.message = message


def setup(bot: commands.Bot) -> None:
    """Adds the GamesCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(GamesCogs(bot), override=True)
