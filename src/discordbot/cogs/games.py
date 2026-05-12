"""Casino-style games (`/dice`, `/blackjack`) wagering economy points."""

from random import SystemRandom
import asyncio
from functools import cached_property

from openai import AsyncOpenAI
import nextcord
from nextcord import Embed, Locale, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.typings.economy import PreparedBet
from discordbot.cogs._games.dice import play_dice, render_rolls
from discordbot.cogs._games.views import BlackjackView, build_final_embed, build_in_progress_embed
from discordbot.cogs._games.dealer import DealerAI
from discordbot.cogs._games.cleanup import (
    track_game_message,
    delete_tracked_game_messages,
    schedule_game_message_delete,
)
from discordbot.cogs._games.blackjack import BlackjackHand
from discordbot.cogs._economy.database import get_balance
from discordbot.cogs._games.settlement import (
    settle_wager,
    settle_blackjack_round,
    blackjack_early_finish_note,
)
from discordbot.cogs._games.dragon_gate import (
    play_dragon_gate,
    render_card_value,
    dragon_gate_detail,
)
from discordbot.cogs._games.presentation import (
    ERROR_COLOR,
    IN_PROGRESS_COLOR,
    duel_lines,
    dealer_quote,
    wager_footer,
    settlement_footer,
    dice_outcome_presentation,
    blackjack_outcome_presentation,
    dragon_gate_outcome_presentation,
)
from discordbot.cogs._economy.presentation import CURRENCY_NAME, bold_currency

# Short pause between the "place your bet" embed and the dice reveal so the
# moment lands. Long enough to feel deliberate, short enough not to annoy.
_DICE_REVEAL_DELAY_SECONDS = 1.5


class GamesCogs(commands.Cog):
    """Slash commands for casino games against an AI dealer.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: The LLM client configuration loaded for dealer banter.
        rng: System randomness used for dice rolls and card draws.
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

    async def _prepare_bet(
        self, *, interaction: Interaction, requested_bet: int
    ) -> PreparedBet | None:
        """Checks the effective bet or sends an insufficient-balance embed.

        Bets are settled only when a round finishes. If the bot restarts during
        an in-memory round, no balance mutation has happened yet.
        """
        if interaction.user is None:
            return None
        balance = await get_balance(user_id=interaction.user.id)
        if requested_bet <= 0 or balance <= 0:
            message = await interaction.followup.send(
                embed=Embed(
                    title="餘額不足",
                    description=(
                        f"### {bold_currency(amount=balance)}\n"
                        f"沒有可下注的{CURRENCY_NAME}\n"
                        f"-# 跟機器人聊天可以累積{CURRENCY_NAME}"
                    ),
                    color=ERROR_COLOR,
                ),
                wait=True,
            )
            schedule_game_message_delete(message=message)
            return None
        return PreparedBet(
            amount=min(requested_bet, balance),
            balance_at_start=balance,
            is_allin=requested_bet > balance,
        )

    @nextcord.slash_command(
        name="dice",
        description="Roll three dice against the dealer; whoever totals higher wins.",
        name_localizations={Locale.zh_TW: "比大小", Locale.ja: "サイコロ勝負"},
        description_localizations={
            Locale.zh_TW: "用三顆骰子跟莊家比點數總和, 大的贏",
            Locale.ja: "3個のサイコロで親と勝負し、合計が大きい方が勝ち。",
        },
        nsfw=False,
    )
    async def dice(
        self,
        interaction: Interaction,
        bet: int = SlashOption(
            name="bet",
            description=f"How much {CURRENCY_NAME} to wager (auto all-ins if over your balance).",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: f"下注的{CURRENCY_NAME} (超過餘額會自動 all-in)",
                Locale.ja: f"賭ける{CURRENCY_NAME} (残高を超えると自動 all-in)。",
            },
            required=True,
            min_value=1,
        ),
    ) -> None:
        """Plays one round of player-vs-dealer compare-the-total dice.

        Args:
            interaction: The interaction that triggered the command.
            bet: How many points to wager.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return

        prepared_bet = await self._prepare_bet(interaction=interaction, requested_bet=bet)
        if prepared_bet is None:
            return
        bet = prepared_bet.amount
        is_allin = prepared_bet.is_allin

        dealer_id, dealer_name, dealer_avatar_url = self._dealer_identity()

        taunt = await self.dealer.taunt_bet(
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            balance_at_start=prepared_bet.balance_at_start,
            bet=bet,
            game="dice",
        )

        in_progress = Embed(
            title="🎲 比大小", description=dealer_quote(text=taunt), color=IN_PROGRESS_COLOR
        )
        in_progress.set_author(
            name=f"{interaction.user.display_name} 的對局",
            icon_url=interaction.user.display_avatar.url,
        )
        in_progress.set_footer(
            text=wager_footer(
                bet=bet,
                balance_at_start=prepared_bet.balance_at_start,
                is_allin=is_allin,
                status="等候開獎",
            )
        )
        message = await interaction.followup.send(embed=in_progress, wait=True)
        await track_game_message(message=message)

        await asyncio.sleep(delay=_DICE_REVEAL_DELAY_SECONDS)
        result = play_dice(rng=self.rng)

        if result.outcome == "win":
            delta = bet
        elif result.outcome == "push":
            delta = 0
        else:
            delta = -bet

        settlement = await settle_wager(
            player_id=interaction.user.id,
            player_account_name=interaction.user.name,
            player_avatar_url=interaction.user.display_avatar.url,
            dealer_id=dealer_id,
            dealer_name=dealer_name,
            dealer_avatar_url=dealer_avatar_url,
            bet=bet,
            delta=delta,
        )
        detail = f"玩家骰 {result.player_total} 點, 莊家骰 {result.dealer_total} 點"
        banter = await self.dealer.settle(
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            outcome=result.outcome,
            bet=bet,
            delta=settlement.delta,
            new_balance=settlement.new_balance,
            game="dice",
            detail=detail,
        )
        outcome_label, color = dice_outcome_presentation(outcome=result.outcome)

        final = Embed(
            title=f"🎲 比大小 | {outcome_label}",
            description=dealer_quote(text=banter),
            color=color,
        )
        final.set_author(
            name=f"{interaction.user.display_name} 的對局",
            icon_url=interaction.user.display_avatar.url,
        )
        final.add_field(
            name="擲骰",
            value=duel_lines(
                player_name=interaction.user.display_name,
                player_value=render_rolls(rolls=result.player_rolls),
                dealer_name=dealer_name,
                dealer_value=render_rolls(rolls=result.dealer_rolls),
            ),
            inline=False,
        )
        final.set_footer(
            text=settlement_footer(
                delta=settlement.delta, new_balance=settlement.new_balance, is_allin=is_allin
            )
        )
        await message.edit(embed=final)
        schedule_game_message_delete(message=message)

    @nextcord.slash_command(
        name="dragon_gate",
        description="Shoot one card between two gate cards; inside wins.",
        name_localizations={Locale.zh_TW: "射龍門", Locale.ja: "ドラゴンゲート"},
        description_localizations={
            Locale.zh_TW: "下注後開兩張門牌, 第三張嚴格落在中間就贏",
            Locale.ja: "2枚のゲートカードの間に3枚目が入れば勝ち。",
        },
        nsfw=False,
    )
    async def dragon_gate(
        self,
        interaction: Interaction,
        bet: int = SlashOption(
            name="bet",
            description=f"How much {CURRENCY_NAME} to wager (auto all-ins if over your balance).",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: f"下注的{CURRENCY_NAME} (超過餘額會自動 all-in)",
                Locale.ja: f"賭ける{CURRENCY_NAME} (残高を超えると自動 all-in)。",
            },
            required=True,
            min_value=1,
        ),
    ) -> None:
        """Plays one round of Dragon Gate.

        Args:
            interaction: The interaction that triggered the command.
            bet: How many points to wager.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return

        prepared_bet = await self._prepare_bet(interaction=interaction, requested_bet=bet)
        if prepared_bet is None:
            return
        bet = prepared_bet.amount
        is_allin = prepared_bet.is_allin

        dealer_id, dealer_name, dealer_avatar_url = self._dealer_identity()

        taunt = await self.dealer.taunt_bet(
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            balance_at_start=prepared_bet.balance_at_start,
            bet=bet,
            game="dragon_gate",
        )

        in_progress = Embed(
            title="🎴 射龍門", description=dealer_quote(text=taunt), color=IN_PROGRESS_COLOR
        )
        in_progress.set_author(
            name=f"{interaction.user.display_name} 的對局",
            icon_url=interaction.user.display_avatar.url,
        )
        in_progress.set_footer(
            text=wager_footer(
                bet=bet,
                balance_at_start=prepared_bet.balance_at_start,
                is_allin=is_allin,
                status="等候開門",
            )
        )
        message = await interaction.followup.send(embed=in_progress, wait=True)
        await track_game_message(message=message)

        await asyncio.sleep(delay=_DICE_REVEAL_DELAY_SECONDS)
        result = play_dragon_gate(rng=self.rng)

        if result.outcome == "win":
            delta = bet
        elif result.outcome == "push":
            delta = 0
        else:
            delta = -bet

        settlement = await settle_wager(
            player_id=interaction.user.id,
            player_account_name=interaction.user.name,
            player_avatar_url=interaction.user.display_avatar.url,
            dealer_id=dealer_id,
            dealer_name=dealer_name,
            dealer_avatar_url=dealer_avatar_url,
            bet=bet,
            delta=delta,
        )
        detail = dragon_gate_detail(result=result)
        banter = await self.dealer.settle(
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            outcome=result.outcome,
            bet=bet,
            delta=settlement.delta,
            new_balance=settlement.new_balance,
            game="dragon_gate",
            detail=detail,
        )
        outcome_label, color = dragon_gate_outcome_presentation(outcome=result.outcome)

        final = Embed(
            title=f"🎴 射龍門 | {outcome_label}",
            description=dealer_quote(text=banter),
            color=color,
        )
        final.set_author(
            name=f"{interaction.user.display_name} 的對局",
            icon_url=interaction.user.display_avatar.url,
        )
        final.add_field(
            name="牌面",
            value=(
                f"**龍門**\n"
                f"{render_card_value(card=result.lower_gate)}  ~  "
                f"{render_card_value(card=result.upper_gate)}\n\n"
                f"**{interaction.user.display_name} 射門**\n"
                f"{render_card_value(card=result.shot)}"
            ),
            inline=False,
        )
        final.set_footer(
            text=settlement_footer(
                delta=settlement.delta, new_balance=settlement.new_balance, is_allin=is_allin
            )
        )
        await message.edit(embed=final)
        schedule_game_message_delete(message=message)

    @nextcord.slash_command(
        name="blackjack",
        description="Play one round of 21 against the dealer.",
        name_localizations={Locale.zh_TW: "二十一點", Locale.ja: "ブラックジャック"},
        description_localizations={
            Locale.zh_TW: "跟莊家玩一局 21 點",
            Locale.ja: "親と21（ブラックジャック）を1ラウンド遊びます。",
        },
        nsfw=False,
    )
    async def blackjack(
        self,
        interaction: Interaction,
        bet: int = SlashOption(
            name="bet",
            description=f"How much {CURRENCY_NAME} to wager (auto all-ins if over your balance).",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: f"下注的{CURRENCY_NAME} (超過餘額會自動 all-in)",
                Locale.ja: f"賭ける{CURRENCY_NAME} (残高を超えると自動 all-in)。",
            },
            required=True,
            min_value=1,
        ),
    ) -> None:
        """Starts one Blackjack hand. The Hit/Stand view drives the rest of the round.

        Args:
            interaction: The interaction that triggered the command.
            bet: How many points to wager.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return

        prepared_bet = await self._prepare_bet(interaction=interaction, requested_bet=bet)
        if prepared_bet is None:
            return
        bet = prepared_bet.amount
        is_allin = prepared_bet.is_allin

        dealer_id, dealer_name, dealer_avatar_url = self._dealer_identity()

        hand = BlackjackHand(rng=self.rng, bet=bet)
        hand.deal_initial()

        taunt = await self.dealer.taunt_bet(
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            balance_at_start=prepared_bet.balance_at_start,
            bet=bet,
            game="blackjack",
        )

        # Natural Blackjack (or dealer Blackjack) ends the hand before the
        # player gets to act; settle and post the final embed straight away.
        if hand.finished:
            settlement = await settle_blackjack_round(
                hand=hand,
                player_id=interaction.user.id,
                player_account_name=interaction.user.name,
                player_avatar_url=interaction.user.display_avatar.url,
                dealer_id=dealer_id,
                dealer_name=dealer_name,
                dealer_avatar_url=dealer_avatar_url,
            )
            banter = await self.dealer.settle(
                author_name=interaction.user.name,
                player_name=interaction.user.display_name,
                outcome=settlement.outcome,
                bet=bet,
                delta=settlement.delta,
                new_balance=settlement.new_balance,
                game="blackjack",
                detail=settlement.detail,
            )
            outcome_label, color = blackjack_outcome_presentation(outcome=settlement.outcome)
            embed = build_final_embed(
                dealer_name=dealer_name,
                player_name=interaction.user.display_name,
                player_avatar_url=interaction.user.display_avatar.url,
                hand=hand,
                delta=settlement.delta,
                new_balance=settlement.new_balance,
                dealer_line=banter,
                outcome_label=outcome_label,
                color=color,
                is_allin=is_allin,
                round_note=blackjack_early_finish_note(hand=hand),
            )
            message = await interaction.followup.send(embed=embed, wait=True)
            await track_game_message(message=message)
            schedule_game_message_delete(message=message)
            return

        view = BlackjackView(
            dealer=self.dealer,
            hand=hand,
            owner_id=interaction.user.id,
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            player_avatar_url=interaction.user.display_avatar.url,
            dealer_id=dealer_id,
            dealer_name=dealer_name,
            dealer_avatar_url=dealer_avatar_url,
            balance_at_start=prepared_bet.balance_at_start,
            is_allin=is_allin,
        )
        embed = build_in_progress_embed(
            dealer_name=dealer_name,
            player_name=interaction.user.display_name,
            player_avatar_url=interaction.user.display_avatar.url,
            hand=hand,
            balance_at_start=prepared_bet.balance_at_start,
            dealer_line=taunt,
            is_allin=is_allin,
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
