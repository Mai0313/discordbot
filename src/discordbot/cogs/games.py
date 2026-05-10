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
from discordbot.cogs._games.dice import play_dice, render_rolls
from discordbot.cogs._games.views import BlackjackView, build_final_embed, build_in_progress_embed
from discordbot.cogs._games.dealer import DealerAI
from discordbot.cogs._games.blackjack import BlackjackHand
from discordbot.cogs._economy.database import PlacedBet, place_bet, get_balance
from discordbot.cogs._games.settlement import (
    settle_wager,
    settle_blackjack_round,
    blackjack_early_finish_note,
)
from discordbot.cogs._games.presentation import (
    ERROR_COLOR,
    IN_PROGRESS_COLOR,
    bet_field_value,
    settlement_footer,
    dice_outcome_presentation,
    blackjack_outcome_presentation,
)

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

    def _dealer_identity(self) -> tuple[int, str]:
        """Returns ``(dealer_id, dealer_name)`` from the bot's own user record.

        Slash commands only fire after the gateway has connected, so
        ``self.bot.user`` is guaranteed non-None at call time. We still fall
        back to a synthetic id / "莊家" name to keep type-narrowing clean and
        to avoid blowing up the round if Discord briefly returns no client
        user (e.g. mid-reconnect).
        """
        if self.bot.user is None:
            return (0, "莊家")
        return (self.bot.user.id, self.bot.user.display_name)

    async def _place_bet(
        self, *, interaction: Interaction, requested_bet: int
    ) -> PlacedBet | None:
        """Withdraws the effective bet or sends an insufficient-balance embed.

        `place_bet()` owns the atomic balance check and auto all-in clamp, so
        slash commands do not make game decisions from stale balances.
        """
        if interaction.user is None:
            return None
        placed_bet = await place_bet(
            user_id=interaction.user.id, name=interaction.user.name, requested_bet=requested_bet
        )
        if placed_bet is None:
            balance = await get_balance(user_id=interaction.user.id)
            await interaction.followup.send(
                embed=Embed(
                    title=":x: 餘額不足",
                    description=(
                        f"你目前只有 **{balance:,}** 點, 沒有可下注的點數。\n"
                        "跟機器人聊天可以累積點數。"
                    ),
                    color=ERROR_COLOR,
                )
            )
            return None
        return placed_bet

    @nextcord.slash_command(
        name="dice",
        description="Roll three dice against the dealer; whoever totals higher wins.",
        name_localizations={Locale.zh_TW: "比大小", Locale.ja: "サイコロ勝負"},
        description_localizations={
            Locale.zh_TW: "用三顆骰子跟莊家比點數總和, 大的贏。",
            Locale.ja: "3個のサイコロで親と勝負し、合計が大きい方が勝ち。",
        },
        nsfw=False,
    )
    async def dice(
        self,
        interaction: Interaction,
        bet: int = SlashOption(
            name="bet",
            description="How many points to wager (auto all-ins if over your balance).",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: "下注的點數 (超過餘額會自動 all-in)。",
                Locale.ja: "賭けるポイント数 (残高を超えると自動 all-in)。",
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

        placed_bet = await self._place_bet(interaction=interaction, requested_bet=bet)
        if placed_bet is None:
            return
        bet = placed_bet.amount
        is_allin = placed_bet.is_allin

        dealer_id, dealer_name = self._dealer_identity()

        taunt = await self.dealer.taunt_bet(
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            balance_after_bet=placed_bet.balance_after,
            bet=bet,
            game="dice",
        )

        in_progress = Embed(
            title=":game_die: 比大小 - 下注", description=taunt, color=IN_PROGRESS_COLOR
        )
        in_progress.add_field(
            name="下注", value=bet_field_value(bet=bet, is_allin=is_allin), inline=True
        )
        in_progress.add_field(
            name="下注後餘額", value=f"{placed_bet.balance_after:,} 點", inline=True
        )
        in_progress.set_footer(text="正在搖骰子...")
        message = await interaction.followup.send(embed=in_progress, wait=True)

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
            dealer_id=dealer_id,
            dealer_name=dealer_name,
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
            title=f":game_die: 比大小 - {outcome_label}", description=banter, color=color
        )
        final.add_field(
            name=f"{interaction.user.display_name}",
            value=render_rolls(rolls=result.player_rolls),
            inline=False,
        )
        final.add_field(
            name=dealer_name, value=render_rolls(rolls=result.dealer_rolls), inline=False
        )
        final.set_footer(
            text=settlement_footer(
                bet=bet,
                delta=settlement.delta,
                new_balance=settlement.new_balance,
                house_balance=settlement.house_balance,
                is_allin=is_allin,
            )
        )
        await message.edit(embed=final)

    @nextcord.slash_command(
        name="blackjack",
        description="Play one round of 21 against the dealer.",
        name_localizations={Locale.zh_TW: "二十一點", Locale.ja: "ブラックジャック"},
        description_localizations={
            Locale.zh_TW: "跟莊家玩一局 21 點。",
            Locale.ja: "親と21（ブラックジャック）を1ラウンド遊びます。",
        },
        nsfw=False,
    )
    async def blackjack(
        self,
        interaction: Interaction,
        bet: int = SlashOption(
            name="bet",
            description="How many points to wager (auto all-ins if over your balance).",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: "下注的點數 (超過餘額會自動 all-in)。",
                Locale.ja: "賭けるポイント数 (残高を超えると自動 all-in)。",
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

        placed_bet = await self._place_bet(interaction=interaction, requested_bet=bet)
        if placed_bet is None:
            return
        bet = placed_bet.amount
        is_allin = placed_bet.is_allin

        dealer_id, dealer_name = self._dealer_identity()

        hand = BlackjackHand(rng=self.rng, bet=bet)
        hand.deal_initial()

        taunt = await self.dealer.taunt_bet(
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            balance_after_bet=placed_bet.balance_after,
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
                dealer_id=dealer_id,
                dealer_name=dealer_name,
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
                hand=hand,
                bet=bet,
                delta=settlement.delta,
                new_balance=settlement.new_balance,
                house_balance=settlement.house_balance,
                dealer_line=banter,
                outcome_label=outcome_label,
                color=color,
                is_allin=is_allin,
                round_note=blackjack_early_finish_note(hand=hand),
            )
            await interaction.followup.send(embed=embed)
            return

        view = BlackjackView(
            dealer=self.dealer,
            hand=hand,
            owner_id=interaction.user.id,
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            dealer_id=dealer_id,
            dealer_name=dealer_name,
            balance_after_bet=placed_bet.balance_after,
            is_allin=is_allin,
        )
        embed = build_in_progress_embed(
            dealer_name=dealer_name,
            player_name=interaction.user.display_name,
            hand=hand,
            balance_after_bet=placed_bet.balance_after,
            dealer_line=taunt,
            is_allin=is_allin,
        )
        message = await interaction.followup.send(embed=embed, view=view, wait=True)
        view.message = message


def setup(bot: commands.Bot) -> None:
    """Adds the GamesCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(GamesCogs(bot), override=True)
