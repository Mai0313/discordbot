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
from discordbot.cogs._games.blackjack import BlackjackHand, settle, is_blackjack
from discordbot.cogs._economy.database import get_balance, settle_game, house_settle

_DICE_IN_PROGRESS_COLOR = 0x5865F2
_WIN_COLOR = 0x57F287
_LOSE_COLOR = 0xED4245
_PUSH_COLOR = 0xFEE75C
_ERROR_COLOR = 0xED4245

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
        """The cached AsyncOpenAI client used for dealer banter."""
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return client

    @property
    def dealer_model(self) -> ModelSettings:
        """Fast model used by the AI dealer; matches `gen_reply.fast_model`."""
        return ModelSettings(name="gemini-flash-latest", effort="none")

    @cached_property
    def dealer(self) -> DealerAI:
        """The cached DealerAI instance reused across commands."""
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

    async def _reject_invalid_bet(
        self, *, interaction: Interaction, balance: int, bet: int
    ) -> bool:
        """Sends an embed and returns True when the bet should be rejected."""
        if bet > balance:
            await interaction.followup.send(
                embed=Embed(
                    title=":x: 餘額不足",
                    description=(
                        f"你目前只有 **{balance:,}** 點, 想下注 **{bet:,}** 點。\n"
                        "跟機器人聊天可以累積點數。"
                    ),
                    color=_ERROR_COLOR,
                )
            )
            return True
        return False

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
            description="How many points to wager (1 ~ your current balance).",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: "下注的點數 (1 ~ 目前餘額)。",
                Locale.ja: "賭けるポイント数 (1 〜 現在の残高)。",
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

        balance = await get_balance(user_id=interaction.user.id)
        if await self._reject_invalid_bet(interaction=interaction, balance=balance, bet=bet):
            return

        dealer_id, dealer_name = self._dealer_identity()

        # Withdraw the bet up-front so a concurrent /dice or /blackjack from
        # the same player can't double-spend the same balance.
        balance_after_bet = await settle_game(
            user_id=interaction.user.id, name=interaction.user.name, delta=-bet
        )

        taunt = await self.dealer.taunt_bet(
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            balance_after_bet=balance_after_bet,
            bet=bet,
            game="dice",
        )

        in_progress = Embed(
            title=":game_die: 比大小 - 下注", description=taunt, color=_DICE_IN_PROGRESS_COLOR
        )
        in_progress.add_field(name="下注", value=f"{bet:,} 點", inline=True)
        in_progress.add_field(name="下注後餘額", value=f"{balance_after_bet:,} 點", inline=True)
        in_progress.set_footer(text="正在搖骰子...")
        message = await interaction.followup.send(embed=in_progress, wait=True)

        await asyncio.sleep(delay=_DICE_REVEAL_DELAY_SECONDS)
        result = play_dice(rng=self.rng)

        if result.outcome == "win":
            payout = bet * 2
            outcome_label = "你贏了"
            color = _WIN_COLOR
        elif result.outcome == "push":
            payout = bet
            outcome_label = "平手"
            color = _PUSH_COLOR
        else:
            payout = 0
            outcome_label = "你輸了"
            color = _LOSE_COLOR

        new_balance = await settle_game(
            user_id=interaction.user.id, name=interaction.user.name, delta=payout
        )
        delta = payout - bet
        await house_settle(user_id=dealer_id, name=dealer_name, delta=-delta)
        detail = f"玩家骰 {result.player_total} 點, 莊家骰 {result.dealer_total} 點"
        banter = await self.dealer.settle(
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            outcome=result.outcome,
            bet=bet,
            delta=delta,
            new_balance=new_balance,
            game="dice",
            detail=detail,
        )

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
        delta_text = f"{delta:+,}" if delta != 0 else "0"
        final.set_footer(text=f"下注 {bet:,} · 本局淨變動 {delta_text} · 餘額 {new_balance:,}")
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
            description="How many points to wager (1 ~ your current balance).",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: "下注的點數 (1 ~ 目前餘額)。",
                Locale.ja: "賭けるポイント数 (1 〜 現在の残高)。",
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

        balance = await get_balance(user_id=interaction.user.id)
        if await self._reject_invalid_bet(interaction=interaction, balance=balance, bet=bet):
            return

        dealer_id, dealer_name = self._dealer_identity()

        balance_after_bet = await settle_game(
            user_id=interaction.user.id, name=interaction.user.name, delta=-bet
        )

        hand = BlackjackHand(rng=self.rng, bet=bet)
        hand.deal_initial()

        taunt = await self.dealer.taunt_bet(
            author_name=interaction.user.name,
            player_name=interaction.user.display_name,
            balance_after_bet=balance_after_bet,
            bet=bet,
            game="blackjack",
        )

        # Natural Blackjack (or dealer Blackjack) ends the hand before the
        # player gets to act; settle and post the final embed straight away.
        if hand.finished:
            outcome, delta = settle(hand=hand)
            payout = max(bet + delta, 0)
            new_balance = await settle_game(
                user_id=interaction.user.id, name=interaction.user.name, delta=payout
            )
            await house_settle(user_id=dealer_id, name=dealer_name, delta=-delta)
            if is_blackjack(cards=hand.player) and is_blackjack(cards=hand.dealer):
                detail = "雙方都是 Blackjack, 平手"
            elif is_blackjack(cards=hand.player):
                detail = f"玩家 21 點 Blackjack, 莊家 {hand.dealer_total()} 點"
            else:
                detail = f"莊家 21 點 Blackjack, 玩家 {hand.player_total()} 點"
            banter = await self.dealer.settle(
                author_name=interaction.user.name,
                player_name=interaction.user.display_name,
                outcome=outcome,
                bet=bet,
                delta=delta,
                new_balance=new_balance,
                game="blackjack",
                detail=detail,
            )
            outcome_label, color = {
                "win": ("你贏了", _WIN_COLOR),
                "lose": ("你輸了", _LOSE_COLOR),
                "push": ("平手", _PUSH_COLOR),
                "blackjack": ("Blackjack!", _WIN_COLOR),
                "player_bust": ("你爆牌了", _LOSE_COLOR),
                "dealer_bust": ("莊家爆牌, 你贏了", _WIN_COLOR),
            }[outcome]
            embed = build_final_embed(
                dealer_name=dealer_name,
                player_name=interaction.user.display_name,
                hand=hand,
                bet=bet,
                delta=delta,
                new_balance=new_balance,
                dealer_line=banter,
                outcome_label=outcome_label,
                color=color,
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
            balance_after_bet=balance_after_bet,
        )
        embed = build_in_progress_embed(
            dealer_name=dealer_name,
            player_name=interaction.user.display_name,
            hand=hand,
            balance_after_bet=balance_after_bet,
            dealer_line=taunt,
        )
        message = await interaction.followup.send(embed=embed, view=view, wait=True)
        view.message = message


def setup(bot: commands.Bot) -> None:
    """Adds the GamesCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(GamesCogs(bot), override=True)
