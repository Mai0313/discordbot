"""Interactive components for the casino games."""

import contextlib

from nextcord import Embed, Message, ButtonStyle, Interaction, ui

from discordbot.cogs._games.dealer import DealerAI
from discordbot.cogs._games.blackjack import (
    BlackjackHand,
    settle,
    is_bust,
    render_hand,
    is_blackjack,
    dealer_visible_value,
)
from discordbot.cogs._economy.database import settle_game

_IN_PROGRESS_COLOR = 0x5865F2
_WIN_COLOR = 0x57F287
_LOSE_COLOR = 0xED4245
_PUSH_COLOR = 0xFEE75C


def _format_player_line(hand: BlackjackHand) -> str:
    return f"{render_hand(cards=hand.player)} (= {hand.player_total()})"


def _format_dealer_line(hand: BlackjackHand, hide_hole: bool) -> str:
    if hide_hole:
        return f"{render_hand(cards=hand.dealer, hide_first=True)} (= ?)"
    return f"{render_hand(cards=hand.dealer)} (= {hand.dealer_total()})"


def build_in_progress_embed(
    *, player_name: str, hand: BlackjackHand, balance_after_bet: int, dealer_line: str
) -> Embed:
    """Builds the embed shown while the player is still acting."""
    embed = Embed(
        title=":black_joker: 21 點 - 進行中",
        description=f"莊家：{dealer_line}",
        color=_IN_PROGRESS_COLOR,
    )
    embed.add_field(name=f"{player_name} 的牌", value=_format_player_line(hand=hand), inline=False)
    embed.add_field(
        name="莊家的牌", value=_format_dealer_line(hand=hand, hide_hole=True), inline=False
    )
    embed.set_footer(text=f"下注 {hand.bet:,} 點 · 下注後餘額 {balance_after_bet:,}")
    return embed


def build_final_embed(  # noqa: PLR0913 -- final embed is one cohesive payload
    *,
    player_name: str,
    hand: BlackjackHand,
    bet: int,
    delta: int,
    new_balance: int,
    dealer_line: str,
    outcome_label: str,
    color: int,
) -> Embed:
    """Builds the embed for a finished round."""
    embed = Embed(
        title=f":black_joker: 21 點 - {outcome_label}",
        description=f"莊家：{dealer_line}",
        color=color,
    )
    embed.add_field(name=f"{player_name} 的牌", value=_format_player_line(hand=hand), inline=False)
    embed.add_field(
        name="莊家的牌", value=_format_dealer_line(hand=hand, hide_hole=False), inline=False
    )
    delta_text = f"{delta:+,}" if delta != 0 else "0"
    embed.set_footer(text=f"下注 {bet:,} · 本局淨變動 {delta_text} · 餘額 {new_balance:,}")
    return embed


_OUTCOME_PRESENTATION: dict[str, tuple[str, int]] = {
    "win": ("你贏了", _WIN_COLOR),
    "lose": ("你輸了", _LOSE_COLOR),
    "push": ("平手", _PUSH_COLOR),
    "blackjack": ("Blackjack!", _WIN_COLOR),
    "player_bust": ("你爆牌了", _LOSE_COLOR),
    "dealer_bust": ("莊家爆牌, 你贏了", _WIN_COLOR),
}


class BlackjackView(ui.View):
    """Hit / Stand buttons for one Blackjack round.

    Attributes:
        dealer: AI dealer used for inline banter.
        hand: Mutable round state.
        owner_id: Discord user ID allowed to press the buttons.
        player_name: Display name used in the embed.
        balance_after_bet: Player's balance immediately after the bet was deducted.
        message: Reference to the rendered Discord message; set on first edit.
    """

    def __init__(
        self,
        *,
        dealer: DealerAI,
        hand: BlackjackHand,
        owner_id: int,
        player_name: str,
        balance_after_bet: int,
    ) -> None:
        """Initialises the view.

        Args:
            dealer: AI dealer used for inline banter.
            hand: Round state.
            owner_id: Discord user ID allowed to interact.
            player_name: Display name used in the embed.
            balance_after_bet: Player balance after the bet was withdrawn.
        """
        super().__init__(timeout=180)
        self.dealer = dealer
        self.hand = hand
        self.owner_id = owner_id
        self.player_name = player_name
        self.balance_after_bet = balance_after_bet
        self.message: Message | None = None

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Restricts the buttons to the original player."""
        if interaction.user is None or interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                content="這局不是你的, 別插手。", ephemeral=True
            )
            return False
        return True

    async def on_timeout(self) -> None:
        """If the player walked away, auto-stand the hand and finalise it."""
        if self.hand.finished or self.message is None:
            return
        self.hand.stand()
        await self._finalize(message=self.message, hint=None)

    @ui.button(label="再要一張", emoji="🃏", style=ButtonStyle.primary)
    async def hit(self, _button: ui.Button, interaction: Interaction) -> None:
        """Draws a card; if the hand is still live, asks the dealer for a hint."""
        await interaction.response.defer()
        if interaction.message is None:
            return
        self.hand.hit()
        if self.hand.finished:
            await self._finalize(message=interaction.message, hint=None)
            return
        hint = await self.dealer.hint(
            player_name=self.player_name,
            player_total=self.hand.player_total(),
            dealer_visible=dealer_visible_value(hand=self.hand),
        )
        embed = build_in_progress_embed(
            player_name=self.player_name,
            hand=self.hand,
            balance_after_bet=self.balance_after_bet,
            dealer_line=hint,
        )
        await interaction.message.edit(embed=embed, view=self)

    @ui.button(label="停手", emoji="✋", style=ButtonStyle.secondary)
    async def stand(self, _button: ui.Button, interaction: Interaction) -> None:
        """Player stops drawing; resolves the dealer's hand and finalises the round."""
        await interaction.response.defer()
        if interaction.message is None:
            return
        self.hand.stand()
        await self._finalize(message=interaction.message, hint=None)

    async def _finalize(self, *, message: Message, hint: str | None) -> None:
        """Settles DB, asks the dealer for a closing line, and updates the embed.

        The bet was already withdrawn before the view was created, so we credit
        back ``bet + delta`` here: ``2 * bet`` on a regular win, ``2.5 * bet``
        on a natural Blackjack, ``bet`` on a push, and ``0`` on a loss.
        """
        outcome, delta = settle(hand=self.hand)
        outcome_label, color = _OUTCOME_PRESENTATION[outcome]
        payout = max(self.hand.bet + delta, 0)
        new_balance = await settle_game(user_id=self.owner_id, name=self.player_name, delta=payout)
        if is_blackjack(cards=self.hand.player):
            detail = f"玩家 21 點 Blackjack, 莊家 {self.hand.dealer_total()} 點"
        elif is_bust(cards=self.hand.player):
            detail = f"玩家爆牌 {self.hand.player_total()} 點"
        elif is_bust(cards=self.hand.dealer):
            detail = f"莊家爆牌 {self.hand.dealer_total()} 點, 玩家 {self.hand.player_total()} 點"
        else:
            detail = f"玩家 {self.hand.player_total()} 點 vs 莊家 {self.hand.dealer_total()} 點"
        banter = await self.dealer.settle(
            player_name=self.player_name,
            outcome=outcome,
            bet=self.hand.bet,
            delta=delta,
            new_balance=new_balance,
            game="blackjack",
            detail=detail,
        )
        if hint:
            banter = f"{hint}\n\n{banter}"
        embed = build_final_embed(
            player_name=self.player_name,
            hand=self.hand,
            bet=self.hand.bet,
            delta=delta,
            new_balance=new_balance,
            dealer_line=banter,
            outcome_label=outcome_label,
            color=color,
        )
        for child in self.children:
            if isinstance(child, ui.Button):
                child.disabled = True
        self.stop()
        with contextlib.suppress(Exception):
            await message.edit(embed=embed, view=self)


__all__: list[str] = ["BlackjackView", "build_final_embed", "build_in_progress_embed"]
