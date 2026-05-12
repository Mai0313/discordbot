"""Interactive components for the casino games."""

import asyncio
import contextlib

from nextcord import Embed, Message, ButtonStyle, Interaction, ui

from discordbot.cogs._games.dealer import DealerAI
from discordbot.cogs._games.cleanup import schedule_game_message_delete
from discordbot.cogs._games.blackjack import BlackjackHand, render_hand, dealer_visible_value
from discordbot.cogs._games.settlement import settle_blackjack_round
from discordbot.cogs._games.presentation import (
    IN_PROGRESS_COLOR,
    duel_lines,
    dealer_quote,
    wager_footer,
    settlement_footer,
    blackjack_outcome_presentation,
)


def _format_player_line(hand: BlackjackHand) -> str:
    """Formats the player's hand with its current total."""
    return f"{render_hand(cards=hand.player)}  **{hand.player_total()}**"


def _format_dealer_line(hand: BlackjackHand, hide_hole: bool) -> str:
    """Formats the dealer's hand, optionally hiding the hole card."""
    if hide_hole:
        return f"{render_hand(cards=hand.dealer, hide_first=True)}  **?**"
    return f"{render_hand(cards=hand.dealer)}  **{hand.dealer_total()}**"


def build_in_progress_embed(  # noqa: PLR0913 -- in-progress embed needs every render input
    *,
    dealer_name: str,
    player_name: str,
    player_avatar_url: str,
    hand: BlackjackHand,
    balance_after_bet: int,
    dealer_line: str,
    is_allin: bool = False,
) -> Embed:
    """Builds the embed shown while the player is still acting.

    Args:
        dealer_name: Display name used for the dealer field title.
        player_name: Display name used for the player field title.
        player_avatar_url: Last-seen Discord avatar URL for the player (used as author icon).
        hand: Current Blackjack hand state.
        balance_after_bet: Player balance after the wager was withdrawn.
        dealer_line: Dealer banter shown in the embed description.
        is_allin: Whether the requested bet was clamped to the full balance.

    Returns:
        The in-progress Blackjack embed.
    """
    embed = Embed(
        title="♠️ 21 點", description=dealer_quote(text=dealer_line), color=IN_PROGRESS_COLOR
    )
    embed.set_author(name=f"{player_name} 的對局", icon_url=player_avatar_url or None)
    embed.add_field(
        name="牌面",
        value=duel_lines(
            player_name=player_name,
            player_value=_format_player_line(hand=hand),
            dealer_name=dealer_name,
            dealer_value=_format_dealer_line(hand=hand, hide_hole=True),
        ),
        inline=False,
    )
    embed.set_footer(
        text=wager_footer(
            bet=hand.bet, balance_after_bet=balance_after_bet, is_allin=is_allin, status="等待操作"
        )
    )
    return embed


def build_final_embed(  # noqa: PLR0913 -- final embed is one cohesive payload
    *,
    dealer_name: str,
    player_name: str,
    player_avatar_url: str,
    hand: BlackjackHand,
    delta: int,
    new_balance: int,
    dealer_line: str,
    outcome_label: str,
    color: int,
    is_allin: bool = False,
    round_note: str | None = None,
) -> Embed:
    """Builds the embed for a finished round.

    Args:
        dealer_name: Display name used for the dealer field title.
        player_name: Display name used for the player field title.
        player_avatar_url: Last-seen Discord avatar URL for the player (used as author icon).
        hand: Final Blackjack hand state.
        delta: Player net point change for the round.
        new_balance: Player balance after settlement.
        dealer_line: Dealer banter shown in the embed description.
        outcome_label: Human-readable result label for the embed title.
        color: Embed color for the outcome.
        is_allin: Whether the requested bet was clamped to the full balance.
        round_note: Optional explanation for a round that ended before player action.

    Returns:
        The final Blackjack embed.
    """
    embed = Embed(
        title=f"♠️ 21 點 | {outcome_label}", description=dealer_quote(text=dealer_line), color=color
    )
    embed.set_author(name=f"{player_name} 的對局", icon_url=player_avatar_url or None)
    embed.add_field(
        name="牌面",
        value=duel_lines(
            player_name=player_name,
            player_value=_format_player_line(hand=hand),
            dealer_name=dealer_name,
            dealer_value=_format_dealer_line(hand=hand, hide_hole=False),
        ),
        inline=False,
    )
    if round_note:
        embed.add_field(name="提前結束", value=f"**{round_note}**", inline=False)
    embed.set_footer(
        text=settlement_footer(delta=delta, new_balance=new_balance, is_allin=is_allin)
    )
    return embed


class BlackjackView(ui.View):
    """Hit / Stand buttons for one Blackjack round.

    Attributes:
        dealer: AI dealer used for inline banter.
        hand: Mutable round state.
        owner_id: Discord user ID allowed to press the buttons.
        author_name: Discord username used as the litellm end-user-id (ASCII-safe).
        player_name: Display name used in the embed.
        player_avatar_url: Last-seen Discord avatar URL for the player account row.
        dealer_id: Discord user ID of the bot itself (used for the house ledger).
        dealer_name: Display name of the bot, surfaced in embeds and the house ledger row.
        dealer_avatar_url: Last-seen Discord avatar URL for the dealer ledger row.
        balance_after_bet: Player's balance immediately after the bet was deducted.
        is_allin: True when the original bet was clamped down to the player's balance.
        message: Reference to the rendered Discord message; set on first edit.
    """

    def __init__(  # noqa: PLR0913 -- view needs every field for dealer + embed render
        self,
        *,
        dealer: DealerAI,
        hand: BlackjackHand,
        owner_id: int,
        author_name: str,
        player_name: str,
        player_avatar_url: str = "",
        dealer_id: int,
        dealer_name: str,
        dealer_avatar_url: str = "",
        balance_after_bet: int,
        is_allin: bool = False,
    ) -> None:
        """Initialises the view.

        Args:
            dealer: AI dealer used for inline banter.
            hand: Round state.
            owner_id: Discord user ID allowed to interact.
            author_name: Discord username used as the litellm end-user-id.
            player_name: Display name used in the embed.
            player_avatar_url: Last-seen Discord avatar URL for the player account row.
            dealer_id: Discord user ID of the bot itself (house ledger key).
            dealer_name: Bot's display name; shown in embeds and stored on the house ledger row.
            dealer_avatar_url: Last-seen Discord avatar URL for the dealer ledger row.
            balance_after_bet: Player balance after the bet was withdrawn.
            is_allin: True when the original bet was clamped down to ``balance``.
        """
        super().__init__(timeout=180)
        self.dealer = dealer
        self.hand = hand
        self.owner_id = owner_id
        self.author_name = author_name
        self.player_name = player_name
        self.player_avatar_url = player_avatar_url
        self.dealer_id = dealer_id
        self.dealer_name = dealer_name
        self.dealer_avatar_url = dealer_avatar_url
        self.balance_after_bet = balance_after_bet
        self.is_allin = is_allin
        self.message: Message | None = None
        self._round_lock = asyncio.Lock()
        self._settled = False

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Restricts the buttons to the original player.

        Args:
            interaction: Button interaction to authorize.

        Returns:
            True when the interaction user owns the round, otherwise False.
        """
        if interaction.user is None or interaction.user.id != self.owner_id:
            await interaction.response.send_message(content="這局不是你的, 別插手", ephemeral=True)
            return False
        return True

    async def on_timeout(self) -> None:
        """If the player walked away, auto-stand the hand and finalise it."""
        if self.message is None:
            return
        async with self._round_lock:
            if self._settled or self.hand.finished:
                return
            self.hand.stand()
            await self._finalize_locked(message=self.message)

    @ui.button(label="再要一張", emoji="🃏", style=ButtonStyle.primary)
    async def hit(self, _button: ui.Button, interaction: Interaction) -> None:
        """Handles the Hit button.

        Draws a player card, finalizes on bust, or asks the dealer for a hint
        before updating the in-progress embed.

        Args:
            _button: Button component that triggered the callback.
            interaction: Button interaction from Discord.
        """
        await interaction.response.defer()
        if interaction.message is None:
            return
        async with self._round_lock:
            if self._settled or self.hand.finished:
                return
            self.hand.hit()
            if self.hand.finished:
                await self._finalize_locked(message=interaction.message)
                return
            hint = await self.dealer.hint(
                author_name=self.author_name,
                player_name=self.player_name,
                player_total=self.hand.player_total(),
                dealer_visible=dealer_visible_value(hand=self.hand),
            )
            embed = build_in_progress_embed(
                dealer_name=self.dealer_name,
                player_name=self.player_name,
                player_avatar_url=self.player_avatar_url,
                hand=self.hand,
                balance_after_bet=self.balance_after_bet,
                dealer_line=hint,
                is_allin=self.is_allin,
            )
            await interaction.message.edit(embed=embed, view=self)

    @ui.button(label="停手", emoji="✋", style=ButtonStyle.secondary)
    async def stand(self, _button: ui.Button, interaction: Interaction) -> None:
        """Handles the Stand button.

        Resolves the dealer hand and finalizes the round.

        Args:
            _button: Button component that triggered the callback.
            interaction: Button interaction from Discord.
        """
        await interaction.response.defer()
        if interaction.message is None:
            return
        async with self._round_lock:
            if self._settled or self.hand.finished:
                return
            self.hand.stand()
            await self._finalize_locked(message=interaction.message)

    async def _finalize(self, *, message: Message) -> None:
        """Settles DB, asks the dealer for a closing line, and updates the embed.

        The bet was already withdrawn before the view was created, so we credit
        back ``bet + delta`` here: ``2 * bet`` on a regular win, ``2.5 * bet``
        on a natural Blackjack, ``bet`` on a push, and ``0`` on a loss.

        We also mirror the player's net change into the bot's house ledger
        (negated, since dealer P&L is the inverse of player P&L), so global
        casino performance is always visible via ``/house``.
        """
        async with self._round_lock:
            await self._finalize_locked(message=message)

    async def _finalize_locked(self, *, message: Message) -> None:
        """Finalizes a round while the caller holds the round lock."""
        if self._settled:
            return
        self._settled = True

        settlement = await settle_blackjack_round(
            hand=self.hand,
            player_id=self.owner_id,
            player_account_name=self.author_name,
            player_avatar_url=self.player_avatar_url,
            dealer_id=self.dealer_id,
            dealer_name=self.dealer_name,
            dealer_avatar_url=self.dealer_avatar_url,
        )
        outcome_label, color = blackjack_outcome_presentation(outcome=settlement.outcome)
        banter = await self.dealer.settle(
            author_name=self.author_name,
            player_name=self.player_name,
            outcome=settlement.outcome,
            bet=self.hand.bet,
            delta=settlement.delta,
            new_balance=settlement.new_balance,
            game="blackjack",
            detail=settlement.detail,
        )
        embed = build_final_embed(
            dealer_name=self.dealer_name,
            player_name=self.player_name,
            player_avatar_url=self.player_avatar_url,
            hand=self.hand,
            delta=settlement.delta,
            new_balance=settlement.new_balance,
            dealer_line=banter,
            outcome_label=outcome_label,
            color=color,
            is_allin=self.is_allin,
        )
        for child in self.children:
            if isinstance(child, ui.Button):
                child.disabled = True
        self.stop()
        with contextlib.suppress(Exception):
            await message.edit(embed=embed, view=self)
        schedule_game_message_delete(message=message)


__all__: list[str] = [
    "BlackjackView",
    "blackjack_outcome_presentation",
    "build_final_embed",
    "build_in_progress_embed",
]
