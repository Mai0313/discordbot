"""Shared base lobby views for multiplayer casino game sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol
import asyncio
import contextlib

import logfire
from nextcord import Embed, Message, ButtonStyle, Interaction, ui

from discordbot.cogs._games.cleanup import schedule_game_message_delete
from discordbot.cogs._economy.database import apply_jackpot_settlement

if TYPE_CHECKING:
    from random import Random

    from discordbot.typings.games import GameParticipant
    from discordbot.cogs._games.dealer import DealerAI


class PrepareParticipant(Protocol):
    """Callable used by lobby join buttons to validate a participant.

    Game-specific wager / mode / insufficient-balance copy are bound by the
    caller (typically via ``functools.partial``) so the callable signature
    stays uniform across lobbies.
    """

    async def __call__(self, interaction: Interaction) -> GameParticipant | None:
        """Returns a prepared participant or sends the interaction error."""


class RefreshParticipants(Protocol):
    """Callable used by lobby start to re-check balances.

    Wager / mode are bound by the caller via ``functools.partial``.
    """

    async def __call__(
        self, participants: list[GameParticipant]
    ) -> tuple[list[GameParticipant], list[str]]:
        """Returns refreshed participants and display names removed from the table."""


class BaseGameLobbyView(ui.View):
    """Join / leave / start scaffold shared by multiplayer game lobbies.

    Subclasses must override:
      - ``_build_lobby_embed(status: str) -> Embed`` — used by refresh + timeout
      - ``_start_game(message: Message | None) -> None`` — invoked after Start

    Optional class attribute:
      - ``max_players: ClassVar[int | None]`` — None means unlimited
    """

    max_players: ClassVar[int | None] = None

    def __init__(  # noqa: PLR0913 -- lobby owns all table dependencies
        self,
        owner: GameParticipant,
        rng: Random,
        dealer: DealerAI,
        dealer_name: str,
        dealer_avatar_url: str,
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
        timeout: int,
    ) -> None:
        super().__init__(timeout=timeout)
        self.owner = owner
        self.rng = rng
        self.dealer = dealer
        self.dealer_name = dealer_name
        self.dealer_avatar_url = dealer_avatar_url
        self.prepare_participant = prepare_participant
        self.refresh_participants = refresh_participants
        self.message: Message | None = None
        self._participants: dict[int, GameParticipant] = {owner.user_id: owner}
        self._lock = asyncio.Lock()
        self._started = False

    @property
    def participants(self) -> list[GameParticipant]:
        """Returns participants in join order."""
        return list(self._participants.values())

    async def on_timeout(self) -> None:
        """Cleans up a lobby that never started."""
        if self._started or self.message is None:
            return
        self._disable_buttons()
        self.stop()
        embed = self._build_lobby_embed(status="Lobby 已逾時")
        with contextlib.suppress(Exception):
            await self.message.edit(embed=embed, view=self)
        schedule_game_message_delete(message=self.message)

    @ui.button(label="加入", emoji="✅", style=ButtonStyle.success)
    async def join(self, _button: ui.Button, interaction: Interaction) -> None:
        """Adds the interacting user to the lobby."""
        if interaction.user is None:
            return
        async with self._lock:
            if self._started:
                await self._send_notice(interaction=interaction, content="這桌已經開始了")
                return
            if interaction.user.id in self._participants:
                await self._send_notice(interaction=interaction, content="你已經在這桌了")
                return
            if self.max_players is not None and len(self._participants) >= self.max_players:
                await self._send_notice(interaction=interaction, content="這桌已經滿了")
                return
            await interaction.response.defer()
            participant = await self.prepare_participant(interaction=interaction)
            if participant is None:
                return
            self._participants[participant.user_id] = participant
            await self._refresh_message(
                message=interaction.message, status=f"{participant.display_name} 已加入"
            )

    @ui.button(label="離開", emoji="🚪", style=ButtonStyle.secondary)
    async def leave(self, _button: ui.Button, interaction: Interaction) -> None:
        """Removes the interacting user from the lobby."""
        if interaction.user is None:
            return
        async with self._lock:
            if self._started:
                await self._send_notice(interaction=interaction, content="這桌已經開始了")
                return
            if interaction.user.id == self.owner.user_id:
                await self._send_notice(interaction=interaction, content="房主不能離開 lobby")
                return
            participant = self._participants.pop(interaction.user.id, None)
            if participant is None:
                await self._send_notice(interaction=interaction, content="你不在這桌")
                return
            await interaction.response.defer()
            await self._refresh_message(
                message=interaction.message, status=f"{participant.display_name} 已離開"
            )

    @ui.button(label="開始", emoji="▶️", style=ButtonStyle.primary)
    async def start(self, _button: ui.Button, interaction: Interaction) -> None:
        """Starts the game if the lobby owner pressed the button."""
        if interaction.user is None:
            return
        if interaction.user.id != self.owner.user_id:
            await self._send_notice(interaction=interaction, content="只有房主可以開始")
            return
        await interaction.response.defer()
        async with self._lock:
            if self._started:
                await self._send_notice(interaction=interaction, content="這桌已經開始了")
                return
            refreshed, dropped = await self.refresh_participants(participants=self.participants)
            self._participants = {participant.user_id: participant for participant in refreshed}
            if self.owner.user_id not in self._participants:
                await self._send_notice(interaction=interaction, content="你的餘額不足, 不能開始")
                await self._refresh_message(message=interaction.message, status="房主餘額不足")
                return
            self._started = True
        if dropped:
            names = ", ".join(dropped)
            await self._send_notice(interaction=interaction, content=f"餘額不足已移出: {names}")
        self.stop()
        await self._start_game(message=interaction.message)

    async def _refresh_message(self, message: Message | None, status: str) -> None:
        if message is None:
            return
        self.message = message
        await message.edit(embed=self._build_lobby_embed(status=status), view=self)

    async def _send_notice(self, interaction: Interaction, content: str) -> None:
        try:
            if interaction.response.is_done():
                await interaction.followup.send(content=content, ephemeral=True)
                return
            await interaction.response.send_message(content=content, ephemeral=True)
        except Exception:
            logfire.warn("Failed to send lobby notice", _exc_info=True)

    async def on_error(self, error: Exception, item: ui.Item, interaction: Interaction) -> None:
        """Logs lobby component failures instead of only printing to stderr."""
        logfire.error(
            "Lobby interaction failed",
            item_label=getattr(item, "label", None),
            user_id=getattr(interaction.user, "id", None),
            _exc_info=(type(error), error, error.__traceback__),
        )

    def _disable_buttons(self) -> None:
        for child in self.children:
            if isinstance(child, ui.Button):
                child.disabled = True

    def _build_lobby_embed(self, status: str = "等待玩家加入") -> Embed:
        raise NotImplementedError

    async def _start_game(self, message: Message | None) -> None:
        raise NotImplementedError


class BaseJackpotLobbyView(BaseGameLobbyView):
    """Base lobby for games sharing a global jackpot pool.

    On Start, each participant is charged ``ante`` into the jackpot via
    ``apply_jackpot_settlement`` before the table begins. Subclasses must
    declare ``game_id`` / ``ante`` and override ``_start_game_after_antes``.
    """

    game_id: ClassVar[str]
    ante: ClassVar[int]

    def __init__(  # noqa: PLR0913 -- jackpot lobby adds initial_jackpot on top of base deps
        self,
        owner: GameParticipant,
        rng: Random,
        dealer: DealerAI,
        dealer_name: str,
        dealer_avatar_url: str,
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
        initial_jackpot: int,
        timeout: int,
    ) -> None:
        super().__init__(
            owner=owner,
            rng=rng,
            dealer=dealer,
            dealer_name=dealer_name,
            dealer_avatar_url=dealer_avatar_url,
            prepare_participant=prepare_participant,
            refresh_participants=refresh_participants,
            timeout=timeout,
        )
        self._jackpot_snapshot = initial_jackpot

    async def _start_game(self, message: Message | None) -> None:
        if message is None:
            return
        final_balances = await self._settle_pregame_antes()
        await self._start_game_after_antes(message=message, final_balances=final_balances)

    async def _settle_pregame_antes(self) -> dict[int, int]:
        """Charges each participant ``ante`` into the jackpot pool.

        Iterates participants in join order; on each iteration applies one
        ``apply_jackpot_settlement(player_delta=-ante)`` call atomically. Each
        return updates ``_jackpot_snapshot`` so a future opener sees a current
        view of the pool. Returns ``{user_id: post_ante_balance}``.
        """
        final_balances: dict[int, int] = {}
        jackpot_after = self._jackpot_snapshot
        for participant in self.participants:
            player_balance, jackpot_after = await apply_jackpot_settlement(
                player_id=participant.user_id,
                player_account_name=participant.account_name,
                player_avatar_url=participant.avatar_url,
                player_delta=-self.ante,
                game_id=self.game_id,
            )
            final_balances[participant.user_id] = player_balance
        self._jackpot_snapshot = jackpot_after
        return final_balances

    async def _start_game_after_antes(
        self, message: Message, final_balances: dict[int, int]
    ) -> None:
        raise NotImplementedError
