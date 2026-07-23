"""Shared base lobby views for multiplayer casino game sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol
import asyncio
import contextlib

import logfire
import nextcord
from nextcord import Embed, Message, ButtonStyle, Interaction
from nextcord.ui import Item, View, Button

from discordbot.typings.economy import JackpotSettlementRequest, JackpotSettlementBatchResult
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import schedule_public_message_delete
from discordbot.cogs._economy.database import apply_jackpot_settlement_batch
from discordbot.cogs._games.interactions import disable_view_components
from discordbot.utils.owned_message_views import send_ephemeral_notice

if TYPE_CHECKING:
    from random import Random
    from collections.abc import Iterable

    from nextcord.ext import commands

    from discordbot.typings.games import GameParticipant, RefreshParticipantsResult


class PrepareParticipant(Protocol):
    """Callable used by lobby join buttons to validate a participant.

    Game-specific wager / mode / insufficient-balance copy are bound by the
    caller (typically via `functools.partial`) so the callable signature
    stays uniform across lobbies.
    """

    async def __call__(self, interaction: Interaction[commands.Bot]) -> GameParticipant | None:
        """Returns a prepared participant or sends the interaction error."""


class RefreshParticipants(Protocol):
    """Callable used by lobby start to re-check balances.

    Wager / mode are bound by the caller via `functools.partial`.
    """

    async def __call__(self, participants: list[GameParticipant]) -> RefreshParticipantsResult:
        """Returns refreshed participants and display names removed from the table."""


class BaseGameLobbyView(View):
    """Join / leave / start scaffold shared by multiplayer game lobbies.

    Subclasses must override:
      - `_build_lobby_embed(status: str) -> Embed` — used by refresh + timeout
      - `_start_game(message: Message | None) -> bool` — invoked after Start

    Optional class attribute:
      - `max_players: ClassVar[int | None]` — None means unlimited
    """

    max_players: ClassVar[int | None] = None

    def __init__(  # noqa: PLR0913 -- lobby owns all table dependencies
        self,
        owner: GameParticipant,
        rng: Random,
        system_name: str,
        system_avatar_url: str,
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
        timeout: int,
        extra_initial_participants: Iterable[GameParticipant] | None = None,
    ) -> None:
        """Initializes shared lobby state and registers the owner."""
        super().__init__(timeout=timeout)
        self.owner = owner
        self.rng = rng
        self.system_name = system_name
        self.system_avatar_url = system_avatar_url
        self.prepare_participant = prepare_participant
        self.refresh_participants = refresh_participants
        self.message: Message | None = None
        self._participants: dict[int, GameParticipant] = {owner.user_id: owner}
        for extra in extra_initial_participants or ():
            if extra.user_id != owner.user_id:
                self._participants[extra.user_id] = extra
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
            await self.message.edit(
                embed=embed,
                view=self,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=self.message),
            )
        schedule_public_message_delete(message=self.message, user_name=self.owner.account_name)

    @nextcord.ui.button(label="加入", emoji="✅", style=ButtonStyle.success)
    async def join(
        self, _button: Button[BaseGameLobbyView], interaction: Interaction[commands.Bot]
    ) -> None:
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

    @nextcord.ui.button(label="離開", emoji="🚪", style=ButtonStyle.secondary)
    async def leave(
        self, _button: Button[BaseGameLobbyView], interaction: Interaction[commands.Bot]
    ) -> None:
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

    @nextcord.ui.button(label="開始", emoji="▶️", style=ButtonStyle.primary)
    async def start(
        self, _button: Button[BaseGameLobbyView], interaction: Interaction[commands.Bot]
    ) -> None:
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
            refreshed = await self.refresh_participants(participants=self.participants)
            self._participants = {
                participant.user_id: participant for participant in refreshed.participants
            }
            if self.owner.user_id not in self._participants:
                await self._send_notice(interaction=interaction, content="你的餘額不足, 不能開始")
                await self._refresh_message(message=interaction.message, status="房主餘額不足")
                return
            self._started = True
        if refreshed.dropped_names:
            names = ", ".join(refreshed.dropped_names)
            await self._send_notice(interaction=interaction, content=f"餘額不足已移出: {names}")
        started = await self._start_game(message=interaction.message)
        if started:
            self.stop()

    async def _refresh_message(self, message: Message | None, status: str) -> None:
        """Edits the lobby message with the latest participant state."""
        if message is None:
            return
        self.message = message
        embed = self._build_lobby_embed(status=status)
        await message.edit(
            embed=embed,
            view=self,
            **embed_spacer_payload(embeds=[embed], is_edit=True, target=message),
        )

    async def _send_notice(self, interaction: Interaction[commands.Bot], content: str) -> None:
        """Sends a private lobby notice to the interacting user."""
        await send_ephemeral_notice(
            interaction=interaction, content=content, log_message="Failed to send lobby notice"
        )

    async def on_error(
        self,
        error: Exception,
        item: Item[BaseGameLobbyView],
        interaction: Interaction[commands.Bot],
    ) -> None:
        """Logs lobby component failures instead of only printing to stderr."""
        logfire.error(
            "Lobby interaction failed",
            item_label=getattr(item, "label", None),
            user_id=getattr(interaction.user, "id", None),
            _exc_info=(type(error), error, error.__traceback__),
        )

    def _disable_buttons(self) -> None:
        """Disables all button components on the lobby view."""
        disable_view_components(children=self.children, component_types=(Button,))

    def _build_lobby_embed(self, status: str = "等待玩家加入") -> Embed:
        """Builds the lobby embed for a concrete game type."""
        raise NotImplementedError

    async def _start_game(self, message: Message | None) -> bool:
        """Starts a concrete game from the current lobby participants."""
        raise NotImplementedError


class BaseJackpotLobbyView(BaseGameLobbyView):
    """Base lobby for games sharing a global jackpot pool.

    On Start, each participant is charged `ante` into the jackpot via
    one `apply_jackpot_settlement_batch` call before the table begins.
    Subclasses must declare `game_id` / `ante` and override
    `_start_game_after_antes`.
    """

    game_id: ClassVar[str]
    ante: ClassVar[int]

    def __init__(  # noqa: PLR0913 -- jackpot lobby adds initial_jackpot on top of base deps
        self,
        owner: GameParticipant,
        rng: Random,
        system_name: str,
        system_avatar_url: str,
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
        initial_jackpot: int,
        timeout: int,
        initial_jackpot_generation: int | None = None,
        extra_initial_participants: Iterable[GameParticipant] | None = None,
    ) -> None:
        """Initializes jackpot lobby state with the live pool snapshot."""
        super().__init__(
            owner=owner,
            rng=rng,
            system_name=system_name,
            system_avatar_url=system_avatar_url,
            prepare_participant=prepare_participant,
            refresh_participants=refresh_participants,
            timeout=timeout,
            extra_initial_participants=extra_initial_participants,
        )
        self._jackpot_snapshot = initial_jackpot
        self._jackpot_generation = initial_jackpot_generation

    async def _start_game(self, message: Message | None) -> bool:
        """Charges antes before delegating to the jackpot game start hook."""
        if message is None:
            self._started = False
            return False
        result = await self._settle_pregame_antes()
        if result.rejected_player_ids:
            rejected = set(result.rejected_player_ids)
            owner_rejected = self.owner.user_id in rejected
            dropped: list[str] = []
            for user_id in rejected:
                if user_id == self.owner.user_id:
                    continue
                participant = self._participants.pop(user_id, None)
                if participant is not None:
                    dropped.append(participant.display_name)
            self._started = False
            if owner_rejected:
                status = "房主餘額不足"
            elif dropped:
                status = f"餘額不足已移出: {', '.join(dropped)}"
            else:
                status = "餘額不足, 請重新開始"
            await self._refresh_message(message=message, status=status)
            return False
        await self._start_game_after_antes(message=message, final_balances=result.player_balances)
        return True

    async def _settle_pregame_antes(self) -> JackpotSettlementBatchResult:
        """Charges each participant `ante` into the jackpot pool.

        Applies all participant antes in one DB transaction so the lobby cannot
        partially charge a table.
        """
        settlements: list[JackpotSettlementRequest] = []
        for participant in self.participants:
            settlements.append(
                JackpotSettlementRequest(
                    player_id=participant.user_id,
                    player_account_name=participant.account_name,
                    player_avatar_url=participant.avatar_url,
                    player_delta=-self.ante,
                    require_full_debit=True,
                )
            )
        result = await apply_jackpot_settlement_batch(
            game_id=self.game_id, settlements=settlements
        )
        self._jackpot_snapshot = result.jackpot_balance
        self._jackpot_generation = result.jackpot_generation
        return result

    async def _start_game_after_antes(
        self, message: Message, final_balances: dict[int, int]
    ) -> None:
        """Starts a jackpot-backed game after ante settlement succeeds."""
        raise NotImplementedError
