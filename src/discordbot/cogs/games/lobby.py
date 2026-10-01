"""Shared base lobby views for multiplayer casino game sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol
import asyncio

import logfire
import nextcord
from nextcord import Embed, Message, NotFound, Forbidden, ButtonStyle, Interaction

from discordbot.typings.economy import JackpotSettlementRequest, JackpotSettlementBatchResult
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import schedule_public_message_delete
from discordbot.cogs.games.interactions import GameView, edit_game_message
from discordbot.services.economy.database import apply_jackpot_settlement_batch

if TYPE_CHECKING:
    from random import Random
    from collections.abc import Iterable

    from nextcord.ui import Button
    from nextcord.ext import commands

    from discordbot.typings.games import GameParticipant, RefreshParticipantsResult


class PrepareParticipant(Protocol):
    """Callable used by lobby join buttons to validate a participant.

    Game-specific wager / mode / insufficient-balance copy are pre-bound by
    the caller, so the callable signature stays uniform across lobbies.
    """

    async def __call__(self, interaction: Interaction[commands.Bot]) -> GameParticipant | None:
        """Returns a prepared participant or sends the interaction error."""


class RefreshParticipants(Protocol):
    """Callable used by lobby start to re-check balances.

    The wager mode is pre-bound by the caller; each participant's own wager
    rides on the participant passed in.
    """

    async def __call__(self, participants: list[GameParticipant]) -> RefreshParticipantsResult:
        """Returns refreshed participants and display names removed from the table."""


class BaseGameLobbyView(GameView):
    """Join / leave / start scaffold shared by multiplayer game lobbies."""

    interaction_failure_log = "Lobby interaction failed"
    notice_failure_log = "Failed to send lobby notice"
    max_players: ClassVar[int | None] = None

    def __init__(  # noqa: PLR0913 -- lobby owns all table dependencies
        self,
        owner: GameParticipant,
        rng: Random,
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
        timeout: int,
        extra_initial_participants: Iterable[GameParticipant] | None = None,
    ) -> None:
        """Initializes shared lobby state and registers the owner."""
        super().__init__(timeout=timeout)
        self.owner = owner
        self.rng = rng
        self.prepare_participant = prepare_participant
        self.refresh_participants = refresh_participants
        self.message: Message | None = None
        # The last press that edited the lobby; the timeout's edit and delete ride its token.
        self.last_press: Interaction[commands.Bot] | None = None
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
        try:
            await edit_game_message(
                message=self.message,
                interaction=self.last_press,
                payload={
                    "embed": embed,
                    "view": self,
                    **embed_spacer_payload(embeds=[embed], is_edit=True, target=self.message),
                },
            )
        except NotFound:
            logfire.info(
                "Lobby message gone before its timeout edit",
                channel_id=self.message.channel.id,
                message_id=self.message.id,
            )
        except Forbidden:
            # Once a press has rebound the message, an edit with no working press behind it
            # goes through the channel, which the server can shut the bot out of; the ids are
            # the whole finding.
            logfire.warn(
                "Discord refused the lobby's timeout edit",
                channel_id=self.message.channel.id,
                message_id=self.message.id,
            )
        # Broad on purpose: a raise here would only reach nextcord's timeout task, and the
        # cleanup below must still be scheduled.
        except Exception as exc:
            logfire.warn(
                "Lobby timeout edit failed",
                channel_id=self.message.channel.id,
                message_id=self.message.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        schedule_public_message_delete(
            message=self.message, user_name=self.owner.account_name, interaction=self.last_press
        )

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
                interaction=interaction, status=f"{participant.display_name} 已加入"
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
                interaction=interaction, status=f"{participant.display_name} 已離開"
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
                await self._refresh_message(interaction=interaction, status="房主餘額不足")
                return
            self._started = True
        if refreshed.dropped_names:
            names = ", ".join(refreshed.dropped_names)
            await self._send_notice(interaction=interaction, content=f"餘額不足已移出: {names}")
        try:
            started = await self._start_game(interaction=interaction)
        except (Forbidden, NotFound) as error:
            # Still started means the table is up and the refusal came from playing it.
            if self._started:
                raise
            # Expected rather than diagnosable, so the code and the ids are the whole finding: a
            # lobby someone deleted is routine, a refusal is not.
            if isinstance(error, NotFound):
                logfire.info(
                    "Lobby message gone before its start edit; lobby reopened",
                    channel_id=interaction.channel_id,
                    message_id=getattr(interaction.message, "id", None),
                    code=error.code,
                )
            else:
                logfire.warn(
                    "Discord refused the lobby's start edit; lobby reopened",
                    channel_id=interaction.channel_id,
                    message_id=getattr(interaction.message, "id", None),
                    code=error.code,
                )
            await self._send_notice(
                interaction=interaction, content="開桌失敗, 機器人無法更新這則訊息"
            )
            return
        if started:
            self.stop()

    async def _refresh_message(self, interaction: Interaction[commands.Bot], status: str) -> None:
        """Edits the lobby message with the latest participant state."""
        message = interaction.message
        if message is None:
            return
        self.message = message
        self.last_press = interaction
        embed = self._build_lobby_embed(status=status)
        await interaction.edit_original_message(
            embed=embed,
            view=self,
            **embed_spacer_payload(embeds=[embed], is_edit=True, target=message),
        )

    async def _show_table(
        self, interaction: Interaction[commands.Bot], payload: dict[str, Any]
    ) -> None:
        """Edits the lobby message into its table; an edit that never lands reopens the lobby.

        No retry here: nextcord's webhook client already retries every Discord 5xx before raising.
        """
        try:
            await interaction.edit_original_message(**payload)
        except Exception:
            self._started = False
            raise

    def _build_lobby_embed(self, status: str) -> Embed:
        """Builds the lobby embed for a concrete game type."""
        raise NotImplementedError

    async def _start_game(self, interaction: Interaction[commands.Bot]) -> bool:
        """Starts a concrete game from the current lobby participants."""
        raise NotImplementedError


class BaseJackpotLobbyView(BaseGameLobbyView):
    """Base lobby for games sharing a global jackpot pool."""

    game_id: ClassVar[str]
    ante: ClassVar[int]

    def __init__(  # noqa: PLR0913 -- jackpot lobby adds initial_jackpot on top of base deps
        self,
        owner: GameParticipant,
        rng: Random,
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
        initial_jackpot: int,
        timeout: int,
        initial_jackpot_generation: int | None = None,
    ) -> None:
        """Initializes jackpot lobby state with the live pool snapshot."""
        super().__init__(
            owner=owner,
            rng=rng,
            prepare_participant=prepare_participant,
            refresh_participants=refresh_participants,
            timeout=timeout,
        )
        self._jackpot_snapshot = initial_jackpot
        self._jackpot_generation = initial_jackpot_generation

    async def _start_game(self, interaction: Interaction[commands.Bot]) -> bool:
        """Charges antes before delegating to the jackpot game start hook."""
        message = interaction.message
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
            await self._refresh_message(interaction=interaction, status=status)
            return False
        try:
            await self._start_game_after_antes(
                interaction=interaction, message=message, final_balances=result.player_balances
            )
        except Exception:
            await self._refund_pregame_antes()
            raise
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

    async def _refund_pregame_antes(self) -> None:
        """Returns every participant's `ante` from the pool, in one DB transaction."""
        result = await apply_jackpot_settlement_batch(
            game_id=self.game_id,
            settlements=[
                JackpotSettlementRequest(
                    player_id=participant.user_id,
                    player_account_name=participant.account_name,
                    player_avatar_url=participant.avatar_url,
                    player_delta=self.ante,
                )
                for participant in self.participants
            ],
        )
        self._jackpot_snapshot = result.jackpot_balance
        self._jackpot_generation = result.jackpot_generation

    async def _start_game_after_antes(
        self,
        interaction: Interaction[commands.Bot],
        message: Message,
        final_balances: dict[int, int],
    ) -> None:
        """Starts a jackpot-backed game after ante settlement succeeds."""
        raise NotImplementedError
