"""The join / leave / start scaffold both wagered tables open on, carrying no game rules.

`BlackjackLobbyView` and `DragonGateLobbyView` are the same view up to two hooks: how a roster
renders as an embed, and what happens once Start is pressed. The seat map, the lock every
mutation runs under, the owner-only Start, the balance re-check just before the deal, and the
idle timeout that greys the buttons and queues the message for deletion all live here, so a
third table would cost an embed builder and a hand-off rather than a second copy of the seating
rules.

Nothing here knows what a wager is. A game's stake, its mode (`clamp` for a table bet, `exact`
for an ante) and its own insufficient-balance copy are bound into the two callables the cog
hands in — `PrepareParticipant` and `RefreshParticipants`, both a `functools.partial` of a games
cog method — which is what lets the refusal a player sees name their own game while the lobby
stays generic.

One public message carries the whole session: the lobby edits it on every roster change and then
hands that same message to the game view, so the table opens where the lobby was rather than as
a second message.

The base lobby moves no money at all; a Blackjack table debits nothing until the round resolves.
`BaseJackpotLobbyView` is the second layer, for a table backed by the shared `jackpot_pool` row
instead of the casino ledger, and it does move money: every seat's `ante` is charged through ONE
`apply_jackpot_settlement_batch` with `require_full_debit=True`, so a seat that cannot cover it
rejects the batch whole with nothing committed. The lobby then drops those seats, re-renders and
stays startable, rather than opening a table someone has already paid into. Only once that
settlement commits does the game hook run, which is why its hand-off edit is the one edit in the
cog that must not be lost.
"""

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
from discordbot.cogs.games.interactions import disable_view_components
from discordbot.services.economy.database import apply_jackpot_settlement_batch
from discordbot.utils.owned_message_views import send_ephemeral_notice

if TYPE_CHECKING:
    from random import Random
    from collections.abc import Iterable

    from nextcord.ext import commands

    from discordbot.typings.games import GameParticipant, RefreshParticipantsResult


class PrepareParticipant(Protocol):
    """Turns a user who pressed Join into a seat, or refuses them.

    The game-specific half — the stake, its clamp-or-exact mode, and the insufficient-balance
    copy — is bound by the caller (via `functools.partial`) so every lobby calls one uniform
    shape. An implementation owns the refusal it shows: the lobby defers the interaction before
    calling one, and says nothing itself when None comes back.
    """

    async def __call__(self, interaction: Interaction[commands.Bot]) -> GameParticipant | None:
        """Prepares a seat for the user behind a Join press.

        Args:
            interaction (Interaction[commands.Bot]): The already-deferred Join interaction.

        Returns:
            The prepared seat, or None once the refusal has been shown to the user.
        """


class RefreshParticipants(Protocol):
    """Re-stakes every queued seat against a freshly read balance, just before the deal.

    Bound to the game's wager and mode by the caller, like `PrepareParticipant`. It is handed no
    interaction and shows nothing: the lobby names the dropped players itself, out of the display
    names this returns.
    """

    async def __call__(self, participants: list[GameParticipant]) -> RefreshParticipantsResult:
        """Re-checks the queued seats against current balances.

        Args:
            participants (list[GameParticipant]): The seats queued in the lobby, in join order.

        Returns:
            The seats that can still cover the stake, plus the display names of those dropped.
        """


class BaseGameLobbyView(View):
    """Join / leave / start scaffold shared by the multiplayer game lobbies.

    Subclasses must override:
      - `_build_lobby_embed(status: str) -> Embed` — used by refresh + timeout
      - `_start_game(message: Message | None) -> bool` — invoked after Start

    Optional class attribute:
      - `max_players: ClassVar[int | None]` — None means unlimited

    Every seat mutation runs under `_lock` and re-reads `_started` inside it, so a Join racing
    the Start press either takes a seat the balance re-check then sees, or is refused. The view
    stops only once `_start_game` reports the table is on screen; anything else leaves the lobby
    open and pressable.
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
        """Seats the owner plus any pre-seated players, and arms the idle timeout.

        `extra_initial_participants` is how a table seats someone who never pressed Join, which
        is how the Blackjack bot takes its place; a duplicate of the owner in it is ignored so
        the owner keeps the first seat and the join order the embeds render. `self.message` is
        left for whoever sends the lobby message to fill in.

        Args:
            owner (GameParticipant): The seat that opened the lobby; alone may press Start, and
                may not leave.
            rng (Random): Random source handed to the game this lobby starts.
            system_name (str): Casino display name the game's dealer side is drawn with.
            system_avatar_url (str): Casino avatar carried through to the game view.
            prepare_participant (PrepareParticipant): Join-time validation, bound to this game's
                stake by the cog.
            refresh_participants (RefreshParticipants): Start-time balance re-check, bound to the
                same stake.
            timeout (int): Idle seconds before `on_timeout` closes a lobby that never started.
            extra_initial_participants (Iterable[GameParticipant] | None): Seats present before
                anyone joins.
        """
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
        """The queued seats, owner first and then in join order.

        Returns:
            A copy of the seat map's values, not a live view of it.
        """
        return list(self._participants.values())

    async def on_timeout(self) -> None:
        """Closes a lobby nobody started, and queues its message for deletion.

        Does nothing once Start has taken: from then on the game view owns the message and its
        own timeout. The closing edit is best-effort because a message deleted or made
        uneditable meanwhile must not cost the cleanup scheduled after it.
        """
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
        """Seats the pressing user, refusing a started, full or already-seated table.

        The whole check-and-seat runs under `_lock`, so two people pressing Join for the last
        seat cannot both take it. Deferring is left until after the cheap refusals because it is
        `prepare_participant` that needs it: its own refusal is an ephemeral followup, which only
        becomes legal once the interaction has been answered.

        Args:
            _button (Button[BaseGameLobbyView]): The Join button, unused.
            interaction (Interaction[commands.Bot]): The Join press.
        """
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
        """Removes the pressing user's seat, refusing the owner and a started table.

        The owner is refused because Start is theirs alone, so a lobby they left could never be
        started. Runs under `_lock`, and defers only once a seat is really being removed.

        Args:
            _button (Button[BaseGameLobbyView]): The Leave button, unused.
            interaction (Interaction[commands.Bot]): The Leave press.
        """
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
        """Re-checks every balance and hands the table over, for the owner only.

        Seats that can no longer cover the stake are dropped by `refresh_participants` and named
        back to whoever pressed Start; an owner among them stops the start and leaves the lobby
        open. `_started` is set inside the lock and `_start_game` then runs outside it, so the
        settlement and the hand-off edit do not hold the lock while a second press is already
        refused by the flag. A hook returning False must have cleared `_started` itself, since
        the view is stopped only once the table is really on screen.

        Args:
            _button (Button[BaseGameLobbyView]): The Start button, unused.
            interaction (Interaction[commands.Bot]): The Start press, deferred before the lock is
                taken.
        """
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
        """Re-renders the lobby message with the roster as it now stands.

        Adopts the passed message as the lobby's own, so `on_timeout` closes the message the last
        press came from. The spacer rides through `embed_spacer_payload(..., target=)` so a lobby
        edited on every press retains the uploaded file instead of re-uploading it into Discord
        error 400009.

        Args:
            message (Message | None): The lobby message to edit; None skips the render.
            status (str): Status line the embed shows.
        """
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
        """Tells the pressing user, privately, why their press was refused.

        Never raises: a notice explains a press that has already been rejected, so a delivery
        failure is logged inside `send_ephemeral_notice` rather than surfacing as a failed
        interaction.

        Args:
            interaction (Interaction[commands.Bot]): The press being refused.
            content (str): The refusal text, shown to that user only.
        """
        await send_ephemeral_notice(
            interaction=interaction, content=content, log_message="Failed to send lobby notice"
        )

    async def on_error(
        self,
        error: Exception,
        item: Item[BaseGameLobbyView],
        interaction: Interaction[commands.Bot],
    ) -> None:
        """Records a component failure nextcord would otherwise only print to stderr.

        Args:
            error (Exception): The exception the component callback raised.
            item (Item[BaseGameLobbyView]): The component that raised; its label is read
                defensively, since not every item carries one.
            interaction (Interaction[commands.Bot]): The press being handled.
        """
        logfire.error(
            "Lobby interaction failed",
            item_label=getattr(item, "label", None),
            user_id=getattr(interaction.user, "id", None),
            _exc_info=(type(error), error, error.__traceback__),
        )

    def _disable_buttons(self) -> None:
        """Greys the buttons out, leaving them on screen for the closing render."""
        disable_view_components(children=self.children, component_types=(Button,))

    def _build_lobby_embed(self, status: str = "等待玩家加入") -> Embed:
        """Renders the current roster as this game's own lobby embed.

        Called on every roster change and once more when the lobby times out, so it must render
        from `participants` alone.

        Args:
            status (str): Status line for the description, e.g. who just joined, or why Start did
                not take.

        Returns:
            The lobby embed for the roster as it stands.

        Raises:
            NotImplementedError: The subclass did not override the hook.
        """
        raise NotImplementedError

    async def _start_game(self, message: Message | None) -> bool:
        """Starts this game from the queued seats, taking the lobby message over.

        Called with `_started` already set, and outside the lobby lock. Returning False leaves
        the lobby alive and pressable, so an implementation that refuses to start has to clear
        `_started` itself.

        Args:
            message (Message | None): The lobby message the game view takes over.

        Returns:
            True once the game is on screen, False when the lobby was left open.

        Raises:
            NotImplementedError: The subclass did not override the hook.
        """
        raise NotImplementedError


class BaseJackpotLobbyView(BaseGameLobbyView):
    """Base lobby for a game backed by the shared jackpot pool rather than the casino ledger.

    On Start each participant is charged `ante` into the pool through one
    `apply_jackpot_settlement_batch`, all or nothing, before the table begins; only then does
    `_start_game_after_antes` run. Subclasses declare `game_id` / `ante` and override that hook
    alongside `_build_lobby_embed`.

    The pool balance and the generation it was read at ride on the view: the balance is what the
    lobby embed shows, and the generation travels into the table so a payout claimed against a
    pool that has since been drained and reseeded is refused instead of paid out of the new seed.
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
        """Seats the lobby and records the pool the table is being opened against.

        Both pool fields are re-read from the ante settlement before the table starts, so what
        the game view receives is the pool after the antes rather than this snapshot.

        Args:
            owner (GameParticipant): The seat that opened the lobby.
            rng (Random): Random source handed to the game this lobby starts.
            system_name (str): Casino display name the game's dealer side is drawn with.
            system_avatar_url (str): Casino avatar carried through to the game view.
            prepare_participant (PrepareParticipant): Join-time validation, bound to the ante.
            refresh_participants (RefreshParticipants): Start-time balance re-check, bound to the
                same ante.
            initial_jackpot (int): Pool balance observed when the lobby was opened.
            timeout (int): Idle seconds before `on_timeout` closes a lobby that never started.
            initial_jackpot_generation (int | None): Generation that balance was read at, or None
                to claim against whichever generation is live at settlement time.
            extra_initial_participants (Iterable[GameParticipant] | None): Seats present before
                anyone joins.
        """
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
        """Charges every ante in one transaction, then starts the table.

        A rejected batch means at least one seat could not cover the ante and nothing was
        committed, so the lobby reopens instead of starting: `_started` goes back to False, every
        rejected seat other than the owner is dropped, and the re-render names them (or says the
        owner is the one short). A missing message is the same kind of non-start, since the
        hand-off would have nothing to edit.

        Args:
            message (Message | None): The lobby message the table takes over.

        Returns:
            True once the antes are committed and the table is on screen, False when the lobby
            was left open.
        """
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
        """Debits `ante` from every seat into the pool, in one transaction.

        `require_full_debit` makes the batch all-or-nothing: a seat short of the ante rejects the
        whole batch with nothing committed, so a table can never be half paid for. No generation
        guard is sent, because that guard only refuses a payout and every ante here is a debit.
        The returned pool balance and generation are adopted either way — on a rejection they are
        the live pool the reopened lobby's embed should show.

        Returns:
            The batch result, carrying the post-ante balances or the rejected player ids.
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
        """Deals the concrete game once every ante is committed.

        Reached only after the settlement succeeded, so the money is already in the pool: an
        implementation that gives up here leaves a table that has been paid for, which is why its
        hand-off edit retries rather than dropping the round.

        Args:
            message (Message): The lobby message, reused as the table message.
            final_balances (dict[int, int]): Post-ante wallet balance per player, from the
                settlement that charged them.

        Raises:
            NotImplementedError: The subclass did not override the hook.
        """
        raise NotImplementedError
