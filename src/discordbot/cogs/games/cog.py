"""Discord surface for the games cog: the wagered casino tables and the fishing panel.

Registers the `/games` group (`blackjack`, `dragon_gate`, `blackjack_history`, `fishing`) plus
one `on_ready` listener. Each command posts ONE public message and hands it to a view that owns
everything after that: the lobby (`blackjack_views.py` / `dragon_gate_views.py`), the round, and
the settlement. What is left here is seat resolution — read a balance, turn it into a
`GameParticipant` at a fixed stake through `wagers.py`, and send the refusal when it does not
cover the wager. No money moves in this file, and no LLM is involved anywhere in this cog.

Two things live on the cog instance because they outlive a single table:

- The per-channel Blackjack shoe (`BlackjackShoeStore`), passed into every lobby so card
  counting carries across rounds in the same channel. It is in-memory, so a restart is a
  natural reshuffle.
- The bot's own seat. It joins every Blackjack table as an ordinary player whenever its wallet
  is positive, staking a fractional-Kelly bet read off that channel's true count. The dealer is
  the casino system, a label rather than a Discord identity, so the bot never deals.

No kill-switch and no permission gate: every command is open to anyone who can use the channel,
and the only refusals are an unparsable bet and a balance too small for the stake. `on_ready`
is the restart half of the public-message TTL and the only caller of
`delete_tracked_public_messages` in the package, so a table left on screen by a killed process
is swept once per process rather than once per reconnect.
"""

from random import SystemRandom
from functools import partial
from collections.abc import Callable

import logfire
import nextcord
from nextcord import User, Embed, Guild, Locale, Member, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.typings.games import (
    SystemIdentity,
    GameParticipant,
    GameParticipantIdentity,
    RefreshParticipantsResult,
    ParticipantPreparationResult,
)
from discordbot.utils.avatars import guild_avatar_url
from discordbot.cogs.games.shoe import BlackjackShoeStore
from discordbot.cogs.games.wagers import WagerMode, parse_wager_amount, build_wager_participant
from discordbot.cogs.games.database import fetch_recent_blackjack_rounds
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.cogs.games.bot_player import kelly_bet, count_adjusted_edge
from discordbot.utils.message_cleanup import (
    track_public_message,
    delete_tracked_public_messages,
    schedule_public_message_delete,
)
from discordbot.cogs.games.dragon_gate import ANTE
from discordbot.cogs.games.history_text import build_blackjack_history_embed
from discordbot.cogs.games.presentation import ERROR_COLOR, SYSTEM_NARRATOR_NAME
from discordbot.cogs.games.fishing.views import FishingPanelView
from discordbot.services.economy.database import get_account, get_balance
from discordbot.utils.owned_message_views import send_ephemeral_notice
from discordbot.cogs.games.blackjack_views import (
    MAX_BLACKJACK_PLAYERS,
    BlackjackLobbyView,
    build_blackjack_lobby_embed,
)
from discordbot.cogs.games.fishing.database import get_fishing_panel, get_grade_config_map
from discordbot.utils.interaction_responses import send_expiring_followup
from discordbot.cogs.games.dragon_gate_views import (
    DragonGateLobbyView,
    build_dragon_gate_lobby_embed,
    fetch_dragon_gate_jackpot_snapshot,
)
from discordbot.services.economy.presentation import CURRENCY_NAME, bold_currency
from discordbot.cogs.games.fishing.presentation import build_panel_embed


class GamesCogs(commands.Cog):
    """Slash commands for multiplayer casino games against the casino system.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        rng: System randomness used for card draws.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the cog with its own RNG and per-channel Blackjack shoe store.

        The shoe store is created here rather than per table, so every round played in a
        channel deals from the shoe the previous one left behind.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot
        self.rng = SystemRandom()
        self._startup_cleanup_done = False
        self._blackjack_shoes = BlackjackShoeStore()

    async def _system_identity(self, guild: Guild | None = None) -> SystemIdentity:
        """Returns the casino's own identity for the dealer side of a table.

        Slash commands only fire after the gateway has connected, so `self.bot.user` is
        non-None in practice; the synthetic id 0 fallback keeps the narrowing clean and stops a
        table from failing to open if Discord briefly reports no client user mid-reconnect.
        Only the name reaches the dealer seat today, so that fallback costs nothing visible.

        Args:
            guild (Guild | None): Guild whose per-guild avatar override should be used, or None
                outside a guild.

        Returns:
            The casino identity, with the fixed `SYSTEM_NARRATOR_NAME` as its display name.
        """
        if self.bot.user is None:
            return SystemIdentity(
                system_id=0, system_name=SYSTEM_NARRATOR_NAME, system_avatar_url=""
            )
        avatar_url = await guild_avatar_url(user=self.bot.user, guild=guild)
        return SystemIdentity(
            system_id=self.bot.user.id,
            system_name=SYSTEM_NARRATOR_NAME,
            system_avatar_url=avatar_url,
        )

    async def _bot_blackjack_participant(
        self, *, guild: Guild | None, table_bet: int, channel_id: int
    ) -> GameParticipant | None:
        """Builds the bot player's own seat for a Blackjack table.

        The stake is the bot's own decision, not the table's: a fractional-Kelly wager off the
        channel shoe's Hi-Lo true count, with `table_bet` only a floor that Kelly's own risk
        ceiling caps, so the bot can sit at any table without over-betting its bankroll. Clamp
        mode is what keeps `MAX_SINGLE_BET` applying to it as to any player. An empty wallet is
        an ordinary outcome, logged at info, and the human table opens without it.

        Args:
            guild (Guild | None): Guild used for the avatar lookup, or None outside a guild.
            table_bet (int): The owner's resolved table stake, used as the floor for the
                Kelly wager.
            channel_id (int): Channel whose persistent shoe supplies the true count.

        Returns:
            The bot's seat, or None when there is no client user or its wallet is empty.
        """
        bot_user = self.bot.user
        if bot_user is None:
            return None
        account = await get_account(user_id=bot_user.id)
        balance = account.balance if account is not None else 0
        if balance <= 0:
            logfire.info(
                "Bot player skipped Blackjack lobby; wallet is empty", user_id=bot_user.id
            )
            return None
        true_count = self._blackjack_shoes.true_count(channel_id=channel_id)
        decided_bet = kelly_bet(
            balance=balance,
            table_minimum=table_bet,
            edge=count_adjusted_edge(true_count=true_count),
        )
        avatar_url = await guild_avatar_url(user=bot_user, guild=guild)
        identity = GameParticipantIdentity(
            user_id=bot_user.id,
            account_name=bot_user.name,
            display_name=bot_user.display_name,
            avatar_url=avatar_url,
        )
        return build_wager_participant(
            identity=identity, balance=balance, wager=decided_bet, mode="clamp"
        )

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Deletes public game messages left behind by a previous bot process.

        `on_ready` fires again on every gateway reconnect, so the flag keeps the sweep to once
        per process; a message this process posted is deleted by its own scheduled task.
        """
        if self._startup_cleanup_done:
            return
        self._startup_cleanup_done = True
        await delete_tracked_public_messages(bot=self.bot)

    @staticmethod
    async def _identity_from_user(
        user: User | Member, guild: Guild | None = None
    ) -> GameParticipantIdentity:
        """Resolves the money-free half of a seat for a Discord user.

        The only awaiting step in building a participant, which is why it is kept apart: a
        lobby can re-stake the same identity against a fresh balance without asking Discord
        again.

        Args:
            user (User | Member): The Discord user taking the seat.
            guild (Guild | None): Guild whose per-guild avatar override should be used, or None
                outside a guild.

        Returns:
            The identity, carrying the display name and avatar the game embeds draw.
        """
        avatar_url = await guild_avatar_url(user=user, guild=guild)
        return GameParticipantIdentity(
            user_id=user.id,
            account_name=user.name,
            display_name=user.display_name,
            avatar_url=avatar_url,
        )

    async def _participant_from_user(
        self, user: User | Member, wager: int, mode: WagerMode, guild: Guild | None = None
    ) -> ParticipantPreparationResult:
        """Reads the user's balance and stakes it at the requested wager and mode.

        The observed balance rides back even when no seat could be built, because the refusal
        embed names it and re-reading would query a figure that may already have moved.

        Args:
            user (User | Member): The Discord user asking for a seat.
            wager (int): Stake to seat them at, in economy points.
            mode (WagerMode): `clamp` to reduce the stake to what they hold, `exact` to require
                the full amount (antes).
            guild (Guild | None): Guild used for the avatar lookup, or None outside a guild.

        Returns:
            The seat (None when the balance could not cover it) plus the balance that was read.
        """
        balance = await get_balance(user_id=user.id)
        return ParticipantPreparationResult(
            participant=build_wager_participant(
                identity=await self._identity_from_user(user=user, guild=guild),
                balance=balance,
                wager=wager,
                mode=mode,
            ),
            balance=balance,
        )

    async def _all_in_participant_from_user(
        self, user: User | Member, guild: Guild | None = None
    ) -> ParticipantPreparationResult:
        """Stakes the user's whole balance, which is what `bet=0` means on a table.

        All in is not unbounded: clamp mode still caps the stake at `MAX_SINGLE_BET`, so the
        seat can end up wagering less than the balance it was built from.

        Args:
            user (User | Member): The Discord user opening the table.
            guild (Guild | None): Guild used for the avatar lookup, or None outside a guild.

        Returns:
            The seat (None when the balance is zero) plus the balance that was read.
        """
        balance = await get_balance(user_id=user.id)
        return ParticipantPreparationResult(
            participant=build_wager_participant(
                identity=await self._identity_from_user(user=user, guild=guild),
                balance=balance,
                wager=balance,
                mode="clamp",
            ),
            balance=balance,
        )

    async def _prepare_participant(
        self,
        interaction: Interaction[commands.Bot],
        wager: int,
        mode: WagerMode,
        insufficient_embed_builder: Callable[[int], Embed],
    ) -> GameParticipant | None:
        """Seats a user who pressed a lobby Join button.

        Bound to a game's wager, mode and refusal copy with `functools.partial` before it is
        handed to a lobby, which is what lets `PrepareParticipant` stay one uniform signature.
        Answers the interaction itself on refusal, with an ephemeral followup so the table is
        not littered with other players' balances.

        Args:
            interaction (Interaction[commands.Bot]): The already-deferred Join interaction.
            wager (int): Stake this table seats players at.
            mode (WagerMode): `clamp` for a table stake, `exact` for an ante.
            insufficient_embed_builder (Callable[[int], Embed]): Builds the game's own refusal
                embed from the observed balance.

        Returns:
            The seat, or None when there is no user or the balance could not cover the wager.
        """
        if interaction.user is None:
            return None
        result = await self._participant_from_user(
            user=interaction.user,
            wager=wager,
            mode=mode,
            guild=getattr(interaction, "guild", None),
        )
        if result.participant is None:
            embed = insufficient_embed_builder(result.balance)
            await interaction.followup.send(
                embed=embed,
                ephemeral=True,
                **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
            )
        return result.participant

    async def _refresh_participants(
        self, participants: list[GameParticipant], mode: WagerMode
    ) -> RefreshParticipantsResult:
        """Re-stakes every queued seat against a freshly read balance, just before the deal.

        A player who spent their money between joining and Start is dropped rather than seated
        at a stake they can no longer cover, and their display name goes back so the lobby can
        say who left. Identity is rebuilt from the queued seat rather than from Discord, so the
        start path costs one balance query per player and no API call.

        Args:
            participants (list[GameParticipant]): The seats queued in the lobby, in join order.
            mode (WagerMode): `clamp` for a table stake, `exact` for an ante.

        Returns:
            The seats that survived the re-check, plus the display names of those dropped.
        """
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
                wager=participant.bet,
                mode=mode,
            )
            if refreshed_participant is None:
                dropped.append(participant.display_name)
                continue
            refreshed.append(refreshed_participant)
        return RefreshParticipantsResult(participants=refreshed, dropped_names=dropped)

    def _insufficient_balance_embed(self, balance: int) -> Embed:
        """Builds the refusal embed for a clamp-mode table the player cannot afford at all.

        Clamp mode only fails at a zero balance, so the copy points at earning more rather than
        at lowering the bet.

        Args:
            balance (int): The balance that was observed, shown to the player.

        Returns:
            The refusal embed.
        """
        return Embed(
            title="餘額不足",
            description=(
                f"### {bold_currency(amount=balance, compact=True)}\n"
                f"沒有可下注的{CURRENCY_NAME}\n"
                f"-# 跟機器人聊天可以累積{CURRENCY_NAME}"
            ),
            color=ERROR_COLOR,
        )

    @staticmethod
    def _invalid_bet_embed() -> Embed:
        """Builds the validation embed for bet text that did not parse.

        Returns:
            The refusal embed, which also states that `0` means all in.
        """
        return Embed(
            title="下注格式錯誤",
            description="請輸入非負整數，可以加逗號，例如 `1,000`；輸入 `0` 會 all in。",
            color=ERROR_COLOR,
        )

    def _dragon_gate_insufficient_balance_embed(self, balance: int) -> Embed:
        """Builds the refusal embed for a 射龍門 ante the player cannot cover.

        The ante is `exact` mode, so unlike a table stake it is never reduced; the copy names
        the fixed amount and where it goes.

        Args:
            balance (int): The balance that was observed, shown to the player.

        Returns:
            The refusal embed.
        """
        return Embed(
            title="餘額不足",
            description=(
                f"### {bold_currency(amount=balance, compact=True)}\n"
                f"射龍門入場費固定 {bold_currency(amount=ANTE, compact=True)} 進彩金池\n"
                f"-# 跟機器人聊天可以累積{CURRENCY_NAME}"
            ),
            color=ERROR_COLOR,
        )

    @nextcord.slash_command(
        name="games",
        description="Game commands.",
        name_localizations={Locale.zh_TW: "小遊戲", Locale.ja: "ゲーム"},
        description_localizations={Locale.zh_TW: "小遊戲指令", Locale.ja: "ゲームコマンド。"},
        nsfw=False,
    )
    async def games(self, interaction: Interaction[commands.Bot]) -> None:
        """Group node for the game commands; Discord only ever dispatches its subcommands.

        Args:
            interaction (Interaction[commands.Bot]): Never delivered, since the group cannot be
                invoked on its own.
        """

    @games.subcommand(
        name="blackjack",
        description="Open a 21 lobby; the casino is the dealer and the bot joins as a player.",
        name_localizations={Locale.zh_TW: "二十一點", Locale.ja: "ブラックジャック"},
        description_localizations={
            Locale.zh_TW: "開一桌 21 點 lobby",
            Locale.ja: "21（ブラックジャック）の lobby を開きます。",
        },
    )
    async def blackjack(
        self,
        interaction: Interaction[commands.Bot],
        bet: str = SlashOption(
            name="bet",
            description=f"Table stake in {CURRENCY_NAME}; enter 0 to go all in. Commas are allowed.",
            name_localizations={Locale.zh_TW: "下注", Locale.ja: "賭け金"},
            description_localizations={
                Locale.zh_TW: f"這桌的基本下注{CURRENCY_NAME}; 可加逗號，輸入 0 會直接 all in",
                Locale.ja: f"Table の基本賭け金{CURRENCY_NAME}; カンマ可、0 で all in。",
            },
            required=True,
            min_length=1,
        ),
    ) -> None:
        """Opens a public Blackjack lobby that the owner then starts.

        The owner's resolved stake becomes the table stake every other seat is prepared at, so
        it is settled here before the lobby exists. The bot's seat is decided at the same point
        and seeded into the lobby, and the shoe store rides along so the round deals from the
        channel's carried-over shoe. Unparsable text is refused ephemerally before the defer;
        an empty wallet is refused after it, so that refusal is public and is scheduled for
        deletion instead. The lobby message itself is tracked so a restart can clean it up.

        Args:
            interaction (Interaction[commands.Bot]): The interaction that triggered the command.
            bet (str): Raw wager text, taken as text so it can exceed Discord's integer option
                ceiling; `0` means all in.
        """
        if interaction.user is None:
            return
        wager = parse_wager_amount(raw_amount=bet)
        if wager is None:
            embed = self._invalid_bet_embed()
            await interaction.response.send_message(
                embed=embed,
                ephemeral=True,
                **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
            )
            return

        await interaction.response.defer()

        guild = getattr(interaction, "guild", None)
        if wager == 0:
            participant_result = await self._all_in_participant_from_user(
                user=interaction.user, guild=guild
            )
        else:
            participant_result = await self._participant_from_user(
                user=interaction.user, wager=wager, mode="clamp", guild=guild
            )
        owner = participant_result.participant
        if owner is None:
            embed = self._insufficient_balance_embed(balance=participant_result.balance)
            message = await interaction.followup.send(
                embed=embed,
                wait=True,
                **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
            )
            schedule_public_message_delete(message=message, user_name=interaction.user.name)
            return

        table_bet = owner.bet
        channel_id = getattr(interaction, "channel_id", None) or 0
        system_identity = await self._system_identity(guild=guild)
        bot_participant = await self._bot_blackjack_participant(
            guild=guild, table_bet=table_bet, channel_id=channel_id
        )
        extra_initial_participants: list[GameParticipant] = (
            [bot_participant] if bot_participant is not None else []
        )
        view = BlackjackLobbyView(
            owner=owner,
            requested_bet=table_bet,
            rng=self.rng,
            system_name=system_identity.system_name,
            system_avatar_url=system_identity.system_avatar_url,
            prepare_participant=partial(
                self._prepare_participant,
                wager=table_bet,
                mode="clamp",
                insufficient_embed_builder=self._insufficient_balance_embed,
            ),
            refresh_participants=partial(self._refresh_participants, mode="clamp"),
            bot_user_id=bot_participant.user_id if bot_participant is not None else None,
            extra_initial_participants=extra_initial_participants,
            shoe_store=self._blackjack_shoes,
            channel_id=channel_id,
        )
        embed = build_blackjack_lobby_embed(
            owner=owner,
            participants=view.participants,
            requested_bet=table_bet,
            max_players=MAX_BLACKJACK_PLAYERS,
        )
        message = await interaction.followup.send(
            embed=embed,
            view=view,
            wait=True,
            **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
        )
        await track_public_message(message=message, user_name=owner.account_name)
        view.message = message

    @games.subcommand(
        name="dragon_gate",
        description="Open an In-Between table with a shared jackpot pool.",
        name_localizations={Locale.zh_TW: "射龍門", Locale.ja: "インビトウィーン"},
        description_localizations={
            Locale.zh_TW: "開一桌共享全域彩金池的射龍門",
            Locale.ja: "共有ジャックポットのインビトウィーン table を開きます。",
        },
    )
    async def dragon_gate(self, interaction: Interaction[commands.Bot]) -> None:
        """Opens a public 射龍門 lobby that the owner then starts.

        The stake is the fixed `ANTE` in `exact` mode for everyone, so an owner who cannot pay
        it in full is refused instead of being seated at less. The jackpot is snapshotted once
        here and its generation carried into the lobby, so a later payout only claims from the
        pool generation this table observed.

        Args:
            interaction (Interaction[commands.Bot]): The interaction that triggered the command.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return

        participant_result = await self._participant_from_user(
            user=interaction.user,
            wager=ANTE,
            mode="exact",
            guild=getattr(interaction, "guild", None),
        )
        owner = participant_result.participant
        if owner is None:
            embed = self._dragon_gate_insufficient_balance_embed(
                balance=participant_result.balance
            )
            message = await interaction.followup.send(
                embed=embed,
                wait=True,
                **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
            )
            schedule_public_message_delete(message=message, user_name=interaction.user.name)
            return

        system_identity = await self._system_identity(guild=getattr(interaction, "guild", None))
        initial_jackpot = await fetch_dragon_gate_jackpot_snapshot()
        view = DragonGateLobbyView(
            owner=owner,
            rng=self.rng,
            system_name=system_identity.system_name,
            system_avatar_url=system_identity.system_avatar_url,
            prepare_participant=partial(
                self._prepare_participant,
                wager=ANTE,
                mode="exact",
                insufficient_embed_builder=self._dragon_gate_insufficient_balance_embed,
            ),
            refresh_participants=partial(self._refresh_participants, mode="exact"),
            initial_jackpot=initial_jackpot.balance,
            initial_jackpot_generation=initial_jackpot.generation,
        )
        embed = build_dragon_gate_lobby_embed(
            owner=owner, participants=view.participants, jackpot=initial_jackpot.balance
        )
        message = await interaction.followup.send(
            embed=embed,
            view=view,
            wait=True,
            **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
        )
        await track_public_message(message=message, user_name=owner.account_name)
        view.message = message

    @games.subcommand(
        name="blackjack_history",
        description="Show a player's recent Blackjack rounds: hands, bets, and results.",
        name_localizations={Locale.zh_TW: "二十一點紀錄", Locale.ja: "ブラックジャック履歴"},
        description_localizations={
            Locale.zh_TW: "查看某位玩家近期的 21 點對局紀錄：手牌、下注與結果",
            Locale.ja: "プレイヤーの最近のブラックジャックの手札・賭け金・結果を表示します。",
        },
    )
    async def blackjack_history(
        self,
        interaction: Interaction[commands.Bot],
        member: Member | None = SlashOption(
            name="member",
            description="Player to inspect; defaults to yourself.",
            name_localizations={Locale.zh_TW: "玩家", Locale.ja: "プレイヤー"},
            description_localizations={
                Locale.zh_TW: "要查看的玩家；預設是自己",
                Locale.ja: "表示するプレイヤー。省略時は自分。",
            },
            required=False,
            default=None,
        ),
        count: int = SlashOption(
            name="count",
            description="How many recent rounds to show (1-50, default 10).",
            name_localizations={Locale.zh_TW: "場數", Locale.ja: "件数"},
            description_localizations={
                Locale.zh_TW: "要顯示的最近場數（1-50，預設 10）",
                Locale.ja: "表示する直近の件数（1〜50、既定 10）。",
            },
            required=False,
            default=10,
            min_value=1,
            max_value=50,
        ),
    ) -> None:
        """Publicly posts a player's recent Blackjack rounds as a text table.

        Read-only and open to anyone: history is a public record of a table other people sat
        at. The posted embed expires on the shared public-message TTL like the tables do.

        Args:
            interaction (Interaction[commands.Bot]): The interaction that triggered the command.
            member (Member | None): Player to inspect; None reads the caller's own rounds.
            count (int): How many of the most recent rounds to render, bounded 1-50 by the
                option itself.
        """
        if interaction.user is None:
            await send_ephemeral_notice(
                interaction=interaction,
                content="無法辨識使用者，請稍後再試",
                log_message="Failed to send Blackjack history missing-user notice",
            )
            return
        await interaction.response.defer()
        target = member or interaction.user
        target_name = getattr(target, "display_name", "") or target.name
        records = await fetch_recent_blackjack_rounds(user_id=target.id, limit=count)
        embed = build_blackjack_history_embed(player_name=target_name, records=records)
        await send_expiring_followup(interaction=interaction, embed=embed)

    @games.subcommand(
        name="fishing",
        description="Open your fishing panel: buy gear, cast a line, and recycle currency.",
        name_localizations={Locale.zh_TW: "釣魚", Locale.ja: "釣り"},
        description_localizations={
            Locale.zh_TW: "打開釣魚面板：買釣具、拋竿，把歡樂豆釣回來",
            Locale.ja: "釣りパネルを開いて、道具購入とキャストで遊びます。",
        },
    )
    async def fishing(self, interaction: Interaction[commands.Bot]) -> None:
        """Opens the personal fishing panel as one public message the view then edits in place.

        Single-player and unwagered, so nothing here resolves a seat; the panel view is locked
        to the opener and deletes itself once idle. The message is tracked so a restart can
        clean up a panel this process will never time out.

        Args:
            interaction (Interaction[commands.Bot]): The interaction that triggered the command.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return
        panel = await get_fishing_panel(user_id=interaction.user.id)
        grade_map = await get_grade_config_map()
        embed = build_panel_embed(panel=panel, grade_map=grade_map)
        view = FishingPanelView(owner_id=interaction.user.id)
        message = await interaction.followup.send(
            embed=embed,
            view=view,
            wait=True,
            **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
        )
        await track_public_message(message=message, user_name=interaction.user.name)
        view.message = message


def setup(bot: commands.Bot) -> None:
    """Registers the games cog. Sync, so the loader adds it before the gateway connects.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(GamesCogs(bot), override=True)
