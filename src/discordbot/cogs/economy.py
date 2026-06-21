"""Slash commands that surface point balances, leaderboards, transfers, and loans."""

from io import BytesIO
from datetime import UTC, datetime

import nextcord
from nextcord import File, Locale, Member, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.cogs._economy import embeds
from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.config import EconomyConfig
from discordbot.typings.economy import (
    VIP_PURCHASE_COST,
    BASE_CHECKIN_REWARD_AMOUNT,
    DEFAULT_LOAN_MONTHLY_RATE_BPS,
    LoanLenderType,
)
from discordbot.cogs._economy.views import CreditLoanDecisionView, CentralBankLoanDecisionView
from discordbot.cogs._economy.boards import (
    LOSS_LEADERBOARD_BOARD_FILENAME,
    BALANCE_LEADERBOARD_BOARD_FILENAME,
    build_loss_leaderboard_board_image,
    build_balance_leaderboard_board_image,
)
from discordbot.cogs._stock.database import get_stock_portfolio
from discordbot.utils.amount_parsing import parse_decimal_amount
from discordbot.cogs._economy.database import (
    top_n,
    buy_vip,
    checkin,
    get_vip,
    transfer,
    get_admin,
    top_losers,
    get_account,
    get_balance,
    get_portfolio,
    adjust_balance,
    get_casino_ledger,
    get_central_banker,
    call_personal_loans,
    list_loan_contracts,
    repay_personal_loans,
    call_central_bank_loans,
    get_central_bank_status,
    repay_central_bank_loans,
    monthly_rate_percent_to_bps,
    create_personal_loan_request,
    create_central_bank_loan_request,
)
from discordbot.cogs._economy.interactions import (
    send_private_followup,
    send_expiring_followup,
    send_ephemeral_response,
    send_loan_request_followup,
)
from discordbot.cogs._economy.presentation import CURRENCY_NAME, currency_text


def _parse_positive_amount(raw_amount: str | None) -> int | None:
    """Parses user-entered positive amount text with optional comma separators."""
    amount = parse_decimal_amount(raw=raw_amount)
    if amount is None or amount <= 0:
        return None
    return amount


def _parse_collect_amount(raw_amount: str | None) -> tuple[bool, int | None]:
    """Parses optional collection-amount text; blank or 0 collects all owed.

    Returns ``(is_valid, amount)`` where ``amount`` is ``None`` when collecting
    everything owed. ``is_valid`` is ``False`` only for malformed text.
    """
    if not (raw_amount or "").strip():
        return True, None
    amount = parse_decimal_amount(raw=raw_amount)
    if amount is None:
        return False, None
    return True, amount or None


class EconomyCogs(commands.Cog):
    """Player-facing point balance, leaderboards, loans, VIP, and check-in commands.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initialises the EconomyCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.economy_config = EconomyConfig()

    @nextcord.slash_command(
        name="admin",
        description=f"Run admin-only {CURRENCY_NAME} maintenance operations.",
        name_localizations={Locale.zh_TW: "管理員", Locale.ja: "管理者"},
        description_localizations={
            Locale.zh_TW: f"執行管理員限定的{CURRENCY_NAME}維護操作",
            Locale.ja: f"管理者専用の{CURRENCY_NAME}メンテナンス操作を実行します。",
        },
        nsfw=False,
    )
    async def admin(self, interaction: Interaction) -> None:
        """Slash command group for economy admin operations."""

    @admin.subcommand(
        name="refund_tax",
        description=f"Admin-only: credit {CURRENCY_NAME} to a member or bot.",
        name_localizations={Locale.zh_TW: "退稅", Locale.ja: "税還付"},
        description_localizations={
            Locale.zh_TW: f"管理員限定：無條件增加某位成員或 bot 的{CURRENCY_NAME}",
            Locale.ja: f"管理者専用：メンバーまたは bot に{CURRENCY_NAME}を付与します。",
        },
    )
    async def admin_refund_tax(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description=f"The member or bot to receive the {CURRENCY_NAME}.",
            name_localizations={Locale.zh_TW: "對象", Locale.ja: "対象"},
            description_localizations={
                Locale.zh_TW: f"要增加{CURRENCY_NAME}的成員或 bot",
                Locale.ja: f"{CURRENCY_NAME}を受け取るメンバーまたは bot。",
            },
            required=True,
        ),
        amount: str = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to add. Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要增加的{CURRENCY_NAME}，可加逗號",
                Locale.ja: f"追加する{CURRENCY_NAME}。カンマ可。",
            },
            required=True,
            min_length=1,
        ),
    ) -> None:
        """Credits points to a member through the manual-adjustment audit path."""
        parsed_amount = _parse_positive_amount(raw_amount=amount)
        if parsed_amount is None:
            await send_ephemeral_response(
                interaction=interaction, embed=embeds.build_invalid_amount_embed(title="退稅失敗")
            )
            return
        await self._run_admin_adjustment(
            interaction=interaction,
            member=member,
            action="refund_tax",
            title="退稅完成",
            delta=parsed_amount,
        )

    @admin.subcommand(
        name="collect_tax",
        description=f"Admin-only: debit {CURRENCY_NAME} from a member or bot.",
        name_localizations={Locale.zh_TW: "收稅", Locale.ja: "徴税"},
        description_localizations={
            Locale.zh_TW: f"管理員限定：無條件扣除某位成員或 bot 的{CURRENCY_NAME}",
            Locale.ja: f"管理者専用：メンバーまたは bot から{CURRENCY_NAME}を徴収します。",
        },
    )
    async def admin_collect_tax(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description=f"The member or bot to debit the {CURRENCY_NAME} from.",
            name_localizations={Locale.zh_TW: "對象", Locale.ja: "対象"},
            description_localizations={
                Locale.zh_TW: f"要扣除{CURRENCY_NAME}的成員或 bot",
                Locale.ja: f"{CURRENCY_NAME}を徴収するメンバーまたは bot。",
            },
            required=True,
        ),
        amount: str = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to debit. Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要扣除的{CURRENCY_NAME}，可加逗號",
                Locale.ja: f"徴収する{CURRENCY_NAME}。カンマ可。",
            },
            required=True,
            min_length=1,
        ),
    ) -> None:
        """Debits points from a member through the manual-adjustment audit path."""
        parsed_amount = _parse_positive_amount(raw_amount=amount)
        if parsed_amount is None:
            await send_ephemeral_response(
                interaction=interaction, embed=embeds.build_invalid_amount_embed(title="收稅失敗")
            )
            return
        await self._run_admin_adjustment(
            interaction=interaction,
            member=member,
            action="collect_tax",
            title="收稅完成",
            delta=-parsed_amount,
        )

    async def _run_admin_adjustment(
        self, interaction: Interaction, member: Member, action: str, title: str, delta: int
    ) -> None:
        """Runs a gated admin balance adjustment and publishes successful results."""
        if interaction.user is None:
            return
        actor = interaction.user
        guild = getattr(interaction, "guild", None)
        actor_avatar_url = await guild_avatar_url(user=actor, guild=guild)
        if not await get_admin(user_id=actor.id):
            await interaction.response.defer(ephemeral=True)
            await send_private_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="權限不足",
                    description="### 只有 economy admin 可以執行這個操作",
                    author_name=actor.display_name,
                    author_icon_url=actor_avatar_url,
                ),
            )
            return
        await interaction.response.defer()
        member_avatar_url = await guild_avatar_url(user=member, guild=guild)
        result = await adjust_balance(
            user_id=member.id,
            name=member.name,
            delta=delta,
            allow_negative=False,
            avatar_url=member_avatar_url,
        )
        embed = embeds.build_admin_adjustment_embed(
            title=title,
            member_mention=member.mention,
            actor_name=actor.display_name,
            actor_avatar_url=actor_avatar_url,
            member_avatar_url=member_avatar_url,
            requested_delta=delta,
            result=result,
            is_collect_clamped=(action == "collect_tax" and result.applied_delta != delta),
        )
        await send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="balance",
        description=f"Check a member's {CURRENCY_NAME} balance, loans, stocks, and VIP status.",
        name_localizations={Locale.zh_TW: "餘額", Locale.ja: "残高"},
        description_localizations={
            Locale.zh_TW: f"查詢成員的{CURRENCY_NAME}餘額、借貸、股票與 VIP 狀態",
            Locale.ja: f"member の{CURRENCY_NAME}残高、loan、stock、VIP 状態を確認します。",
        },
        nsfw=False,
    )
    async def balance(
        self,
        interaction: Interaction,
        member: Member | None = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="Member to inspect; defaults to yourself.",
            name_localizations={Locale.zh_TW: "成員", Locale.ja: "メンバー"},
            description_localizations={
                Locale.zh_TW: "要查看的成員；預設是自己",
                Locale.ja: "表示する member。省略時は自分。",
            },
            required=False,
            default=None,
        ),
    ) -> None:
        """Replies with a member's balance, loans, stocks, and VIP status.

        Args:
            interaction: The interaction that triggered the command.
            member: Optional member to inspect.
        """
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        target = member or interaction.user
        portfolio = await get_portfolio(user_id=target.id)
        stock_portfolio = await get_stock_portfolio(user_id=target.id)
        is_vip = await get_vip(user_id=target.id)
        age_days = (datetime.now(tz=UTC) - target.created_at).days
        embed = embeds.build_balance_embed(
            display_name=target.display_name,
            avatar_url=target.display_avatar.url,
            portfolio=portfolio,
            stock_portfolio=stock_portfolio,
            is_vip=is_vip,
            age_days=age_days,
        )
        await send_private_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="leaderboard",
        description=f"Show the global top {CURRENCY_NAME} holders.",
        name_localizations={Locale.zh_TW: "排行榜", Locale.ja: "リーダーボード"},
        description_localizations={
            Locale.zh_TW: f"顯示全域 {CURRENCY_NAME}前 10 名",
            Locale.ja: f"グローバル{CURRENCY_NAME}トップ10を表示します。",
        },
        nsfw=False,
    )
    async def leaderboard(self, interaction: Interaction) -> None:
        """Replies with the top 10 point holders.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        rows = await top_n(limit=10)
        if not rows:
            embed = embeds.build_simple_embed(
                title=f"🏆 {CURRENCY_NAME} Top 10",
                description="### 尚未開張\n/games blackjack 或 /games dragon_gate 開局就會上榜",
                color=embeds.LEADERBOARD_COLOR,
            )
            await send_expiring_followup(interaction=interaction, embed=embed)
            return

        champion = rows[0]
        board = build_balance_leaderboard_board_image(rows=rows)
        embed = embeds.build_leaderboard_embed(champion=champion)
        await send_expiring_followup(
            interaction=interaction,
            embed=embed,
            file=File(fp=BytesIO(board), filename=BALANCE_LEADERBOARD_BOARD_FILENAME),
        )

    @nextcord.slash_command(
        name="loss_leaderboard",
        description=f"Show today's accumulated {CURRENCY_NAME} casino losses.",
        name_localizations={Locale.zh_TW: "輸錢榜", Locale.ja: "負け額ランキング"},
        description_localizations={
            Locale.zh_TW: f"顯示今日累計輸掉{CURRENCY_NAME}的前 10 名 (每天 0:00 重置)",
            Locale.ja: f"本日累計で失った{CURRENCY_NAME}の上位10名 (毎日 0:00 リセット)。",
        },
        nsfw=False,
    )
    async def loss_leaderboard(self, interaction: Interaction) -> None:
        """Replies with the top 10 gross casino losses for the current day.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        rows = await top_losers(limit=10)
        if not rows:
            embed = embeds.build_simple_embed(
                title=f"💸 今日輸局累計 {CURRENCY_NAME}",
                description="### 今天還沒有人輸錢\n/games blackjack 或 /games dragon_gate 開局就可能進榜",
                color=embeds.LOSS_LEADERBOARD_COLOR,
            )
            await send_expiring_followup(interaction=interaction, embed=embed)
            return

        champion = rows[0]
        board = build_loss_leaderboard_board_image(rows=rows)
        embed = embeds.build_loss_leaderboard_embed(champion=champion)
        await send_expiring_followup(
            interaction=interaction,
            embed=embed,
            file=File(fp=BytesIO(board), filename=LOSS_LEADERBOARD_BOARD_FILENAME),
        )

    @nextcord.slash_command(
        name="give",
        description=f"Transfer your {CURRENCY_NAME} to another member or bot.",
        name_localizations={Locale.zh_TW: "轉帳", Locale.ja: "送金"},
        description_localizations={
            Locale.zh_TW: f"把你的{CURRENCY_NAME}轉給其他成員或 bot",
            Locale.ja: f"他のメンバーまたは bot に{CURRENCY_NAME}を送ります。",
        },
        nsfw=False,
    )
    async def give(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description=f"The member or bot to receive the {CURRENCY_NAME}.",
            name_localizations={Locale.zh_TW: "對象", Locale.ja: "受取人"},
            description_localizations={
                Locale.zh_TW: f"要接收{CURRENCY_NAME}的成員或 bot",
                Locale.ja: f"{CURRENCY_NAME}を受け取るメンバーまたは bot。",
            },
            required=True,
        ),
        amount: str = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to transfer (must be positive). Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要轉的{CURRENCY_NAME} (必須大於 0)，可加逗號",
                Locale.ja: f"送る{CURRENCY_NAME} (1以上)。カンマ可。",
            },
            required=True,
            min_length=1,
        ),
    ) -> None:
        """Transfers points from the caller to `member`.

        Args:
            interaction: The interaction that triggered the command.
            member: The recipient.
            amount: Raw transfer amount text parsed by the bot.
        """
        parsed_amount = _parse_positive_amount(raw_amount=amount)
        if parsed_amount is None:
            await send_ephemeral_response(
                interaction=interaction, embed=embeds.build_invalid_amount_embed(title="轉帳失敗")
            )
            return
        await interaction.response.defer()
        if interaction.user is None:
            return

        sender = interaction.user
        guild = getattr(interaction, "guild", None)
        sender_avatar_url = await guild_avatar_url(user=sender, guild=guild)

        if member.id == sender.id:
            await send_expiring_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="轉帳失敗",
                    description="### 不能轉給自己",
                    author_name=sender.display_name,
                    author_icon_url=sender_avatar_url,
                ),
            )
            return

        receiver_avatar_url = await guild_avatar_url(user=member, guild=guild)
        transfer_result = await transfer(
            sender_id=sender.id,
            sender_name=sender.name,
            sender_avatar_url=sender_avatar_url,
            receiver_id=member.id,
            receiver_name=member.name,
            receiver_avatar_url=receiver_avatar_url,
            amount=parsed_amount,
        )
        if transfer_result is None:
            balance_now = await get_balance(user_id=sender.id)
            await send_expiring_followup(
                interaction=interaction,
                embed=embeds.build_transfer_insufficient_embed(
                    sender_name=sender.display_name,
                    sender_avatar_url=sender_avatar_url,
                    balance_now=balance_now,
                    amount=parsed_amount,
                ),
            )
            return

        embed = embeds.build_transfer_embed(
            amount=parsed_amount,
            sender=embeds.TransferParticipant(
                mention=sender.mention, display_name=sender.display_name
            ),
            sender_avatar_url=sender_avatar_url,
            receiver=embeds.TransferParticipant(
                mention=member.mention, display_name=member.display_name
            ),
            receiver_avatar_url=receiver_avatar_url,
            result=transfer_result,
        )
        await send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="casino",
        description="Show the casino system's cumulative profit and loss.",
        name_localizations={Locale.zh_TW: "賭場", Locale.ja: "カジノ"},
        description_localizations={
            Locale.zh_TW: "顯示賭場系統累積 P&L (跨伺服器)",
            Locale.ja: "カジノシステムの累計 P&L を表示します。",
        },
        nsfw=False,
    )
    async def casino(self, interaction: Interaction) -> None:
        """Shows the casino system's accumulated P&L (was `/house`)."""
        await interaction.response.defer()
        snapshot = await get_casino_ledger()
        embed = embeds.build_casino_embed(snapshot=snapshot)
        await send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="pocat",
        description="Show the bot player's own wallet (shortcut for /balance @bot).",
        name_localizations={Locale.zh_TW: "破貓", Locale.ja: "ポキャット"},
        description_localizations={
            Locale.zh_TW: "顯示機器人玩家自己的錢包 (等同 /balance @bot)",
            Locale.ja: "ボットプレイヤー自身の財布を表示します (/balance @bot のショートカット)。",
        },
        nsfw=False,
    )
    async def pocat(self, interaction: Interaction) -> None:
        """Shows the bot player's `user_wallet` balance and gross flows."""
        await interaction.response.defer()
        if self.bot.user is None:
            await send_expiring_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="❌ 無法查詢", description="目前無法取得機器人身份"
                ),
            )
            return

        bot_user = self.bot.user
        account = await get_account(user_id=bot_user.id)
        name = bot_user.display_name
        if account is None:
            balance, total_earned, total_spent = 0, 0, 0
        else:
            balance = account.balance
            total_earned = account.total_earned
            total_spent = account.total_spent
            name = name or account.name

        embed = embeds.build_pocat_embed(
            name=name,
            avatar_url=bot_user.display_avatar.url,
            balance=balance,
            total_earned=total_earned,
            total_spent=total_spent,
        )
        await send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="credit",
        description="Personal credit operations.",
        name_localizations={Locale.zh_TW: "信貸", Locale.ja: "信用"},
        description_localizations={
            Locale.zh_TW: "個人信貸操作",
            Locale.ja: "personal credit 操作。",
        },
        nsfw=False,
    )
    async def credit(self, interaction: Interaction) -> None:
        """Slash command group for personal credit operations."""

    @credit.subcommand(
        name="borrow",
        description=f"Request a personal {CURRENCY_NAME} loan from another member.",
        name_localizations={Locale.zh_TW: "借款", Locale.ja: "借入"},
        description_localizations={
            Locale.zh_TW: f"向指定成員提出{CURRENCY_NAME}借款申請",
            Locale.ja: f"指定メンバーに{CURRENCY_NAME}借入リクエストを送ります。",
        },
    )
    async def credit_borrow(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="The member you want to borrow from.",
            name_localizations={Locale.zh_TW: "貸方", Locale.ja: "貸し手"},
            description_localizations={
                Locale.zh_TW: "要向誰借款",
                Locale.ja: "借入先のメンバー。",
            },
            required=True,
        ),
        amount: str = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to request. Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要借入的{CURRENCY_NAME}，可加逗號",
                Locale.ja: f"借入する{CURRENCY_NAME}。カンマ可。",
            },
            required=True,
            min_length=1,
        ),
        monthly_rate_percent: float = SlashOption(
            name="monthly_rate_percent",
            description="Monthly simple-interest rate percent.",
            name_localizations={Locale.zh_TW: "月利率", Locale.ja: "月利率"},
            description_localizations={
                Locale.zh_TW: "每月單利百分比",
                Locale.ja: "月次 simple interest rate percent。",
            },
            required=False,
            default=DEFAULT_LOAN_MONTHLY_RATE_BPS / 100,
            min_value=0,
            max_value=100,
        ),
    ) -> None:
        """Creates a personal loan request for the target lender.

        Args:
            interaction: The interaction that triggered the command.
            member: The requested lender.
            amount: Raw borrow amount text parsed by the bot.
            monthly_rate_percent: Monthly simple-interest rate.
        """
        parsed_amount = _parse_positive_amount(raw_amount=amount)
        if parsed_amount is None:
            await send_ephemeral_response(
                interaction=interaction, embed=embeds.build_invalid_amount_embed(title="借款失敗")
            )
            return
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user
        guild = getattr(interaction, "guild", None)
        user_avatar_url = await guild_avatar_url(user=user, guild=guild)
        lender_avatar_url = await guild_avatar_url(user=member, guild=guild)
        if member.bot:
            await send_expiring_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="借款失敗",
                    description="### 不能向 bot 借款",
                    author_name=user.display_name,
                    author_icon_url=user_avatar_url,
                ),
            )
            return
        if member.id == user.id:
            await send_expiring_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="借款失敗",
                    description="### 不能向自己借款",
                    author_name=user.display_name,
                    author_icon_url=user_avatar_url,
                ),
            )
            return

        monthly_rate_bps = monthly_rate_percent_to_bps(monthly_rate_percent=monthly_rate_percent)
        proposal = await create_personal_loan_request(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            lender_id=member.id,
            lender_name=member.name,
            lender_avatar_url=lender_avatar_url,
            amount=parsed_amount,
            monthly_rate_bps=monthly_rate_bps,
        )
        if proposal is None:
            await send_expiring_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="借款失敗",
                    description="### 無法建立借款申請",
                    author_name=user.display_name,
                    author_icon_url=user_avatar_url,
                ),
            )
            return

        embed = embeds.build_credit_request_embed(
            borrower=embeds.LoanParty(
                mention=user.mention, display_name=user.display_name, avatar_url=user_avatar_url
            ),
            lender=embeds.LoanParty(mention=member.mention, avatar_url=lender_avatar_url),
            amount=parsed_amount,
            monthly_rate_bps=monthly_rate_bps,
        )
        await send_loan_request_followup(
            interaction=interaction,
            embed=embed,
            view=CreditLoanDecisionView(
                proposal_id=proposal.proposal_id, lender_id=member.id, creator_id=user.id
            ),
        )

    @credit.subcommand(
        name="repay",
        description=f"Repay a personal {CURRENCY_NAME} loan to a member.",
        name_localizations={Locale.zh_TW: "還款", Locale.ja: "返済"},
        description_localizations={
            Locale.zh_TW: "還款給指定貸方",
            Locale.ja: "指定 lender へ personal loan を返済します。",
        },
    )
    async def credit_repay(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="The lender to repay.",
            name_localizations={Locale.zh_TW: "貸方", Locale.ja: "貸し手"},
            description_localizations={Locale.zh_TW: "要還款給誰", Locale.ja: "返済先の lender。"},
            required=True,
        ),
        amount: str = SlashOption(
            name="amount",
            description=f"Maximum {CURRENCY_NAME} to apply against personal debt. Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要還款的最高{CURRENCY_NAME}，可加逗號",
                Locale.ja: f"返済する{CURRENCY_NAME}の上限。カンマ可。",
            },
            required=True,
            min_length=1,
        ),
    ) -> None:
        """Pays down active personal loans owed to `member`.

        Args:
            interaction: The interaction that triggered the command.
            member: The personal lender.
            amount: Raw repayment amount text parsed by the bot.
        """
        if interaction.user is None:
            return
        parsed_amount = _parse_positive_amount(raw_amount=amount)
        if parsed_amount is None:
            await send_ephemeral_response(
                interaction=interaction, embed=embeds.build_invalid_amount_embed(title="還款失敗")
            )
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(
            user=user, guild=getattr(interaction, "guild", None)
        )

        result = await repay_personal_loans(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            lender_id=member.id,
            amount=parsed_amount,
        )
        if result is None:
            await interaction.response.defer(ephemeral=True)
            await send_private_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="還款失敗",
                    description=f"### 沒有可還給 {member.display_name} 的有效個人借款",
                    author_name=user.display_name,
                    author_icon_url=user_avatar_url,
                    thumbnail_url=user_avatar_url,
                ),
            )
            return

        await interaction.response.defer()
        embed = embeds.build_credit_repay_embed(
            actor_name=user.display_name,
            actor_avatar_url=user_avatar_url,
            lender_display_name=member.display_name,
            result=result,
        )
        await send_expiring_followup(interaction=interaction, embed=embed)

    @credit.subcommand(
        name="call",
        description=f"Forcibly collect a personal {CURRENCY_NAME} loan.",
        name_localizations={Locale.zh_TW: "催收", Locale.ja: "回収"},
        description_localizations={
            Locale.zh_TW: "從借方可用餘額強制回收個人借款",
            Locale.ja: "借り手の利用可能残高から personal loan を回収します。",
        },
    )
    async def credit_call(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="The borrower to collect from.",
            name_localizations={Locale.zh_TW: "借方", Locale.ja: "借り手"},
            description_localizations={
                Locale.zh_TW: "要向誰強制回收",
                Locale.ja: "回収対象の borrower。",
            },
            required=True,
        ),
        amount: str = SlashOption(
            name="amount",
            description="Maximum amount to collect; omit or 0 means all owed. Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: "最多回收多少；留空或 0 代表嘗試全收，可加逗號",
                Locale.ja: "回収上限。空欄または 0 は全額。カンマ可。",
            },
            required=False,
            default="",
        ),
    ) -> None:
        """Forcibly collects a personal loan from a borrower."""
        if interaction.user is None:
            return
        is_valid, collect_amount = _parse_collect_amount(raw_amount=amount)
        if not is_valid:
            await send_ephemeral_response(
                interaction=interaction, embed=embeds.build_invalid_amount_embed(title="催收失敗")
            )
            return
        user = interaction.user
        borrower_avatar_url = await guild_avatar_url(
            user=member, guild=getattr(interaction, "guild", None)
        )
        result = await call_personal_loans(
            lender_id=user.id,
            borrower_id=member.id,
            borrower_name=member.name,
            borrower_avatar_url=borrower_avatar_url,
            amount=collect_amount,
        )
        if result is None:
            await interaction.response.defer(ephemeral=True)
            await send_private_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="催收失敗",
                    description=f"### {member.display_name} 沒有欠你有效個人借款，或目前無可扣餘額",
                    author_name=user.display_name,
                    author_icon_url=user.display_avatar.url,
                ),
            )
            return
        await interaction.response.defer()
        embed = embeds.build_credit_call_embed(
            actor_name=user.display_name,
            actor_avatar_url=user.display_avatar.url,
            borrower_mention=member.mention,
            result=result,
        )
        await send_expiring_followup(interaction=interaction, embed=embed)

    @credit.subcommand(
        name="status",
        description="Show your active personal loan contracts.",
        name_localizations={Locale.zh_TW: "狀態", Locale.ja: "状態"},
        description_localizations={
            Locale.zh_TW: "查看你的有效個人信貸",
            Locale.ja: "active personal loan contracts を表示します。",
        },
    )
    async def credit_status(self, interaction: Interaction) -> None:
        """Shows the caller's active personal credit contracts."""
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        contracts = [
            contract
            for contract in await list_loan_contracts(user_id=interaction.user.id)
            if contract.lender_type == LoanLenderType.USER
        ]
        if not contracts:
            embed = embeds.build_simple_embed(
                title="信貸狀態", description="### 目前沒有有效信貸", color=embeds.BORROW_COLOR
            )
            await send_private_followup(interaction=interaction, embed=embed)
            return
        embed = embeds.build_credit_status_embed(
            contracts=contracts, viewer_id=interaction.user.id
        )
        await send_private_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="central_bank",
        description="Central bank lending operations.",
        name_localizations={Locale.zh_TW: "中央銀行", Locale.ja: "中央銀行"},
        description_localizations={
            Locale.zh_TW: "中央銀行借款操作",
            Locale.ja: "中央銀行 loan 操作。",
        },
        nsfw=False,
    )
    async def central_bank(self, interaction: Interaction) -> None:
        """Slash command group for central bank operations."""

    @central_bank.subcommand(
        name="borrow",
        description=f"Request a central bank {CURRENCY_NAME} loan.",
        name_localizations={Locale.zh_TW: "借款", Locale.ja: "借入"},
        description_localizations={
            Locale.zh_TW: f"向中央銀行提出{CURRENCY_NAME}借款申請",
            Locale.ja: f"中央銀行に{CURRENCY_NAME}借入 request を送ります。",
        },
    )
    async def central_bank_borrow(
        self,
        interaction: Interaction,
        amount: str = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to request. Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要向中央銀行借的{CURRENCY_NAME}，可加逗號",
                Locale.ja: f"中央銀行から借入する{CURRENCY_NAME}。カンマ可。",
            },
            required=True,
            min_length=1,
        ),
        monthly_rate_percent: float = SlashOption(
            name="monthly_rate_percent",
            description="Monthly simple-interest rate percent.",
            name_localizations={Locale.zh_TW: "月利率", Locale.ja: "月利率"},
            description_localizations={
                Locale.zh_TW: "每月單利百分比",
                Locale.ja: "月次 simple interest rate percent。",
            },
            required=False,
            default=DEFAULT_LOAN_MONTHLY_RATE_BPS / 100,
            min_value=0,
            max_value=100,
        ),
    ) -> None:
        """Creates a central bank loan request."""
        parsed_amount = _parse_positive_amount(raw_amount=amount)
        if parsed_amount is None:
            await send_ephemeral_response(
                interaction=interaction,
                embed=embeds.build_invalid_amount_embed(title="央行借款失敗"),
            )
            return
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(
            user=user, guild=getattr(interaction, "guild", None)
        )
        monthly_rate_bps = monthly_rate_percent_to_bps(monthly_rate_percent=monthly_rate_percent)
        proposal = await create_central_bank_loan_request(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            amount=parsed_amount,
            monthly_rate_bps=monthly_rate_bps,
        )
        if proposal is None:
            await send_expiring_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="央行借款失敗", description="### 無法建立央行借款申請"
                ),
            )
            return
        embed = embeds.build_central_bank_request_embed(
            borrower=embeds.LoanParty(
                mention=user.mention, display_name=user.display_name, avatar_url=user_avatar_url
            ),
            amount=parsed_amount,
            monthly_rate_bps=monthly_rate_bps,
        )
        await send_loan_request_followup(
            interaction=interaction,
            embed=embed,
            view=CentralBankLoanDecisionView(
                bot=self.bot,
                proposal_id=proposal.proposal_id,
                creator_id=user.id,
                allow_self_approval=self.economy_config.allow_central_bank_self_approval,
            ),
        )

    @central_bank.subcommand(
        name="repay",
        description="Repay your central bank loan.",
        name_localizations={Locale.zh_TW: "還款", Locale.ja: "返済"},
        description_localizations={
            Locale.zh_TW: "償還自己的央行借款",
            Locale.ja: "自分の central bank loan を返済します。",
        },
    )
    async def central_bank_repay(
        self,
        interaction: Interaction,
        amount: str = SlashOption(
            name="amount",
            description="Maximum amount to repay. Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: "最多還款多少，可加逗號",
                Locale.ja: "返済上限。カンマ可。",
            },
            required=True,
            min_length=1,
        ),
    ) -> None:
        """Repays central-bank debt."""
        if interaction.user is None:
            return
        parsed_amount = _parse_positive_amount(raw_amount=amount)
        if parsed_amount is None:
            await send_ephemeral_response(
                interaction=interaction,
                embed=embeds.build_invalid_amount_embed(title="央行還款失敗"),
            )
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(
            user=user, guild=getattr(interaction, "guild", None)
        )
        result = await repay_central_bank_loans(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            amount=parsed_amount,
        )
        if result is None:
            await interaction.response.defer(ephemeral=True)
            await send_private_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="央行還款失敗", description="### 沒有有效央行借款，或目前無可扣餘額"
                ),
            )
            return
        await interaction.response.defer()
        embed = embeds.build_central_bank_repay_embed(
            actor_name=user.display_name,
            actor_avatar_url=user_avatar_url,
            user_mention=user.mention,
            result=result,
        )
        await send_expiring_followup(interaction=interaction, embed=embed)

    @central_bank.subcommand(
        name="call",
        description="Central banker forced collection from a borrower.",
        name_localizations={Locale.zh_TW: "催收", Locale.ja: "回収"},
        description_localizations={
            Locale.zh_TW: "央行成員從借方可用餘額強制回收",
            Locale.ja: "central banker が borrower から強制回収します。",
        },
    )
    async def central_bank_call(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="The borrower to collect from.",
            name_localizations={Locale.zh_TW: "借方", Locale.ja: "借り手"},
            description_localizations={
                Locale.zh_TW: "要向誰催收",
                Locale.ja: "回収対象 borrower。",
            },
            required=True,
        ),
        amount: str = SlashOption(
            name="amount",
            description="Maximum amount to collect; omit or 0 means all owed. Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: "最多回收多少；留空或 0 代表嘗試全收，可加逗號",
                Locale.ja: "回収上限。空欄または 0 は全額。カンマ可。",
            },
            required=False,
            default="",
        ),
    ) -> None:
        """Central-bank forced collection."""
        if interaction.user is None:
            return
        is_valid, collect_amount = _parse_collect_amount(raw_amount=amount)
        if not is_valid:
            await send_ephemeral_response(
                interaction=interaction,
                embed=embeds.build_invalid_amount_embed(title="央行催收失敗"),
            )
            return
        if not await get_central_banker(user_id=interaction.user.id):
            await interaction.response.defer(ephemeral=True)
            await send_private_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="權限不足", description="### 只有央行成員可以執行央行催收"
                ),
            )
            return
        borrower_avatar_url = await guild_avatar_url(
            user=member, guild=getattr(interaction, "guild", None)
        )
        result = await call_central_bank_loans(
            borrower_id=member.id,
            borrower_name=member.name,
            borrower_avatar_url=borrower_avatar_url,
            amount=collect_amount,
        )
        if result is None:
            await interaction.response.defer(ephemeral=True)
            await send_private_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="央行催收失敗", description="### 目標沒有有效央行借款，或目前無可扣餘額"
                ),
            )
            return
        await interaction.response.defer()
        embed = embeds.build_central_bank_call_embed(
            actor_name=interaction.user.display_name,
            actor_avatar_url=interaction.user.display_avatar.url,
            borrower_mention=member.mention,
            borrower_avatar_url=borrower_avatar_url,
            result=result,
        )
        await send_expiring_followup(interaction=interaction, embed=embed)

    @central_bank.subcommand(
        name="status",
        description="Show central bank lending capacity.",
        name_localizations={Locale.zh_TW: "狀態", Locale.ja: "状態"},
        description_localizations={
            Locale.zh_TW: "查看中央銀行可放貸額度",
            Locale.ja: "中央銀行の lending capacity を表示します。",
        },
    )
    async def central_bank_status(self, interaction: Interaction) -> None:
        """Shows central bank lending capacity."""
        await interaction.response.defer()
        exclude_user_ids = (self.bot.user.id,) if self.bot.user else ()
        status = await get_central_bank_status(exclude_user_ids=exclude_user_ids)
        embed = embeds.build_central_bank_status_embed(status=status)
        await send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="checkin",
        description=f"Daily {CURRENCY_NAME} check-in with a 7-day streak bonus.",
        name_localizations={Locale.zh_TW: "簽到", Locale.ja: "デイリーチェックイン"},
        description_localizations={
            Locale.zh_TW: (
                f"每日簽到領 {currency_text(amount=BASE_CHECKIN_REWARD_AMOUNT, compact=True)}, "
                "連續 7 天加成, VIP 2x"
            ),
            Locale.ja: (
                f"毎日{currency_text(amount=BASE_CHECKIN_REWARD_AMOUNT, compact=True)}, "
                "7日連続でボーナス。"
            ),
        },
        nsfw=False,
    )
    async def checkin_command(self, interaction: Interaction) -> None:
        """Claims today's check-in reward; ephemeral so only the caller sees it.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(
            user=user, guild=getattr(interaction, "guild", None)
        )
        result = await checkin(user_id=user.id, name=user.name, avatar_url=user_avatar_url)
        if result is None:
            await send_private_followup(
                interaction=interaction,
                embed=embeds.build_error_embed(
                    title="今天已經簽到過了",
                    description="### 0:00 (Asia/Taipei) 後再回來簽吧",
                    author_name=user.display_name,
                    author_icon_url=user_avatar_url,
                ),
            )
            return

        embed = embeds.build_checkin_embed(
            actor_name=user.display_name, avatar_url=user_avatar_url, result=result
        )
        await send_private_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="vip",
        description=(
            f"Buy permanent VIP for {currency_text(amount=VIP_PURCHASE_COST, compact=True)}: "
            "2x check-in and 1.2x Blackjack wins."
        ),
        name_localizations={Locale.zh_TW: "購買vip", Locale.ja: "vip購入"},
        description_localizations={
            Locale.zh_TW: "購買永久 VIP：簽到 2x、Blackjack 贏局 1.2x",
            Locale.ja: "永久 VIP を購入: check-in 2x、Blackjack 勝利 1.2x。",
        },
        nsfw=False,
    )
    async def vip_command(self, interaction: Interaction) -> None:
        """Buys the permanent VIP perk for a one-time fixed cost.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(
            user=user, guild=getattr(interaction, "guild", None)
        )
        already_vip = await get_vip(user_id=user.id)
        if already_vip:
            await send_private_followup(
                interaction=interaction,
                embed=embeds.build_vip_already_embed(
                    actor_name=user.display_name, avatar_url=user_avatar_url
                ),
            )
            return

        result = await buy_vip(user_id=user.id, name=user.name, avatar_url=user_avatar_url)
        if result is None:
            balance_now = await get_balance(user_id=user.id)
            await send_private_followup(
                interaction=interaction,
                embed=embeds.build_vip_insufficient_embed(
                    actor_name=user.display_name,
                    avatar_url=user_avatar_url,
                    balance_now=balance_now,
                ),
            )
            return

        embed = embeds.build_vip_success_embed(
            actor_name=user.display_name, avatar_url=user_avatar_url, result=result
        )
        await send_private_followup(interaction=interaction, embed=embed)


def setup(bot: commands.Bot) -> None:
    """Adds the EconomyCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(EconomyCogs(bot), override=True)
