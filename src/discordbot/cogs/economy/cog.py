"""Slash commands for balances, leaderboards, transfers, loans, VIP, and admin tax."""

from io import BytesIO
from datetime import UTC, datetime

import nextcord
from nextcord import File, Locale, Member, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.economy import (
    LEADERBOARD_SIZE,
    VIP_PURCHASE_COST,
    VIP_WIN_MULTIPLIER_LABEL,
    DEFAULT_LOAN_MONTHLY_RATE_BPS,
    EconomyConfig,
    LoanLenderType,
    monthly_rate_percent_to_bps,
)
from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.cogs.economy.views import (
    CreditLoanDecisionView,
    CentralBankLoanDecisionView,
    is_guild_admin,
)
from discordbot.cogs.economy.boards import (
    LOSS_LEADERBOARD_BOARD_FILENAME,
    BALANCE_LEADERBOARD_BOARD_FILENAME,
    build_loss_leaderboard_board_image,
    build_balance_leaderboard_board_image,
)
from discordbot.cogs.economy.embeds import (
    EmbedParty,
    build_error_embed,
    build_pocat_embed,
    build_casino_embed,
    build_balance_embed,
    build_transfer_embed,
    build_credit_call_embed,
    build_leaderboard_embed,
    build_vip_already_embed,
    build_vip_success_embed,
    build_credit_repay_embed,
    build_credit_status_embed,
    build_credit_request_embed,
    build_invalid_amount_embed,
    build_admin_adjustment_embed,
    build_loss_leaderboard_embed,
    build_vip_insufficient_embed,
    build_central_bank_call_embed,
    build_empty_leaderboard_embed,
    build_central_bank_repay_embed,
    build_central_bank_status_embed,
    build_central_bank_ceiling_embed,
    build_central_bank_request_embed,
    build_transfer_insufficient_embed,
    build_empty_loss_leaderboard_embed,
)
from discordbot.utils.amount_parsing import parse_decimal_amount
from discordbot.services.economy.database import (
    top_n,
    buy_vip,
    get_vip,
    transfer,
    get_admin,
    top_losers,
    get_account,
    get_balance,
    get_portfolio,
    adjust_balance,
    get_casino_ledger,
    get_credit_ceiling,
    call_personal_loans,
    list_loan_contracts,
    repay_personal_loans,
    call_central_bank_loans,
    get_central_bank_status,
    record_guild_participant,
    repay_central_bank_loans,
    create_personal_loan_request,
    create_central_bank_loan_request,
)
from discordbot.utils.interaction_responses import (
    send_private_followup,
    send_expiring_followup,
    send_ephemeral_response,
    send_loan_request_followup,
    send_expiring_followup_after_private_defer,
)
from discordbot.services.economy.presentation import CURRENCY_NAME, currency_text


async def _amount_or_refuse(
    interaction: Interaction[commands.Bot], raw_amount: str, title: str, collect: bool = False
) -> int | None:
    """Parses a money option, answering malformed text with an ephemeral notice titled `title`.

    The amount must be positive, except a collection's, where blank or 0 means everything owed
    and comes back as 0. Returns None once the notice is sent, so the caller must not reply again.
    """
    if collect and not raw_amount.strip():
        return 0
    amount = parse_decimal_amount(raw=raw_amount)
    if amount is not None and (collect or amount > 0):
        return amount
    await send_ephemeral_response(
        interaction=interaction, embed=build_invalid_amount_embed(title=title)
    )
    return None


class EconomyCogs(commands.Cog):
    """Point balance, leaderboard, loan, VIP, and economy-admin commands."""

    def __init__(self, bot: commands.Bot) -> None:
        """Initialises the cog with environment-backed economy settings."""
        self.bot = bot
        self.economy_config = EconomyConfig()

    @nextcord.slash_command(
        name="admin",
        description=f"Economy admins only: run {CURRENCY_NAME} maintenance operations.",
        name_localizations={Locale.zh_TW: "管理員", Locale.ja: "管理者"},
        description_localizations={
            Locale.zh_TW: f"economy admin 限定：執行{CURRENCY_NAME}維護操作",
            Locale.ja: f"economy admin 専用：{CURRENCY_NAME}メンテナンス操作を実行します。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def admin(self, interaction: Interaction[commands.Bot]) -> None:
        """Slash command group for economy admin operations."""

    @admin.subcommand(
        name="refund_tax",
        description=f"Economy admins only: credit {CURRENCY_NAME} to a member or bot.",
        name_localizations={Locale.zh_TW: "退稅", Locale.ja: "税還付"},
        description_localizations={
            Locale.zh_TW: f"economy admin 限定：無條件增加某位成員或 bot 的{CURRENCY_NAME}",
            Locale.ja: f"economy admin 専用：メンバーまたは bot に{CURRENCY_NAME}を付与します。",
        },
    )
    async def admin_refund_tax(
        self,
        interaction: Interaction[commands.Bot],
        member: Member = SlashOption(
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
        """Credits points to a member through a manual balance adjustment."""
        parsed_amount = await _amount_or_refuse(
            interaction=interaction, raw_amount=amount, title="退稅失敗"
        )
        if parsed_amount is None:
            return
        await self._run_admin_adjustment(
            interaction=interaction, member=member, title="退稅完成", delta=parsed_amount
        )

    @admin.subcommand(
        name="collect_tax",
        description=f"Economy admins only: debit {CURRENCY_NAME} from a member or bot.",
        name_localizations={Locale.zh_TW: "收稅", Locale.ja: "徴税"},
        description_localizations={
            Locale.zh_TW: f"economy admin 限定：無條件扣除某位成員或 bot 的{CURRENCY_NAME}",
            Locale.ja: f"economy admin 専用：メンバーまたは bot から{CURRENCY_NAME}を徴収します。",
        },
    )
    async def admin_collect_tax(
        self,
        interaction: Interaction[commands.Bot],
        member: Member = SlashOption(
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
        """Debits points from a member through a manual balance adjustment."""
        parsed_amount = await _amount_or_refuse(
            interaction=interaction, raw_amount=amount, title="收稅失敗"
        )
        if parsed_amount is None:
            return
        await self._run_admin_adjustment(
            interaction=interaction, member=member, title="收稅完成", delta=-parsed_amount
        )

    async def _run_admin_adjustment(
        self, interaction: Interaction[commands.Bot], member: Member, title: str, delta: int
    ) -> None:
        """Runs a gated admin balance adjustment and publishes successful results."""
        if interaction.user is None:
            return
        actor = interaction.user
        guild = interaction.guild
        actor_avatar_url = await guild_avatar_url(user=actor, guild=guild)
        if not await get_admin(user_id=actor.id):
            await interaction.response.defer(ephemeral=True)
            await send_private_followup(
                interaction=interaction,
                embed=build_error_embed(
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
        embed = build_admin_adjustment_embed(
            title=title,
            member_mention=member.mention,
            actor_name=actor.display_name,
            actor_avatar_url=actor_avatar_url,
            member_avatar_url=member_avatar_url,
            requested_delta=delta,
            result=result,
            is_collect_clamped=result.applied_delta != delta,
        )
        await send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="balance",
        description=f"Check a member's {CURRENCY_NAME} balance, loans, and VIP status.",
        name_localizations={Locale.zh_TW: "餘額", Locale.ja: "残高"},
        description_localizations={
            Locale.zh_TW: f"查詢成員的{CURRENCY_NAME}餘額、借貸與 VIP 狀態",
            Locale.ja: f"member の{CURRENCY_NAME}残高、loan、VIP 状態を確認します。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def balance(
        self,
        interaction: Interaction[commands.Bot],
        member: Member | None = SlashOption(
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
        """Replies with a member's balance, loans, and VIP status."""
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        target = member or interaction.user
        target_avatar_url = await guild_avatar_url(user=target, guild=interaction.guild)
        portfolio = await get_portfolio(user_id=target.id)
        is_vip = await get_vip(user_id=target.id)
        age_days = (datetime.now(tz=UTC) - target.created_at).days
        embed = build_balance_embed(
            display_name=target.display_name,
            avatar_url=target_avatar_url,
            portfolio=portfolio,
            is_vip=is_vip,
            age_days=age_days,
        )
        await send_private_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="leaderboard",
        description=f"Show the global top {CURRENCY_NAME} holders.",
        name_localizations={Locale.zh_TW: "排行榜", Locale.ja: "リーダーボード"},
        description_localizations={
            Locale.zh_TW: f"顯示全域 {CURRENCY_NAME}前 {LEADERBOARD_SIZE} 名",
            Locale.ja: f"グローバル{CURRENCY_NAME}トップ{LEADERBOARD_SIZE}を表示します。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def leaderboard(self, interaction: Interaction[commands.Bot]) -> None:
        """Replies with the top point holders."""
        await interaction.response.defer()
        rows = await top_n(limit=LEADERBOARD_SIZE)
        if not rows:
            await send_expiring_followup(
                interaction=interaction, embed=build_empty_leaderboard_embed()
            )
            return

        champion = rows[0]
        board = build_balance_leaderboard_board_image(rows=rows)
        embed = build_leaderboard_embed(champion=champion)
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
            Locale.zh_TW: f"顯示今日累計輸掉{CURRENCY_NAME}的前 {LEADERBOARD_SIZE} 名 (每天 0:00 重置)",
            Locale.ja: f"本日累計で失った{CURRENCY_NAME}の上位{LEADERBOARD_SIZE}名 (毎日 0:00 リセット)。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def loss_leaderboard(self, interaction: Interaction[commands.Bot]) -> None:
        """Replies with the top gross casino losses for the current day."""
        await interaction.response.defer()
        rows = await top_losers(limit=LEADERBOARD_SIZE)
        if not rows:
            await send_expiring_followup(
                interaction=interaction, embed=build_empty_loss_leaderboard_embed()
            )
            return

        champion = rows[0]
        board = build_loss_leaderboard_board_image(rows=rows)
        embed = build_loss_leaderboard_embed(champion=champion)
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
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def give(
        self,
        interaction: Interaction[commands.Bot],
        member: Member = SlashOption(
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
        """Transfers points from the caller to `member`."""
        parsed_amount = await _amount_or_refuse(
            interaction=interaction, raw_amount=amount, title="轉帳失敗"
        )
        if parsed_amount is None:
            return
        await interaction.response.defer()
        if interaction.user is None:
            return

        sender = interaction.user
        guild = interaction.guild
        sender_avatar_url = await guild_avatar_url(user=sender, guild=guild)

        if member.id == sender.id:
            await send_expiring_followup(
                interaction=interaction,
                embed=build_error_embed(
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
                embed=build_transfer_insufficient_embed(
                    sender_name=sender.display_name,
                    sender_avatar_url=sender_avatar_url,
                    balance_now=balance_now,
                    amount=parsed_amount,
                ),
            )
            return

        embed = build_transfer_embed(
            amount=parsed_amount,
            sender=EmbedParty(
                mention=sender.mention,
                display_name=sender.display_name,
                avatar_url=sender_avatar_url,
            ),
            receiver=EmbedParty(
                mention=member.mention,
                display_name=member.display_name,
                avatar_url=receiver_avatar_url,
            ),
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
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def casino(self, interaction: Interaction[commands.Bot]) -> None:
        """Shows the casino system's accumulated P&L."""
        await interaction.response.defer()
        snapshot = await get_casino_ledger()
        embed = build_casino_embed(snapshot=snapshot)
        await send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="pocat",
        description="Show the bot player's own wallet: balance plus total earned and spent.",
        name_localizations={Locale.zh_TW: "破貓", Locale.ja: "ポキャット"},
        description_localizations={
            Locale.zh_TW: "顯示機器人玩家自己的錢包：餘額與累計流水",
            Locale.ja: "ボットプレイヤー自身の財布 (残高と累計の収支) を表示します。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def pocat(self, interaction: Interaction[commands.Bot]) -> None:
        """Shows the bot player's `user_wallet` balance and gross flows."""
        await interaction.response.defer()
        if self.bot.user is None:
            await send_expiring_followup(
                interaction=interaction,
                embed=build_error_embed(title="❌ 無法查詢", description="目前無法取得機器人身份"),
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

        embed = build_pocat_embed(
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
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def credit(self, interaction: Interaction[commands.Bot]) -> None:
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
        interaction: Interaction[commands.Bot],
        member: Member = SlashOption(
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
        """Creates a personal loan request for the target lender."""
        parsed_amount = await _amount_or_refuse(
            interaction=interaction, raw_amount=amount, title="借款失敗"
        )
        if parsed_amount is None:
            return
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user
        guild = interaction.guild
        user_avatar_url = await guild_avatar_url(user=user, guild=guild)
        lender_avatar_url = await guild_avatar_url(user=member, guild=guild)
        if member.bot:
            await send_expiring_followup(
                interaction=interaction,
                embed=build_error_embed(
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
                embed=build_error_embed(
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
        embed = build_credit_request_embed(
            borrower=EmbedParty(
                mention=user.mention, display_name=user.display_name, avatar_url=user_avatar_url
            ),
            lender=EmbedParty(
                mention=member.mention,
                display_name=member.display_name,
                avatar_url=lender_avatar_url,
            ),
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
        interaction: Interaction[commands.Bot],
        member: Member = SlashOption(
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
        """Pays down active personal loans owed to `member`."""
        if interaction.user is None:
            return
        parsed_amount = await _amount_or_refuse(
            interaction=interaction, raw_amount=amount, title="還款失敗"
        )
        if parsed_amount is None:
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(user=user, guild=interaction.guild)

        # Deferred before the write, not after it: the token dies three seconds after dispatch
        # while a SQLite writer waits out lock contention for longer, so acking on the result
        # lets a committed settlement reach its owner as "the application did not respond".
        # Ephemeral because the failure below is private and this ack is what the caller sees
        # while it runs.
        await interaction.response.defer(ephemeral=True)
        result = await repay_personal_loans(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            lender_id=member.id,
            amount=parsed_amount,
        )
        if result is None:
            await send_private_followup(
                interaction=interaction,
                embed=build_error_embed(
                    title="還款失敗",
                    description=f"### 沒有可還給 {member.display_name} 的有效個人借款",
                    author_name=user.display_name,
                    author_icon_url=user_avatar_url,
                    thumbnail_url=user_avatar_url,
                ),
            )
            return

        embed = build_credit_repay_embed(
            actor_name=user.display_name,
            actor_avatar_url=user_avatar_url,
            lender_display_name=member.display_name,
            result=result,
        )
        await send_expiring_followup_after_private_defer(interaction=interaction, embed=embed)

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
        interaction: Interaction[commands.Bot],
        member: Member = SlashOption(
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
        collect_amount = await _amount_or_refuse(
            interaction=interaction, raw_amount=amount, title="催收失敗", collect=True
        )
        if collect_amount is None:
            return
        user = interaction.user
        guild = interaction.guild
        borrower_avatar_url = await guild_avatar_url(user=member, guild=guild)
        actor_avatar_url = await guild_avatar_url(user=user, guild=guild)
        # Acked before the write, and ephemerally — see `credit_repay` for why both halves
        # matter.
        await interaction.response.defer(ephemeral=True)
        result = await call_personal_loans(
            lender_id=user.id,
            borrower_id=member.id,
            borrower_name=member.name,
            borrower_avatar_url=borrower_avatar_url,
            amount=collect_amount or None,
        )
        if result is None:
            await send_private_followup(
                interaction=interaction,
                embed=build_error_embed(
                    title="催收失敗",
                    description=f"### {member.display_name} 沒有欠你有效個人借款，或目前無可扣餘額",
                    author_name=user.display_name,
                    author_icon_url=actor_avatar_url,
                ),
            )
            return
        embed = build_credit_call_embed(
            actor_name=user.display_name,
            actor_avatar_url=actor_avatar_url,
            borrower_mention=member.mention,
            result=result,
        )
        await send_expiring_followup_after_private_defer(interaction=interaction, embed=embed)

    @credit.subcommand(
        name="status",
        description="Show your active personal loan contracts.",
        name_localizations={Locale.zh_TW: "狀態", Locale.ja: "状態"},
        description_localizations={
            Locale.zh_TW: "查看你的有效個人信貸",
            Locale.ja: "active personal loan contracts を表示します。",
        },
    )
    async def credit_status(self, interaction: Interaction[commands.Bot]) -> None:
        """Shows the active personal credit contracts the caller borrowed or lent on."""
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        contracts = [
            contract
            for contract in await list_loan_contracts(user_id=interaction.user.id)
            if contract.lender_type == LoanLenderType.USER
        ]
        embed = build_credit_status_embed(contracts=contracts, viewer_id=interaction.user.id)
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
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def central_bank(self, interaction: Interaction[commands.Bot]) -> None:
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
        interaction: Interaction[commands.Bot],
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
        parsed_amount = await _amount_or_refuse(
            interaction=interaction, raw_amount=amount, title="央行借款失敗"
        )
        if parsed_amount is None:
            return
        # Refused before the proposal exists rather than at the button: outside a guild
        # nobody holds `administrator`, so an approve and a reject would both be
        # impossible and the request would sit until it timed out with no explanation.
        if interaction.guild_id is None:
            await send_ephemeral_response(
                interaction=interaction,
                embed=build_error_embed(
                    title="央行借款失敗", description="### 央行借款只能在伺服器裡提出"
                ),
            )
            return
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user
        await record_guild_participant(guild_id=interaction.guild_id, user_id=user.id)
        user_avatar_url = await guild_avatar_url(user=user, guild=interaction.guild)
        monthly_rate_bps = monthly_rate_percent_to_bps(monthly_rate_percent=monthly_rate_percent)
        ceiling = await get_credit_ceiling(user_id=user.id)
        if ceiling < parsed_amount:
            await send_expiring_followup(
                interaction=interaction,
                embed=build_central_bank_ceiling_embed(
                    borrower_mention=user.mention, requested=parsed_amount, ceiling=ceiling
                ),
            )
            return
        proposal = await create_central_bank_loan_request(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            amount=parsed_amount,
            monthly_rate_bps=monthly_rate_bps,
        )
        embed = build_central_bank_request_embed(
            borrower=EmbedParty(
                mention=user.mention, display_name=user.display_name, avatar_url=user_avatar_url
            ),
            amount=parsed_amount,
            monthly_rate_bps=monthly_rate_bps,
        )
        await send_loan_request_followup(
            interaction=interaction,
            embed=embed,
            view=CentralBankLoanDecisionView(
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
        interaction: Interaction[commands.Bot],
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
        parsed_amount = await _amount_or_refuse(
            interaction=interaction, raw_amount=amount, title="央行還款失敗"
        )
        if parsed_amount is None:
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(user=user, guild=interaction.guild)
        # Acked before the write, and ephemerally — see `credit_repay` for why both halves
        # matter.
        await interaction.response.defer(ephemeral=True)
        if interaction.guild_id is not None:
            await record_guild_participant(guild_id=interaction.guild_id, user_id=user.id)
        result = await repay_central_bank_loans(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            amount=parsed_amount,
        )
        if result is None:
            await send_private_followup(
                interaction=interaction,
                embed=build_error_embed(
                    title="央行還款失敗", description="### 沒有有效央行借款，或目前無可扣餘額"
                ),
            )
            return
        embed = build_central_bank_repay_embed(
            actor_name=user.display_name,
            actor_avatar_url=user_avatar_url,
            user_mention=user.mention,
            result=result,
        )
        await send_expiring_followup_after_private_defer(interaction=interaction, embed=embed)

    @central_bank.subcommand(
        name="call",
        description="server admin only: forced collection from a borrower in this server.",
        name_localizations={Locale.zh_TW: "催收", Locale.ja: "回収"},
        description_localizations={
            Locale.zh_TW: "server admin 限定：向本伺服器的借方強制回收",
            Locale.ja: "server admin 専用：このサーバーの borrower から強制回収します。",
        },
    )
    async def central_bank_call(
        self,
        interaction: Interaction[commands.Bot],
        member: Member = SlashOption(
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
        collect_amount = await _amount_or_refuse(
            interaction=interaction, raw_amount=amount, title="央行催收失敗", collect=True
        )
        if collect_amount is None:
            return
        if interaction.guild_id is None or not is_guild_admin(interaction=interaction):
            await interaction.response.defer(ephemeral=True)
            await send_private_followup(
                interaction=interaction,
                embed=build_error_embed(
                    title="權限不足", description="### 只有這個伺服器的管理員可以執行央行催收"
                ),
            )
            return
        guild = interaction.guild
        borrower_avatar_url = await guild_avatar_url(user=member, guild=guild)
        actor_avatar_url = await guild_avatar_url(user=interaction.user, guild=guild)
        # Acked before the write, and ephemerally — see `credit_repay` for why both halves
        # matter.
        await interaction.response.defer(ephemeral=True)
        await record_guild_participant(guild_id=interaction.guild_id, user_id=interaction.user.id)
        result = await call_central_bank_loans(
            guild_id=interaction.guild_id,
            borrower_id=member.id,
            borrower_name=member.name,
            borrower_avatar_url=borrower_avatar_url,
            amount=collect_amount or None,
        )
        if result is None:
            await send_private_followup(
                interaction=interaction,
                embed=build_error_embed(
                    title="央行催收失敗",
                    description="### 目標不是本伺服器的參與者、沒有有效央行借款，或目前無可扣餘額",
                ),
            )
            return
        embed = build_central_bank_call_embed(
            actor_name=interaction.user.display_name,
            actor_avatar_url=actor_avatar_url,
            borrower_mention=member.mention,
            borrower_avatar_url=borrower_avatar_url,
            result=result,
        )
        await send_expiring_followup_after_private_defer(interaction=interaction, embed=embed)

    @central_bank.subcommand(
        name="status",
        description="Show central bank lending capacity.",
        name_localizations={Locale.zh_TW: "狀態", Locale.ja: "状態"},
        description_localizations={
            Locale.zh_TW: "查看中央銀行可放貸額度",
            Locale.ja: "中央銀行の lending capacity を表示します。",
        },
    )
    async def central_bank_status(self, interaction: Interaction[commands.Bot]) -> None:
        """Shows this server's central bank lending capacity."""
        if interaction.guild_id is None:
            await send_ephemeral_response(
                interaction=interaction,
                embed=build_error_embed(
                    title="央行狀態", description="### 央行額度是每個伺服器各自計算的"
                ),
            )
            return
        await interaction.response.defer()
        if interaction.user is not None:
            await record_guild_participant(
                guild_id=interaction.guild_id, user_id=interaction.user.id
            )
        status = await get_central_bank_status(guild_id=interaction.guild_id)
        embed = build_central_bank_status_embed(status=status)
        await send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="vip",
        description=(
            f"Buy permanent VIP for {currency_text(amount=VIP_PURCHASE_COST, compact=True)}: "
            f"{VIP_WIN_MULTIPLIER_LABEL} Blackjack wins."
        ),
        name_localizations={Locale.zh_TW: "購買vip", Locale.ja: "vip購入"},
        description_localizations={
            Locale.zh_TW: f"購買永久 VIP：Blackjack 贏局 {VIP_WIN_MULTIPLIER_LABEL}",
            Locale.ja: f"永久 VIP を購入: Blackjack 勝利 {VIP_WIN_MULTIPLIER_LABEL}。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def vip_command(self, interaction: Interaction[commands.Bot]) -> None:
        """Buys the permanent VIP perk for a one-time fixed cost."""
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(user=user, guild=interaction.guild)
        already_vip = await get_vip(user_id=user.id)
        if already_vip:
            await send_private_followup(
                interaction=interaction,
                embed=build_vip_already_embed(
                    actor_name=user.display_name, avatar_url=user_avatar_url
                ),
            )
            return

        result = await buy_vip(user_id=user.id, name=user.name, avatar_url=user_avatar_url)
        if result is None:
            balance_now = await get_balance(user_id=user.id)
            await send_private_followup(
                interaction=interaction,
                embed=build_vip_insufficient_embed(
                    actor_name=user.display_name,
                    avatar_url=user_avatar_url,
                    balance_now=balance_now,
                ),
            )
            return

        embed = build_vip_success_embed(
            actor_name=user.display_name, avatar_url=user_avatar_url, result=result
        )
        await send_private_followup(interaction=interaction, embed=embed)


def setup(bot: commands.Bot) -> None:
    """Adds the EconomyCogs to the bot."""
    bot.add_cog(EconomyCogs(bot), override=True)
