from datetime import datetime

import nextcord
from nextcord import Embed, Locale, Interaction
from nextcord.ext import commands

from .models import Auction
from .utils import get_currency_display
from .database import AuctionDatabase
from .views import AuctionCurrencySelectionView, AuctionView


class AuctionCogs(commands.Cog):
    """拍賣系統功能"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # 初始化競標資料庫
        self.auction_db = AuctionDatabase()

    @nextcord.slash_command(
        name="auction_create",
        description="Create a new item auction",
        name_localizations={Locale.zh_TW: "創建拍賣", Locale.ja: "オークション作成"},
        description_localizations={
            Locale.zh_TW: "創建新的物品拍賣",
            Locale.ja: "新しいアイテムオークションを作成",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def auction_create(self, interaction: Interaction) -> None:
        """創建新拍賣"""
        # 檢查是否在伺服器中執行命令
        if interaction.guild is None:
            embed = Embed(
                title="❌ 錯誤",
                description="拍賣功能只能在伺服器中使用，不支援私人訊息!",
                color=0xFF0000,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        embed = Embed(
            title="🏺 創建拍賣", description="請先選擇拍賣使用的貨幣類型：", color=0xFFD700
        )
        view = AuctionCurrencySelectionView()
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    @nextcord.slash_command(
        name="auction_list",
        description="View active auctions",
        name_localizations={Locale.zh_TW: "拍賣列表", Locale.ja: "オークションリスト"},
        description_localizations={
            Locale.zh_TW: "查看進行中的拍賣列表",
            Locale.ja: "進行中のオークション一覧を表示",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def auction_list(self, interaction: Interaction) -> None:
        """查看拍賣列表"""
        await interaction.response.defer()

        # 檢查是否在伺服器中執行命令
        if interaction.guild is None:
            embed = Embed(
                title="❌ 錯誤",
                description="拍賣功能只能在伺服器中使用，不支援私人訊息!",
                color=0xFF0000,
            )
            await interaction.followup.send(embed=embed)
            return

        auctions = self.auction_db.get_active_auctions(interaction.guild.id)

        if not auctions:
            embed = Embed(
                title="📋 拍賣列表", description="目前沒有進行中的拍賣。", color=0xFFAA00
            )
            embed.add_field(
                name="💡 提示", value="使用 `/auction_create` 來創建新的拍賣!", inline=False
            )
            await interaction.followup.send(embed=embed)
            return

        embed = Embed(
            title="📋 進行中的拍賣",
            description=f"共有 {len(auctions)} 個拍賣進行中",
            color=0x00AAFF,
        )

        # 顯示前5個拍賣的摘要
        auction_summary = []
        for i, auction in enumerate(auctions, 1):
            remaining_time = auction.end_time - datetime.now()
            hours = int(remaining_time.total_seconds() // 3600)
            currency = get_currency_display(auction.currency_type)

            summary = (
                f"{i}. **{auction.item_name}** (#{auction.id})\n"
                f"   💰 {auction.current_price:,.2f} {currency} | ⏰ {hours}h 剩餘"
            )
            auction_summary.append(summary)

        embed.add_field(name="🏺 拍賣預覽", value="\n\n".join(auction_summary), inline=False)

        if len(auctions) > 5:
            embed.add_field(name="📝 說明", value="請使用下方選單查看詳細資訊。", inline=False)

        from .views import AuctionListView
        view = AuctionListView(auctions)
        await interaction.followup.send(embed=embed, view=view)

    @nextcord.slash_command(
        name="auction_info",
        description="View detailed information about a specific auction",
        name_localizations={Locale.zh_TW: "拍賣資訊", Locale.ja: "オークション情報"},
        description_localizations={
            Locale.zh_TW: "查看特定拍賣的詳細資訊",
            Locale.ja: "特定のオークションの詳細情報を表示",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def auction_info(
        self,
        interaction: Interaction,
        auction_id: int = nextcord.SlashOption(
            name="auction_id",
            description="Auction ID to view",
            name_localizations={Locale.zh_TW: "拍賣編號", Locale.ja: "オークション番号"},
            description_localizations={
                Locale.zh_TW: "要查看的拍賣編號",
                Locale.ja: "表示するオークション番号",
            },
            required=True,
        ),
    ) -> None:
        """查看特定拍賣資訊"""
        await interaction.response.defer()

        # 檢查是否在伺服器中執行命令
        if interaction.guild is None:
            embed = Embed(
                title="❌ 錯誤",
                description="拍賣功能只能在伺服器中使用，不支援私人訊息!",
                color=0xFF0000,
            )
            await interaction.followup.send(embed=embed)
            return

        auction = self.auction_db.get_auction(auction_id, interaction.guild.id)

        if not auction:
            embed = Embed(
                title="❌ 錯誤", description=f"找不到編號 #{auction_id} 的拍賣。", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        embed = self._create_auction_embed(auction)
        view = AuctionView(auction)

        await interaction.followup.send(embed=embed, view=view)

    @nextcord.slash_command(
        name="auction_my",
        description="View your auctions and bids",
        name_localizations={Locale.zh_TW: "我的拍賣", Locale.ja: "マイオークション"},
        description_localizations={
            Locale.zh_TW: "查看你的拍賣和出價記錄",
            Locale.ja: "あなたのオークションと入札記録を表示",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def auction_my(self, interaction: Interaction) -> None:
        """查看個人拍賣記錄"""
        await interaction.response.defer()

        # 檢查是否在伺服器中執行命令
        if interaction.guild is None:
            embed = Embed(
                title="❌ 錯誤",
                description="拍賣功能只能在伺服器中使用，不支援私人訊息!",
                color=0xFF0000,
            )
            await interaction.followup.send(embed=embed)
            return

        active_auctions = self.auction_db.get_active_auctions(interaction.guild.id)
        user_auctions = self._get_user_created_auctions(active_auctions, interaction.user.id)
        leading_auctions = self._get_user_leading_auctions(active_auctions, interaction.user.id)

        embed = Embed(title=f"📋 {interaction.user.mention} 的拍賣記錄", color=0x9966FF)

        self._add_auction_fields_to_embed(embed, user_auctions, leading_auctions)
        await interaction.followup.send(embed=embed)

    def _get_user_created_auctions(self, auctions: list[Auction], user_id: int) -> list[Auction]:
        """取得用戶創建的拍賣"""
        return [auction for auction in auctions if auction.creator_id == user_id]

    def _get_user_leading_auctions(self, auctions: list[Auction], user_id: int) -> list[Auction]:
        """取得用戶領先的拍賣"""
        return [auction for auction in auctions if auction.current_bidder_id == user_id]

    def _add_auction_fields_to_embed(
        self, embed: Embed, user_auctions: list[Auction], leading_auctions: list[Auction]
    ) -> None:
        """將拍賣資訊添加到 embed"""
        if user_auctions:
            auction_list = self._format_auction_list(user_auctions)
            embed.add_field(name="🏺 我創建的拍賣", value="\n".join(auction_list), inline=False)

        if leading_auctions:
            leading_list = self._format_auction_list(leading_auctions)
            embed.add_field(name="👑 我領先的拍賣", value="\n".join(leading_list), inline=False)

        if not user_auctions and not leading_auctions:
            embed.description = "你還沒有創建或參與任何拍賣。"
            embed.add_field(
                name="💡 開始使用",
                value="使用 `/auction_create` 創建拍賣\n使用 `/auction_list` 查看並參與拍賣",
                inline=False,
            )

    def _format_auction_list(self, auctions: list[Auction]) -> list[str]:
        """格式化拍賣清單"""
        auction_list = []
        for auction in auctions:
            remaining_time = auction.end_time - datetime.now()
            hours = int(remaining_time.total_seconds() // 3600)
            currency = get_currency_display(auction.currency_type)
            auction_list.append(
                f"#{auction.id} **{auction.item_name}** - {auction.current_price:,.2f} {currency} ({hours}h)"
            )
        return auction_list

    def _create_auction_embed(self, auction: Auction) -> Embed:
        """創建競標 Embed"""
        embed = Embed(
            title=f"🏺 {auction.item_name}", description=f"拍賣編號：#{auction.id}", color=0xFFD700
        )

        currency = get_currency_display(auction.currency_type)
        embed.add_field(
            name="💰 當前價格", value=f"{auction.current_price:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="📈 加價金額", value=f"{auction.increment:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="👤 當前領先", value=auction.current_bidder_name or "暫無出價", inline=True
        )

        embed.add_field(name="🏁 拍賣發起人", value=auction.creator_name, inline=True)

        remaining_time = auction.end_time - datetime.now()
        if remaining_time.total_seconds() > 0:
            hours, remainder = divmod(int(remaining_time.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)

            embed.add_field(name="⏰ 剩餘時間", value=f"{hours}時{minutes}分", inline=True)
        else:
            embed.add_field(name="⏰ 狀態", value="已結束", inline=True)

        embed.add_field(
            name="📅 結束時間", value=auction.end_time.strftime("%m/%d %H:%M"), inline=True
        )

        embed.set_footer(text="點擊下方按鈕參與競標!")
        return embed