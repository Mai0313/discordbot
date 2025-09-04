"""拍賣系統UI視圖"""

from datetime import datetime

import nextcord
from nextcord import Embed, Interaction, SelectOption
from nextcord.ui import View, Button, Select

from .models import Auction
from .database import AuctionDatabase
from .utils import get_currency_display
from .modals import AuctionBidModal, AuctionCreateModal


class AuctionCurrencySelectionView(View):
    """貨幣類型選擇視圖"""

    def __init__(self):
        super().__init__(timeout=300)

    @nextcord.ui.select(
        placeholder="選擇貨幣類型...",
        options=[
            SelectOption(label="楓幣", value="楓幣", emoji="🍁", description="遊戲內楓幣"),
            SelectOption(label="雪花", value="雪花", emoji="❄️", description="雪花貨幣"),
            SelectOption(label="台幣", value="台幣", emoji="💰", description="台灣新台幣"),
        ],
        min_values=1,
        max_values=1,
    )
    async def currency_select(self, select: Select, interaction: Interaction) -> None:
        selected_currency = select.values[0]
        modal = AuctionCreateModal(currency_type=selected_currency)
        await interaction.response.send_modal(modal)


class AuctionView(View):
    """競標互動視圖"""

    def __init__(self, auction: Auction):
        super().__init__(timeout=None)  # 不設置超時
        self.auction = auction

    @nextcord.ui.button(label="出價", style=nextcord.ButtonStyle.green, emoji="💰")
    async def bid_button(self, button: Button, interaction: Interaction) -> None:
        # 檢查是否在伺服器中執行命令
        if interaction.guild is None:
            await interaction.response.send_message(
                "❌ 拍賣功能只能在伺服器中使用!", ephemeral=True
            )
            return

        # 自動歸屬未歸屬的拍賣
        if self.auction.guild_id == 0 and self.auction.id is not None:
            db = AuctionDatabase()
            if db.claim_auction_to_guild(self.auction.id, interaction.guild.id):
                self.auction.guild_id = interaction.guild.id
                # 重新載入拍賣資訊以獲取最新數據
                updated_auction = db.get_auction(self.auction.id, interaction.guild.id)
                if updated_auction:
                    self.auction = updated_auction

        # 檢查競標是否已結束
        if datetime.now() >= self.auction.end_time:
            await interaction.response.send_message("❌ 此拍賣已結束!", ephemeral=True)
            return

        modal = AuctionBidModal(self.auction)
        await interaction.response.send_modal(modal)

    @nextcord.ui.button(label="查看記錄", style=nextcord.ButtonStyle.gray, emoji="📊")
    async def history_button(self, button: Button, interaction: Interaction) -> None:
        if self.auction.id is None:
            await interaction.response.send_message("❌ 拍賣ID無效!", ephemeral=True)
            return

        # 檢查是否在伺服器中執行命令
        if interaction.guild is None:
            await interaction.response.send_message(
                "❌ 拍賣功能只能在伺服器中使用!", ephemeral=True
            )
            return

        # 自動歸屬未歸屬的拍賣
        if self.auction.guild_id == 0 and self.auction.id is not None:
            db = AuctionDatabase()
            if db.claim_auction_to_guild(self.auction.id, interaction.guild.id):
                self.auction.guild_id = interaction.guild.id
                # 重新載入拍賣資訊以獲取最新數據
                updated_auction = db.get_auction(self.auction.id, interaction.guild.id)
                if updated_auction:
                    self.auction = updated_auction

        if self.auction.id is None:
            await interaction.response.send_message("❌ 拍賣ID無效!", ephemeral=True)
            return

        db = AuctionDatabase()
        bids = db.get_auction_bids(self.auction.id, interaction.guild.id)

        if not bids:
            await interaction.response.send_message("📭 此拍賣還沒有出價記錄。", ephemeral=True)
            return

        embed = Embed(
            title=f"📊 拍賣記錄 - {self.auction.item_name}",
            description=f"拍賣編號：#{self.auction.id}",
            color=0x00AAFF,
        )

        currency = get_currency_display(self.auction.currency_type)
        bid_list = []
        for i, bid in enumerate(bids, 1):
            time_str = bid.timestamp.strftime("%m/%d %H:%M")
            bid_list.append(
                f"{i}. **{bid.bidder_name}** - {bid.amount:,.2f} {currency} ({time_str})"
            )

        embed.add_field(
            name="💰 出價記錄 (前10筆)",
            value="\n".join(bid_list) if bid_list else "暫無記錄",
            inline=False,
        )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @nextcord.ui.button(label="刷新", style=nextcord.ButtonStyle.gray, emoji="🔄")
    async def refresh_button(self, button: Button, interaction: Interaction) -> None:
        if self.auction.id is None:
            await interaction.response.send_message("❌ 拍賣ID無效!", ephemeral=True)
            return

        # 檢查是否在伺服器中執行命令
        if interaction.guild is None:
            await interaction.response.send_message(
                "❌ 拍賣功能只能在伺服器中使用!", ephemeral=True
            )
            return

        db = AuctionDatabase()

        # 自動歸屬未歸屬的拍賣
        if self.auction.guild_id == 0 and db.claim_auction_to_guild(
            self.auction.id, interaction.guild.id
        ):
            self.auction.guild_id = interaction.guild.id

        updated_auction = db.get_auction(self.auction.id, interaction.guild.id)

        if updated_auction:
            self.auction = updated_auction
            embed = self._create_auction_embed(updated_auction)
            view = AuctionView(updated_auction)

            await interaction.response.edit_message(embed=embed, view=view)
        else:
            await interaction.response.send_message("❌ 無法載入拍賣資訊!", ephemeral=True)

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


class AuctionListView(View):
    """拍賣列表視圖"""

    def __init__(self, auctions: list[Auction]):
        super().__init__(timeout=300)
        self.auctions = auctions

        if auctions:
            options = []
            for auction in auctions:
                remaining_time = auction.end_time - datetime.now()
                hours = int(remaining_time.total_seconds() // 3600)
                currency = get_currency_display(auction.currency_type)

                description = f"當前價格: {auction.current_price:,.2f} {currency} | 剩餘: {hours}h"
                options.append(
                    SelectOption(
                        label=auction.item_name, description=description, value=str(auction.id)
                    )
                )

            self.auction_select.options = options
        else:
            self.auction_select.disabled = True

    @nextcord.ui.select(placeholder="選擇要查看的拍賣...", min_values=1, max_values=1)
    async def auction_select(self, select: Select, interaction: Interaction) -> None:
        auction_id = int(select.values[0])

        # 檢查是否在伺服器中執行命令
        if interaction.guild is None:
            await interaction.response.send_message(
                "❌ 拍賣功能只能在伺服器中使用!", ephemeral=True
            )
            return

        db = AuctionDatabase()
        auction = db.get_auction(auction_id, interaction.guild.id)

        if auction:
            embed = self._create_auction_embed(auction)
            view = AuctionView(auction)

            await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        else:
            await interaction.response.send_message("❌ 找不到該拍賣!", ephemeral=True)

    def _create_auction_embed(self, auction: Auction) -> Embed:
        """創建競標 Embed"""
        # 為未認領的拍賣添加特殊標記
        title_prefix = "🔒 " if auction.guild_id == 0 else "🏺 "
        embed = Embed(
            title=f"{title_prefix}{auction.item_name}",
            description=f"拍賣編號：#{auction.id}",
            color=0xFFD700 if auction.guild_id != 0 else 0xFF8C00,
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

        # 為未認領的拍賣添加特殊說明
        footer_text = "點擊下方按鈕參與競標!"
        if auction.guild_id == 0:
            footer_text += " | 此拍賣將在您互動時自動歸屬於本伺服器"

        embed.set_footer(text=footer_text)
        return embed