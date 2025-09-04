"""拍賣系統模態對話框"""

from datetime import datetime, timedelta

import logfire
import nextcord
from nextcord import Embed, Interaction
from nextcord.ui import Modal, TextInput

from .models import Auction
from .database import AuctionDatabase
from .utils import get_currency_display


class AuctionCreateModal(Modal):
    """創建競標的模態對話框"""

    def __init__(self, currency_type: str = "楓幣"):
        super().__init__(title="創建拍賣", timeout=300)
        self.selected_currency = currency_type

        self.item_name = TextInput(
            label="物品名稱",
            placeholder="請輸入要拍賣的物品名稱...",
            required=True,
            max_length=100,
        )

        currency_display = get_currency_display(currency_type)
        self.starting_price = TextInput(
            label="起標價格",
            placeholder=f"請輸入起標價格 ({currency_display})，支援小數點...",
            required=True,
            max_length=20,
        )

        self.increment = TextInput(
            label="加價金額",
            placeholder=f"請輸入每次最少加價金額 ({currency_display})，支援小數點...",
            required=True,
            max_length=20,
        )

        self.duration = TextInput(
            label="拍賣時長 (小時)",
            placeholder="請輸入拍賣持續時間 (1-168小時)...",
            required=True,
            max_length=3,
            default_value="24",
        )

        self.add_item(self.item_name)
        self.add_item(self.starting_price)
        self.add_item(self.increment)
        self.add_item(self.duration)

    async def callback(self, interaction: Interaction) -> None:
        try:
            starting_price = float(self.starting_price.value)
            increment = float(self.increment.value)
            duration_hours = int(self.duration.value)
            currency_type = self.selected_currency

            if starting_price <= 0:
                await interaction.response.send_message("❌ 起標價格必須大於 0!", ephemeral=True)
                return

            if increment <= 0:
                await interaction.response.send_message("❌ 加價金額必須大於 0!", ephemeral=True)
                return

            if not (1 <= duration_hours <= 168):
                await interaction.response.send_message(
                    "❌ 拍賣時長必須在 1-168 小時之間!", ephemeral=True
                )
                return

            # 檢查是否在伺服器中執行命令
            if interaction.guild is None:
                await interaction.response.send_message(
                    "❌ 拍賣功能只能在伺服器中使用，不支援私人訊息!", ephemeral=True
                )
                return

            # 創建競標
            auction = Auction(
                guild_id=interaction.guild.id,
                item_name=self.item_name.value,
                starting_price=starting_price,
                increment=increment,
                duration_hours=duration_hours,
                creator_id=interaction.user.id,
                creator_name=interaction.user.display_name,
                end_time=datetime.now() + timedelta(hours=duration_hours),
                current_price=starting_price,
                currency_type=currency_type,
            )

            db = AuctionDatabase()
            auction_id = db.create_auction(auction)
            auction.id = auction_id

            # 創建競標顯示
            embed = self._create_auction_embed(auction)
            from .views import AuctionView
            view = AuctionView(auction)

            await interaction.response.send_message(
                f"🎉 拍賣已成功創建!拍賣編號：#{auction_id}", embed=embed, view=view
            )

        except ValueError:
            await interaction.response.send_message("❌ 請輸入有效的數字格式!", ephemeral=True)
        except Exception as e:
            logfire.error(f"創建拍賣時發生錯誤: {e}")
            await interaction.response.send_message(
                "❌ 創建拍賣時發生錯誤，請稍後再試!", ephemeral=True
            )

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
        hours, remainder = divmod(int(remaining_time.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)

        embed.add_field(name="⏰ 剩餘時間", value=f"{hours}時{minutes}分", inline=True)

        embed.add_field(
            name="📅 結束時間", value=auction.end_time.strftime("%m/%d %H:%M"), inline=True
        )

        # 為未認領的拍賣添加特殊說明
        footer_text = "點擊下方按鈕參與競標!"
        if auction.guild_id == 0:
            footer_text += " | 此拍賣將在您互動時自動歸屬於本伺服器"

        embed.set_footer(text=footer_text)
        return embed


class AuctionBidModal(Modal):
    """出價模態對話框"""

    def __init__(self, auction: Auction):
        super().__init__(title=f"競標 - {auction.item_name}", timeout=300)
        self.auction = auction

        min_bid = auction.current_price + auction.increment
        currency = get_currency_display(auction.currency_type)

        self.bid_amount = TextInput(
            label="出價金額",
            placeholder=f"最低出價：{min_bid:,.2f} {currency} (支援小數點)",
            required=True,
            max_length=20,
        )

        self.add_item(self.bid_amount)

    async def callback(self, interaction: Interaction) -> None:
        try:
            bid_amount = float(self.bid_amount.value)
            min_bid = self.auction.current_price + self.auction.increment
            currency = get_currency_display(self.auction.currency_type)

            if bid_amount < min_bid:
                await interaction.response.send_message(
                    f"❌ 出價金額必須至少為 {min_bid:,.2f} {currency}!", ephemeral=True
                )
                return

            # 檢查用戶是否為當前最高出價者
            if self.auction.current_bidder_id == interaction.user.id:
                await interaction.response.send_message(
                    "❌ 你已經是當前最高出價者了!", ephemeral=True
                )
                return

            # 檢查用戶是否為拍賣創建者
            if self.auction.creator_id == interaction.user.id:
                await interaction.response.send_message(
                    "❌ 拍賣創建者不能參與自己的拍賣!", ephemeral=True
                )
                return

            # 處理出價
            db = AuctionDatabase()
            if self.auction.id is None:
                await interaction.response.send_message("❌ 拍賣ID無效!", ephemeral=True)
                return

            # 檢查是否在伺服器中執行命令
            if interaction.guild is None:
                await interaction.response.send_message(
                    "❌ 拍賣功能只能在伺服器中使用!", ephemeral=True
                )
                return

            success = db.place_bid(
                self.auction.id,
                interaction.user.id,
                interaction.user.display_name,
                bid_amount,
                interaction.guild.id,
            )

            if success:
                # 更新競標資訊
                updated_auction = db.get_auction(self.auction.id, interaction.guild.id)
                if updated_auction:
                    embed = self._create_auction_embed(updated_auction)
                    from .views import AuctionView
                    view = AuctionView(updated_auction)

                    await interaction.response.edit_message(
                        content=f"🎉 出價成功! {interaction.user.mention} 出價 {bid_amount:,.2f} {currency}",
                        embed=embed,
                        view=view,
                    )
                else:
                    await interaction.response.send_message(
                        "❌ 出價失敗，請稍後再試!", ephemeral=True
                    )
            else:
                await interaction.response.send_message(
                    "❌ 出價失敗，可能有其他人同時出價了!", ephemeral=True
                )

        except ValueError:
            await interaction.response.send_message("❌ 請輸入有效的數字格式!", ephemeral=True)
        except Exception as e:
            logfire.error(f"出價時發生錯誤: {e}")
            await interaction.response.send_message(
                "❌ 出價時發生錯誤，請稍後再試!", ephemeral=True
            )

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