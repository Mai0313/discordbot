import nextcord

from .state import get_participants
from .models import LotteryData, LotteryParticipant


def add_participants_field(embed: nextcord.Embed, participants: list[LotteryParticipant]) -> None:
    """Append a participant summary field to the provided embed."""
    names = [p.name for p in participants]
    total = len(names)
    embed.add_field(
        name=f"參與者（{total} 人）", value=", ".join(names) if names else "無", inline=False
    )


def build_creation_embed(lottery_data: LotteryData) -> nextcord.Embed:
    """Create the embed used to acknowledge lottery creation."""
    embed = nextcord.Embed(title="🎉 抽獎活動已創建!", color=0x00FF00)
    embed.add_field(name="活動標題", value=lottery_data.title, inline=False)
    embed.add_field(name="活動描述", value=lottery_data.description or "無", inline=False)
    embed.add_field(name="每次抽出", value=f"{lottery_data.draw_count} 人", inline=True)

    if lottery_data.registration_method == "youtube":
        if lottery_data.youtube_url:
            embed.add_field(name="YouTube直播", value=str(lottery_data.youtube_url), inline=False)
        if lottery_data.youtube_keyword:
            embed.add_field(
                name="報名關鍵字", value=str(lottery_data.youtube_keyword), inline=True
            )

    participants = get_participants(lottery_data.lottery_id)
    if participants:
        add_participants_field(embed, participants)
    else:
        embed.add_field(name="參與者", value="目前沒有參與者", inline=False)
    return embed
