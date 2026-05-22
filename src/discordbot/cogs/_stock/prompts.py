"""Prompts and copy templates for simulated stock news."""

from typing import Final

STOCK_NEWS_PROMPT = """
You write one short fictional news headline for a Discord bot's simulated stock market.
The company is virtual. Do not claim this is real financial news, real investment advice, or a real exchange event.
Return one concise Traditional Chinese headline and a market sentiment value in basis points from -180 to 180.

Style:
- Make the headline absurd, goofy, and deadpan, like a fake tabloid finance headline.
- Use ridiculous but harmless fictional causes for market moves, such as a chairman getting a flat tire, a mascot winning an argument, a vending machine refusing coins, or an intern naming a meeting room.
- Keep the market reaction tied to the joke, such as 股價重挫, 市場暴衝, 投資人笑到下單, or 法人當場沉默.
- Do not mention real people, real brands, real disasters, injuries, sexual content, hate, or politics.
- The headline should fit naturally in a Discord embed and should not include markdown.
""".strip()

STOCK_NEWS_FALLBACK_TEMPLATES: Final[tuple[tuple[str, int], ...]] = (
    ("{name} 董事長開車爆胎, 投資人懷疑輪胎 KPI 失守, 股價重挫", -145),
    ("{symbol} 公司吉祥物在電梯裡按錯樓層, 市場解讀為拓展新事業版圖", 95),
    ("{name} 自動販賣機拒收零錢, 法人認定成本控管太硬, 股價小漲", 70),
    ("{symbol} 會議室被實習生命名為宇宙飛船, 多頭喊出 {category} 想像空間", 125),
    ("{name} 午餐便當少一顆滷蛋, 管理層士氣遭質疑, 短線賣壓湧現", -85),
    ("{symbol} 財務長用計算機按出 888, 散戶覺得兆頭很會, 買盤突然排隊", 110),
    ("{name} 門口盆栽長太好, 市場謠傳現金流也會發芽, 股價走揚", 80),
    ("{symbol} 早會投影機自己修好, 法人稱營運韌性不可小看", 60),
    ("{name} 茶水間咖啡太淡, 空方認為研發動能也被稀釋, 股價下探", -75),
    ("{symbol} 保全把貓當 VIP 放行, 市場相信高階客戶服務升級", 90),
)

__all__ = ["STOCK_NEWS_FALLBACK_TEMPLATES", "STOCK_NEWS_PROMPT"]
