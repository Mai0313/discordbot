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
- Use the supplied market context as inspiration. If the stock is rising or has buy pressure, the joke can explain bullish behavior. If it is falling or has sell pressure, the joke can explain bearish behavior. If signals disagree, make the contradiction part of the deadpan joke.
- Keep sentiment broadly consistent with the headline and market context, but do not claim the fictional event is the only cause of the price move.
- Do not repeat the latest previous headline.
- Do not mention real people, real brands, real disasters, injuries, sexual content, hate, or politics.
- The headline should fit naturally in a Discord embed and should not include markdown.
""".strip()

STOCK_NEWS_BULLISH_FALLBACK_TEMPLATES: Final[tuple[tuple[str, int], ...]] = (
    ("{symbol} 公司吉祥物在電梯裡按錯樓層, 市場解讀為拓展新事業版圖", 95),
    ("{name} 自動販賣機拒收零錢, 法人認定成本控管太硬, 股價小漲", 70),
    ("{symbol} 會議室被實習生命名為宇宙飛船, 多頭喊出 {category} 想像空間", 125),
    ("{symbol} 財務長用計算機按出 888, 散戶覺得兆頭很會, 買盤突然排隊", 110),
    ("{name} 門口盆栽長太好, 市場謠傳現金流也會發芽, 股價走揚", 80),
    ("{symbol} 早會投影機自己修好, 法人稱營運韌性不可小看", 60),
    ("{symbol} 保全把貓當 VIP 放行, 市場相信高階客戶服務升級", 90),
)

STOCK_NEWS_BEARISH_FALLBACK_TEMPLATES: Final[tuple[tuple[str, int], ...]] = (
    ("{name} 董事長開車爆胎, 投資人懷疑輪胎 KPI 失守, 股價重挫", -145),
    ("{name} 午餐便當少一顆滷蛋, 管理層士氣遭質疑, 短線賣壓湧現", -85),
    ("{name} 茶水間咖啡太淡, 空方認為研發動能也被稀釋, 股價下探", -75),
    ("{symbol} 掃地機器人卡在門口, 法人擔心出貨也會卡住, 賣單默默變長", -100),
    ("{name} 早會白板筆全部沒水, 市場質疑 {category} 策略乾掉, 股價走弱", -90),
    ("{symbol} 電梯語音突然開始嘆氣, 投資人解讀為營運心情不美麗", -65),
)

STOCK_NEWS_NEUTRAL_FALLBACK_TEMPLATES: Final[tuple[tuple[str, int], ...]] = (
    ("{name} 茶水間貼出請勿偷喝豆漿公告, 多空雙方研究半天決定先觀望", 15),
    ("{symbol} 會議邀請標題只有一個問號, 市場情緒跟著進入待確認模式", -10),
    ("{name} 前台盆栽被移到左邊三公分, 交易員認為趨勢需要重新畫線", 20),
    ("{symbol} 影印機吐出空白紙, 法人表示資訊揭露非常極簡", -20),
)

STOCK_NEWS_FALLBACK_TEMPLATES: Final[tuple[tuple[str, int], ...]] = (
    STOCK_NEWS_BULLISH_FALLBACK_TEMPLATES
    + STOCK_NEWS_BEARISH_FALLBACK_TEMPLATES
    + STOCK_NEWS_NEUTRAL_FALLBACK_TEMPLATES
)

__all__ = [
    "STOCK_NEWS_BEARISH_FALLBACK_TEMPLATES",
    "STOCK_NEWS_BULLISH_FALLBACK_TEMPLATES",
    "STOCK_NEWS_FALLBACK_TEMPLATES",
    "STOCK_NEWS_NEUTRAL_FALLBACK_TEMPLATES",
    "STOCK_NEWS_PROMPT",
]
