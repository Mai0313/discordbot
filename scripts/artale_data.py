import json
from typing import Any
import contextlib
from urllib.parse import urljoin

from bs4 import Tag, BeautifulSoup
from rich.console import Console
from playwright.sync_api import sync_playwright

console = Console()

BASE_URL = "https://a2983456456.github.io/artale-drop/"


def parse_monster_card(card: Tag) -> dict:
    name = card.select_one(".monster-name").text.strip()
    image = urljoin(BASE_URL, card.select_one(".monster-image")["src"])

    # 屬性轉 dict
    attr_boxes = card.select(".attr-box")
    attr_dict = {}
    for attr in attr_boxes:
        text = attr.text.strip().replace("：", ":")  # 中文冒號換成英文冒號方便 split
        if ":" in text:
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()

            # key 對應表（中文 -> 英文）
            key_map = {
                "等級": "level",
                "HP": "hp",
                "MP": "mp",
                "經驗": "exp",
                "迴避": "evasion",
                "物理防禦": "pdef",
                "魔法防禦": "mdef",
                "命中需求": "accuracy_required",
            }
            if key in key_map:
                key = key_map[key]
                # 嘗試轉為 int，如果失敗保留原文字
                with contextlib.suppress(Exception):
                    value = int(value)
                attr_dict[key] = value

    # 出沒地圖
    map_names = [m.text.strip() for m in card.select(".map-name")]

    # 掉落物
    drop_items = []
    item_blocks = card.select("div.item")

    for item in item_blocks:
        a = item.select_one("a")
        img_tag = item.select_one("img")
        span = item.select_one("span")

        if a and img_tag and span:
            link = a["href"]
            img = urljoin(BASE_URL, img_tag["src"])
            item_name = span.text.strip()

            if "/equip/" in link:
                category = "裝備"
            elif "/item/" in link:
                category = "消耗品/素材"
            else:
                category = "其他"

            drop_items.append({"name": item_name, "type": category, "link": link, "img": img})

    return {
        "name": name,
        "image": image,
        "attributes": attr_dict,
        "maps": map_names,
        "drops": drop_items,
    }


def fetch_monster_cards() -> list[dict[str, Any]]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(BASE_URL)
        page.wait_for_timeout(3000)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        monster_cards = soup.select(".monster-card")
        console.print(f"✅ 抓取完成，共抓到 {len(monster_cards)} 筆資料")
        browser.close()
        results = []
        for monster_card in monster_cards:
            parsed_card = parse_monster_card(monster_card)
            results.append(parsed_card)
        return results


# 主程式
if __name__ == "__main__":
    data = fetch_monster_cards()

    # 儲存為 JSON
    with open("./data/monsters.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    console.print("✅ JSON 已儲存至 monsters.json")
