import json
from typing import Any
from pathlib import Path

from rich.console import Console
from playwright.sync_api import sync_playwright

console = Console()

BASE_URL = "https://www.artalemaplestory.com/zh/monsters"


def fetch_monster_cards() -> list[dict[str, Any]]:
    """獲取所有怪物卡片資料"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        all_monsters = []
        page_num = 1

        console.print("🔄 開始抓取怪物資料...")

        while True:
            # 構建分頁URL
            url = f"{BASE_URL}?viewMode=grid&pageSize=100&page={page_num}"
            console.print(f"📖 正在處理第 {page_num} 頁: {url}")

            try:
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(3000)

                # 使用 Playwright 直接獲取怪物資料
                monsters_on_page = page.evaluate("""
                    () => {
                        const monsters = [];

                        // 尋找所有包含怪物資料的連結
                        const monsterLinks = Array.from(document.querySelectorAll('a[href*="/monsters/"]'));

                        monsterLinks.forEach(link => {
                            try {
                                // 找到包含完整怪物資料的父容器
                                let container = link;
                                while (container && !container.textContent.includes('HP：')) {
                                    container = container.parentElement;
                                    if (!container || container.tagName === 'BODY') break;
                                }

                                if (!container) return;

                                const monsterName = link.textContent.trim();
                                const monsterUrl = link.href;

                                if (!monsterName || monsters.some(m => m.name === monsterName)) return;

                                // 提取等級
                                const levelMatch = container.textContent.match(/LV\\.\\s*(\\d+)/);
                                const level = levelMatch ? parseInt(levelMatch[1]) : null;

                                // 提取基本屬性
                                const text = container.textContent;
                                const attributes = {};

                                if (level) attributes.level = level;

                                // 解析各種屬性
                                const patterns = {
                                    hp: /HP：\\s*([\\d,]+)/,
                                    mp: /MP：\\s*([\\d,]+)/,
                                    exp: /EXP：\\s*([\\d,]+)/,
                                    evasion: /迴避：\\s*([\\d,]+)/,
                                    pdef: /物防：\\s*([\\d,]+)/,
                                    mdef: /魔防：\\s*([\\d,]+)/,
                                    accuracy_required: /命中需求：\\s*([^\\n]+)/,
                                    meso_range: /楓幣範圍：\\s*([\\d,\\s-]+)/
                                };

                                Object.entries(patterns).forEach(([key, pattern]) => {
                                    const match = text.match(pattern);
                                    if (match) {
                                        let value = match[1].trim();
                                        // 嘗試轉換為數字
                                        if (key !== 'accuracy_required' && key !== 'meso_range') {
                                            const numValue = parseInt(value.replace(/,/g, ''));
                                            if (!isNaN(numValue)) value = numValue;
                                        }
                                        attributes[key] = value;
                                    }
                                });

                                // 提取屬性標籤（弱火、強冰等）
                                const elementAttributes = [];
                                const elementMatches = text.match(/(弱|強|免疫)[冰雷火毒聖]/g);
                                if (elementMatches) {
                                    elementAttributes.push(...elementMatches);
                                }

                                // 提取地圖
                                const maps = [];
                                const mapLinks = Array.from(container.querySelectorAll('a[href*="/maps/"]'));
                                mapLinks.forEach(mapLink => {
                                    const mapName = mapLink.textContent.trim();
                                    if (mapName && !maps.includes(mapName)) {
                                        maps.push(mapName);
                                    }
                                });

                                // 提取掉落物
                                const drops = [];
                                const itemLinks = Array.from(container.querySelectorAll('a[href*="/equipment/"], a[href*="/useable/"], a[href*="/scrolls/"], a[href*="/misc/"]'));
                                itemLinks.forEach(itemLink => {
                                    const itemName = itemLink.textContent.trim();
                                    const itemUrl = itemLink.href;

                                    if (!itemName) return;

                                    let category = "其它";
                                    if (itemUrl.includes("/equipment/")) category = "裝備";
                                    else if (itemUrl.includes("/useable/")) category = "消耗品";
                                    else if (itemUrl.includes("/scrolls/")) category = "捲軸";
                                    else if (itemUrl.includes("/misc/")) category = "其它";

                                    // 獲取圖片
                                    const img = itemLink.querySelector('img');
                                    let imgSrc = "";
                                    if (img && img.src) {
                                        imgSrc = img.src.startsWith('http') ? img.src :
                                                 'https://www.artalemaplestory.com' + img.src;
                                    }

                                    drops.push({
                                        name: itemName,
                                        type: category,
                                        link: itemUrl,
                                        img: imgSrc
                                    });
                                });

                                // 獲取怪物圖片
                                let monsterImg = "";
                                const imgElement = container.querySelector('img');
                                if (imgElement && imgElement.src) {
                                    monsterImg = imgElement.src.startsWith('http') ? imgElement.src :
                                                'https://www.artalemaplestory.com' + imgElement.src;
                                }

                                monsters.push({
                                    name: monsterName,
                                    image: monsterImg,
                                    attributes: attributes,
                                    element_attributes: elementAttributes,
                                    maps: maps,
                                    drops: drops,
                                    url: monsterUrl
                                });

                            } catch (error) {
                                console.error('解析怪物資料時出錯:', error);
                            }
                        });

                        return monsters;
                    }
                """)

                if not monsters_on_page:
                    console.print(f"❌ 第 {page_num} 頁沒有找到怪物資料，結束抓取")
                    break

                all_monsters.extend(monsters_on_page)
                console.print(
                    f"✅ 第 {page_num} 頁成功解析 {len(monsters_on_page)} 個怪物，累計: {len(all_monsters)}"
                )

                # 檢查是否還有下一頁
                if len(monsters_on_page) < 100:  # pageSize=100
                    console.print("📄 這是最後一頁")
                    break

                page_num += 1

            except Exception as e:
                console.print(f"❌ 處理第 {page_num} 頁時出錯: {e}")
                break

        browser.close()

        console.print(f"✅ 抓取完成，共抓到 {len(all_monsters)} 筆怪物資料")

        # 儲存結果
        output_path = Path("./data/monsters.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_monsters, f, ensure_ascii=False, indent=4)

        console.print(f"✅ JSON 已儲存至 {output_path}")
        return all_monsters


# 主程式
if __name__ == "__main__":
    fetch_monster_cards()
