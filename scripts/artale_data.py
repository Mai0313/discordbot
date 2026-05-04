"""Artale MapleStory data scraper.

Extracts game data from artalemaplestory.com by parsing the Next.js RSC
(React Server Components) payload embedded in each page's HTML.
"""

import re
import json
import time
from pathlib import Path
from collections.abc import Mapping

import requests
from rich.console import Console

console = Console()

# Raw JSON record from the website RSC payload
type JsonRecord = dict[str, str | int | float | bool | list | dict | None]

BASE_URL = "https://www.artalemaplestory.com"
LOCALE = "zh"
DATA_DIR = Path("./data/maplestory")

# (url_path, rsc_data_key)
CATEGORIES: dict[str, tuple[str, str]] = {
    "monsters": ("monsters", "inScopeMonsters"),
    "equipment": ("equipment", "inScopeEquipmentItems"),
    "scrolls": ("scrolls", "inScopeScrolls"),
    "useable": ("useable", "inScopeUseableItems"),
    "npcs": ("npcs", "inScopeNpcs"),
    "quests": ("quests", "inScopeQuests"),
    "misc": ("misc", "inScopeMiscItems"),
}


def fetch_html(url: str, *, max_retries: int = 3) -> str:
    """Fetches HTML content from a URL with retry logic.

    Args:
        url: The URL to fetch.
        max_retries: Maximum number of retries on failure.

    Returns:
        The HTML content of the page.

    Raises:
        requests.RequestException: If fetching fails after max_retries.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
    return ""  # unreachable


def extract_rsc_text(html: str) -> str:
    """Concatenates all RSC payload chunks from script tags.

    Args:
        html: The raw HTML content containing Next.js RSC payloads.

    Returns:
        The concatenated RSC text payload.
    """
    chunks = re.findall(r"self\.__next_f\.push\(\s*(\[.*?\])\s*\)", html, re.DOTALL)
    parts: list[str] = []
    for raw in chunks:
        try:
            parsed = json.loads(raw)
            if len(parsed) >= 2 and isinstance(parsed[1], str):
                parts.append(parsed[1])
        except (json.JSONDecodeError, IndexError):
            pass
    return "".join(parts)


def _raw_decode_value(text: str, key: str) -> list[JsonRecord] | dict[str, str] | None:
    """Extract the JSON value for a given key using json.JSONDecoder.raw_decode."""
    pattern = f'"{key}":'
    try:
        idx = text.index(pattern)
    except ValueError:
        return None
    start = idx + len(pattern)
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start)
        return value
    except json.JSONDecodeError:
        return None


def extract_json_list(text: str, key: str) -> list[JsonRecord] | None:
    """Extracts a JSON array for a given key.

    Args:
        text: The text containing JSON-like structures.
        key: The key to look for in the text.

    Returns:
        A list of JSON records if found and is a list, otherwise None.
    """
    value = _raw_decode_value(text, key)
    return value if isinstance(value, list) else None


def _get_nested_dict(d: Mapping[str, object], *keys: str) -> dict[str, str]:
    """Traverse nested dicts safely, returning {} if any key is missing."""
    current: object = d
    for k in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(k, {})
    return current if isinstance(current, dict) else {}


def extract_translations(rsc_text: str) -> dict[str, dict[str, str]]:
    """Extracts all name translation dicts from the RSC messages section.

    Args:
        rsc_text: The concatenated RSC text payload.

    Returns:
        A dictionary mapping categories to their translation dictionaries.
    """
    raw = _raw_decode_value(rsc_text, "messages")
    if not isinstance(raw, dict):
        return {}
    messages = raw
    translations: dict[str, dict[str, str]] = {}

    # Entity name translations (en -> zh)
    for key in (
        "monsters",
        "equipment",
        "scrolls",
        "useable",
        "misc",
        "maps",
        "quests",
        "npcs",
        "skill",
    ):
        val = messages.get(key)
        if isinstance(val, dict):
            translations[key] = val
    # Enum translations
    maple = messages.get("maple")
    if isinstance(maple, dict):
        for key in ("job", "eqType", "region"):
            val = maple.get(key)
            if isinstance(val, dict):
                translations[key] = val
        for parent_key, trans_key in [("misc", "miscType"), ("npc", "npcType")]:
            nested = _get_nested_dict(maple, parent_key, "type")
            if nested:
                translations[trans_key] = nested

    modifiers = _get_nested_dict(messages, "monster", "modifiers")
    if modifiers:
        translations["modifiers"] = modifiers

    return translations


def apply_name_translations(items: list[JsonRecord], name_dict: dict[str, str]) -> None:
    """Adds 'nameZh' field to each item from translation dictionary.

    Args:
        items: A list of JSON records to be updated.
        name_dict: A dictionary mapping English names to Traditional Chinese names.
    """
    for item in items:
        en_name = item.get("name")
        if isinstance(en_name, str) and en_name in name_dict:
            item["nameZh"] = name_dict[en_name]


def scrape_category(
    category: str, url_path: str, rsc_key: str, translations: dict[str, dict[str, str]]
) -> list[JsonRecord]:
    """Scrapes data for a specific category from the website.

    Args:
        category: The category name (e.g., "monsters").
        url_path: The URL path segment for this category.
        rsc_key: The key in the RSC payload containing the data.
        translations: The loaded translation dictionaries.

    Returns:
        A list of JSON records for the category.
    """
    url = f"{BASE_URL}/{LOCALE}/{url_path}"
    console.print(f"  📖 {url}")
    html = fetch_html(url)
    rsc_text = extract_rsc_text(html)

    data = extract_json_list(rsc_text, rsc_key)
    if data is None:
        console.print(f"  [red]✗ Key '{rsc_key}' not found[/red]")
        return []

    name_dict = translations.get(category, {})
    if name_dict:
        apply_name_translations(data, name_dict)

    return data


def scrape_maps(translations: dict[str, dict[str, str]]) -> list[JsonRecord]:
    """Scrapes maps by fetching each region page.

    Args:
        translations: The loaded translation dictionaries.

    Returns:
        A list of JSON records for all maps.
    """
    # Discover region slugs from main maps page
    main_url = f"{BASE_URL}/{LOCALE}/maps"
    console.print(f"  📖 {main_url}")
    html = fetch_html(main_url)
    rsc_text = extract_rsc_text(html)

    region_slugs = sorted(
        s
        for s in set(re.findall(r"/maps/([a-z0-9-]+)", rsc_text))
        if s != "zh" and not s.startswith("page-")
    )
    console.print(f"  📍 Found {len(region_slugs)} regions")

    map_names = translations.get("maps", {})
    all_maps: list[JsonRecord] = []

    for slug in region_slugs:
        time.sleep(1)
        url = f"{BASE_URL}/{LOCALE}/maps/{slug}"
        console.print(f"    📖 {url}")
        try:
            html = fetch_html(url)
            rsc_text = extract_rsc_text(html)
            maps = extract_json_list(rsc_text, "maps")
            if maps is not None:
                if map_names:
                    apply_name_translations(maps, map_names)
                all_maps.extend(maps)
                console.print(f"    ✓ {len(maps)} maps")
            else:
                console.print("    [yellow]⚠ No maps data[/yellow]")
        except requests.RequestException as e:
            console.print(f"    [red]✗ {e}[/red]")

    return all_maps


def save_json(data: list[JsonRecord] | dict[str, dict[str, str]], filename: str) -> Path:
    """Saves data to a JSON file in the data directory.

    Args:
        data: The data to save (list or dict).
        filename: The name of the file (without extension).

    Returns:
        The path to the saved file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{filename}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    """Orchestrates the scraping process for all categories and maps."""
    console.print("[bold]🍁 Artale Data Scraper[/bold]\n")

    # Step 1: fetch translations from first page (all translations are shared)
    console.print("[bold cyan]Fetching translations...[/bold cyan]")
    html = fetch_html(f"{BASE_URL}/{LOCALE}/monsters")
    rsc_text = extract_rsc_text(html)
    translations = extract_translations(rsc_text)
    save_json(translations, "translations")
    console.print(
        "  ✓ Translations: " + ", ".join(f"{k}={len(v)}" for k, v in translations.items())
    )

    # Step 2: extract monsters from the already-fetched page
    console.print("\n[bold cyan]Scraping categories...[/bold cyan]")

    monsters = extract_json_list(rsc_text, "inScopeMonsters")
    if monsters is not None:
        apply_name_translations(monsters, translations.get("monsters", {}))
        save_json(monsters, "monsters")
        console.print(f"  ✓ monsters: {len(monsters)} items")
    else:
        console.print("  [red]✗ monsters: failed[/red]")

    # Step 3: scrape remaining categories
    for category, (url_path, rsc_key) in CATEGORIES.items():
        if category == "monsters":
            continue  # already done above
        time.sleep(1)
        data = scrape_category(category, url_path, rsc_key, translations)
        if data:
            save_json(data, category)
            console.print(f"  ✓ {category}: {len(data)} items")
        else:
            console.print(f"  [red]✗ {category}: no data[/red]")

    # Step 4: scrape maps (multiple region pages)
    console.print("\n[bold cyan]Scraping maps...[/bold cyan]")
    maps = scrape_maps(translations)
    if maps:
        save_json(maps, "maps")
        console.print(f"  ✓ maps: {len(maps)} items")
    else:
        console.print("  [red]✗ maps: no data[/red]")

    console.print("\n[bold green]✓ Done![/bold green]")


if __name__ == "__main__":
    main()
