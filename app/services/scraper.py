import re
import asyncio
from typing import Optional
import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://www.singlestar.jp"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def _parse_goods_name(raw: str) -> dict:
    """
    商品名テキストをパースして各フィールドを返す。

    例:
      [FOIL] 恐ろしき癒し手、アンチヴェノム/Anti-Venom, Horrifying Healer 【英語版】 [SPM-白MR]
      マーベル スパイダーマン/Spider-Man 【日本語版】 [SPM-黒R]
    """
    raw = raw.strip()

    # foil
    foil = raw.startswith("[FOIL]")
    text = raw.removeprefix("[FOIL]").strip()

    # lang
    if "日本語版" in text or "日本語" in text:
        lang = "jp"
    elif "英語版" in text or "英語" in text:
        lang = "en"
    else:
        lang = None

    # set_code: [SPM-...] or 【SPM】 形式に対応
    set_match = re.search(r"\[([A-Z0-9]{2,6})[^\]]*\]", text)
    set_code = set_match.group(1).upper() if set_match else None

    # カード名: "/" より後が英語名、前が日本語名
    # 【...】 や [...]  以降を除去
    name_part = re.split(r"【|〔|\[", text)[0].strip()

    if "/" in name_part:
        card_name_ja, card_name_en = name_part.split("/", 1)
        card_name_ja = card_name_ja.strip()
        card_name_en = card_name_en.strip()
    else:
        card_name_ja = None
        card_name_en = name_part.strip() or None

    return {
        "raw_name": raw,
        "card_name_ja": card_name_ja,
        "card_name_en": card_name_en,
        "lang": lang,
        "foil": foil,
        "set_code": set_code,
    }


def _parse_price(price_text: str) -> Optional[int]:
    """価格テキスト（例: '400円'）から整数を返す。"""
    digits = re.sub(r"[^\d]", "", price_text)
    return int(digits) if digits else None


async def _fetch_page(client: httpx.AsyncClient, url: str) -> BeautifulSoup:
    resp = await client.get(url, headers=HEADERS, follow_redirects=True)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


async def scrape_product_group(url: str) -> list[dict]:
    """
    シングルスターの商品グループページ全ページをスクレイピングして
    商品情報リストを返す（ページネーション対応）。
    """
    results = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1ページ目を取得してページ数を把握
        soup = await _fetch_page(client, url)
        items = _parse_items(soup)
        results.extend(items)

        # ページャから全ページ数を取得
        pager = soup.select(".pager a.pager_btn")
        page_nums = []
        for a in pager:
            href = a.get("href", "")
            m = re.search(r"page=(\d+)", href)
            if m:
                page_nums.append(int(m.group(1)))

        max_page = max(page_nums) if page_nums else 1

        # 2ページ目以降を並列取得
        if max_page > 1:
            tasks = [
                _fetch_page(client, f"{url}?page={p}")
                for p in range(2, max_page + 1)
            ]
            pages = await asyncio.gather(*tasks, return_exceptions=True)
            for page in pages:
                if isinstance(page, Exception):
                    continue
                results.extend(_parse_items(page))

    return results


def _parse_items(soup: BeautifulSoup) -> list[dict]:
    items = []
    for item_el in soup.select(".item_list_area li, .item_list li"):
        name_el = item_el.select_one(".goods_name")
        price_el = item_el.select_one(".figure")
        if not name_el:
            continue

        raw_name = name_el.get_text(strip=True)
        parsed = _parse_goods_name(raw_name)
        parsed["price"] = _parse_price(price_el.get_text()) if price_el else None
        parsed["card_number"] = None  # Scryfall で後から補完
        items.append(parsed)

    return items
