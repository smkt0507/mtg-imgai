import asyncio
import httpx
from typing import Optional

SCRYFALL_API_BASE = "https://api.scryfall.com"
_HEADERS = {"User-Agent": "mtg-imgai/1.0"}


async def lookup_card(set_code: str, collector_number: str) -> Optional[dict]:
    """
    Scryfall API でセットコード + コレクター番号からカード情報を取得する。
    見つからない場合は None を返す。
    """
    url = f"{SCRYFALL_API_BASE}/cards/{set_code.lower()}/{collector_number}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=_HEADERS)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


async def search_cards_by_name_set(
    card_name_en: str,
    set_code: str,
    lang: Optional[str] = None,
    foil: bool = False,
) -> list[dict]:
    """
    カード英語名 + セットコードで Scryfall 全文検索し候補一覧を返す。
    言語・foilで絞り込む。
    """
    q_parts = [f'!"{card_name_en}"', f"set:{set_code.lower()}"]
    if lang == "jp":
        q_parts.append("lang:ja")
    elif lang == "en":
        q_parts.append("lang:en")

    query = " ".join(q_parts)
    url = f"{SCRYFALL_API_BASE}/cards/search"

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            url,
            params={"q": query, "unique": "prints"},
            headers=_HEADERS,
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        cards = data.get("data", [])

    # foil フラグで絞り込み（foil=True なら foil 版のみ）
    if foil:
        cards = [c for c in cards if c.get("foil")]
    else:
        cards = [c for c in cards if c.get("nonfoil")]

    return cards


async def enrich_card_number(item: dict) -> dict:
    """
    スクレイプ結果 1件に Scryfall のカード番号を補完する。
    複数候補がある場合は candidates リストも返す。
    """
    card_name_en = item.get("card_name_en")
    set_code = item.get("set_code")

    if not card_name_en or not set_code:
        return {**item, "card_number": None, "candidates": []}

    # Scryfall 検索（言語・foil で絞り込み）
    candidates = await search_cards_by_name_set(
        card_name_en=card_name_en,
        set_code=set_code,
        lang=item.get("lang"),
        foil=item.get("foil", False),
    )

    if len(candidates) == 1:
        return {**item, "card_number": candidates[0]["collector_number"], "candidates": []}

    # 複数 or 0件
    candidate_summaries = [
        {
            "collector_number": c["collector_number"],
            "name": c.get("name"),
            "frame_effects": c.get("frame_effects", []),
            "border_color": c.get("border_color"),
            "full_art": c.get("full_art", False),
            "promo": c.get("promo", False),
        }
        for c in candidates
    ]
    return {**item, "card_number": None, "candidates": candidate_summaries}



def extract_image_uri(card_data: dict) -> Optional[str]:
    """カードデータから正面画像URIを取得する。"""
    if "image_uris" in card_data:
        return card_data["image_uris"].get("normal")
    # 両面カードの場合は表面を使用
    faces = card_data.get("card_faces", [])
    if faces and "image_uris" in faces[0]:
        return faces[0]["image_uris"].get("normal")
    return None
