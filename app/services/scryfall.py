import httpx
from typing import Optional

SCRYFALL_API_BASE = "https://api.scryfall.com"


async def lookup_card(set_code: str, collector_number: str) -> Optional[dict]:
    """
    Scryfall API でセットコード + コレクター番号からカード情報を取得する。
    見つからない場合は None を返す。
    """
    url = f"{SCRYFALL_API_BASE}/cards/{set_code.lower()}/{collector_number}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers={"User-Agent": "mtg-imgai/1.0"})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


def extract_image_uri(card_data: dict) -> Optional[str]:
    """カードデータから正面画像URIを取得する。"""
    if "image_uris" in card_data:
        return card_data["image_uris"].get("normal")
    # 両面カードの場合は表面を使用
    faces = card_data.get("card_faces", [])
    if faces and "image_uris" in faces[0]:
        return faces[0]["image_uris"].get("normal")
    return None
