import httpx
import re
from typing import Optional

SCRYFALL_API_BASE = "https://api.scryfall.com"
_HEADERS = {
    "User-Agent": "mtg-imgai/1.0",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}


class ScryfallRateLimitError(Exception):
    """Scryfall API のレート制限到達を表す例外。"""


def _normalize_name_for_compare(name: str) -> str:
    normalized = name.lower()
    normalized = re.sub(r"\bno\.\s*[0-9a-z/]+\b", "", normalized)
    normalized = re.sub(r"\s*[（(][^()（）]*[)）]\s*", " ", normalized)
    normalized = normalized.replace("//", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _candidate_name_matches(search_name: str, candidate_name: str) -> bool:
    needle = _normalize_name_for_compare(search_name)
    if not needle:
        return False

    haystacks = [_normalize_name_for_compare(candidate_name)]
    haystacks.extend(
        _normalize_name_for_compare(part)
        for part in candidate_name.split("//")
    )

    for haystack in haystacks:
        if not haystack:
            continue
        if haystack == needle or haystack.startswith(f"{needle} ") or f" {needle}" in haystack:
            return True
    return False


def _dedupe_cards(cards: list[dict]) -> list[dict]:
    deduped: dict[str, dict] = {}
    for card in cards:
        collector_number = card.get("collector_number")
        if not collector_number:
            continue
        deduped.setdefault(collector_number, card)
    return list(deduped.values())


async def _lookup_card(client: httpx.AsyncClient, set_code: str, collector_number: str) -> Optional[dict]:
    """
    Scryfall API でセットコード + コレクター番号からカード情報を取得する。
    見つからない場合は None を返す。
    """
    url = f"{SCRYFALL_API_BASE}/cards/{set_code.lower()}/{collector_number}"
    resp = await client.get(url, headers=_HEADERS)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


async def lookup_card(
    set_code: str,
    collector_number: str,
    client: httpx.AsyncClient | None = None,
) -> Optional[dict]:
    if client is not None:
        return await _lookup_card(client, set_code, collector_number)

    async with httpx.AsyncClient(timeout=10.0) as local_client:
        return await _lookup_card(local_client, set_code, collector_number)


async def _search_cards_by_name_set(
    client: httpx.AsyncClient,
    card_name_en: str,
    set_code: str,
    foil: bool = False,
) -> list[dict]:
    """
    カード英語名 + セットコードで Scryfall 検索し候補一覧を返す。
    完全一致で見つからない場合は部分一致へフォールバックする。
    """
    url = f"{SCRYFALL_API_BASE}/cards/search"
    queries = [
        f'!"{card_name_en}" set:{set_code.lower()}',
        f'name:"{card_name_en}" set:{set_code.lower()}',
        f'"{card_name_en}" set:{set_code.lower()}',
    ]

    seen_queries: set[str] = set()
    cards: list[dict] = []
    for query in queries:
        if query in seen_queries:
            continue
        seen_queries.add(query)

        resp = await client.get(
            url,
            params={"q": query, "unique": "prints"},
            headers=_HEADERS,
        )
        if resp.status_code == 404:
            continue
        if resp.status_code == 429:
            raise ScryfallRateLimitError("Scryfall API rate limited the request.")

        resp.raise_for_status()
        data = resp.json()
        raw_cards = data.get("data", [])
        cards = [
            card
            for card in raw_cards
            if card.get("set", "").lower() == set_code.lower()
            and _candidate_name_matches(card_name_en, card.get("name", ""))
        ]
        if cards:
            break

    # foil フラグで絞り込み（foil=True なら foil 版のみ）
    if foil:
        cards = [c for c in cards if c.get("foil")]
    else:
        cards = [c for c in cards if c.get("nonfoil")]

    return _dedupe_cards(cards)


async def search_cards_by_name_set(
    card_name_en: str,
    set_code: str,
    foil: bool = False,
    client: httpx.AsyncClient | None = None,
) -> list[dict]:
    if client is not None:
        return await _search_cards_by_name_set(client, card_name_en, set_code, foil=foil)

    async with httpx.AsyncClient(timeout=10.0) as local_client:
        return await _search_cards_by_name_set(
            local_client,
            card_name_en,
            set_code,
            foil=foil,
        )


async def enrich_card_number(item: dict, client: httpx.AsyncClient | None = None) -> dict:
    """
    スクレイプ結果 1件に Scryfall のカード番号を補完する。
    複数候補がある場合は candidates リストも返す。
    """
    number_hint = item.get("number_hint")
    if number_hint:
        return {**item, "card_number": number_hint, "candidates": []}

    card_name_en = item.get("search_name_en") or item.get("card_name_en")
    set_code = item.get("set_code")

    if not card_name_en or not set_code:
        return {**item, "card_number": None, "candidates": []}

    # Scryfall 検索（言語・foil で絞り込み）
    try:
        candidates = await search_cards_by_name_set(
            card_name_en=card_name_en,
            set_code=set_code,
            foil=item.get("foil", False),
            client=client,
        )
    except ScryfallRateLimitError:
        return {**item, "card_number": None, "candidates": [], "scryfall_error": "rate_limited"}

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
