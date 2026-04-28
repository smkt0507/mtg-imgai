import asyncio
from fastapi import APIRouter, HTTPException

from app.schemas import ScrapeRequest, ScrapeResponse, CardItem
from app.services.scraper import scrape_product_group
from app.services.scryfall import enrich_card_number
from app.services.vision import disambiguate_card_number

router = APIRouter(prefix="/api", tags=["scrape"])

# Scryfall API レート制限対策: リクエスト間隔 (秒)
_SCRYFALL_INTERVAL = 0.15


@router.post("/scrape", response_model=ScrapeResponse, summary="シングルスターの商品ページをスクレイプする")
async def scrape(request: ScrapeRequest) -> ScrapeResponse:
    """
    シングルスターの商品グループURLを受け取り、全商品の情報を返す。

    1. ページ全体をスクレイピング（ページネーション対応）
    2. 各商品の英語名 + セットコードで Scryfall 検索してカード番号を補完（シリアル処理）
    3. 複数候補が残った場合は AI で絞り込む
    """
    url = str(request.url)

    try:
        raw_items = await scrape_product_group(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"スクレイピングエラー: {e}")

    if not raw_items:
        raise HTTPException(status_code=404, detail="商品が見つかりませんでした。")

    # Scryfall 補完: シリアル処理でレート制限を回避
    enriched: list[dict] = []
    for item in raw_items:
        result = await enrich_card_number(item)
        enriched.append(result)
        await asyncio.sleep(_SCRYFALL_INTERVAL)

    # 複数候補が残ったものだけ AI で disambiguate
    final: list[dict] = []
    for item in enriched:
        candidates = item.get("candidates", [])
        if not candidates:
            final.append({**item, "disambiguated_by_ai": False})
        else:
            number = await disambiguate_card_number(
                raw_name=item.get("raw_name", ""),
                candidates=candidates,
            )
            final.append({**item, "card_number": number, "disambiguated_by_ai": True, "candidates": []})

    items = [
        CardItem(
            raw_name=i.get("raw_name", ""),
            card_name_ja=i.get("card_name_ja"),
            card_name_en=i.get("card_name_en"),
            lang=i.get("lang"),
            foil=i.get("foil", False),
            set_code=i.get("set_code"),
            card_number=i.get("card_number"),
            price=i.get("price"),
            disambiguated_by_ai=i.get("disambiguated_by_ai", False),
        )
        for i in final
    ]

    return ScrapeResponse(total=len(items), items=items)
