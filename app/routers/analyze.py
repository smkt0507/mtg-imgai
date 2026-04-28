from fastapi import APIRouter, HTTPException

from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.vision import (
    AIProviderError,
    AIQuotaExceededError,
    analyze_card_image,
)
from app.services.scryfall import lookup_card, extract_image_uri

router = APIRouter(prefix="/api", tags=["mtg"])


@router.post("/analyze", response_model=AnalyzeResponse, summary="MTGカード画像を解析する")
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    MTGカードの画像URLを受け取り、セットコードとコレクター番号を返す。

    1. AI Vision で画像を解析してセットコード・番号を抽出
    2. Scryfall API で検証・補完
    """
    image_url = str(request.image_url)

    # Step 1: AI解析
    try:
        ai_result = await analyze_card_image(image_url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"AI解析エラー: {e}")
    except AIQuotaExceededError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except AIProviderError as e:
        raise HTTPException(status_code=502, detail=f"AIプロバイダーエラー: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI APIエラー: {e}")

    set_code: str = (ai_result.get("set_code") or "").strip().upper()
    collector_number: str = (ai_result.get("collector_number") or "").strip()
    card_name: str | None = ai_result.get("card_name")

    if not set_code or not collector_number:
        raise HTTPException(
            status_code=422,
            detail="AIがセットコードまたはコレクター番号を特定できませんでした。",
        )

    # Step 2: Scryfall検証
    try:
        scryfall_data = await lookup_card(set_code, collector_number)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Scryfall APIエラー: {e}")

    if scryfall_data:
        return AnalyzeResponse(
            set_code=scryfall_data.get("set", set_code).upper(),
            collector_number=scryfall_data.get("collector_number", collector_number),
            card_name=scryfall_data.get("name", card_name),
            scryfall_uri=scryfall_data.get("scryfall_uri"),
            scryfall_image_uri=extract_image_uri(scryfall_data),
            validated=True,
        )

    # Scryfall で見つからなかった場合はAI結果をそのまま返す
    return AnalyzeResponse(
        set_code=set_code,
        collector_number=collector_number,
        card_name=card_name,
        scryfall_uri=None,
        scryfall_image_uri=None,
        validated=False,
    )
