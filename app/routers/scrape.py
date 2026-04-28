import asyncio
import time
import uuid
from fastapi import APIRouter, HTTPException

from app.schemas import (
    ScrapeRequest,
    ScrapeResponse,
    CardItem,
    ScrapeStartResponse,
    ScrapeStatusResponse,
)
from app.services.scraper import scrape_product_group
from app.services.scryfall import enrich_card_number
from app.services.vision import disambiguate_card_number

router = APIRouter(prefix="/api", tags=["scrape"])

# Scryfall API レート制限対策: リクエスト間隔 (秒)
_SCRYFALL_INTERVAL = 0.15
_JOBS: dict[str, dict] = {}


def _now() -> float:
    return time.time()


def _elapsed_seconds(job: dict) -> int:
    started = job.get("started_at")
    if not started:
        return 0
    finished = job.get("finished_at") or _now()
    return max(0, int(finished - started))


def _set_job_progress(job: dict, stage: str, processed: int, total: int) -> None:
    job["stage"] = stage
    job["processed"] = processed
    job["total"] = total


async def _run_scrape(url: str, job: dict | None = None) -> ScrapeResponse:
    try:
        raw_items = await scrape_product_group(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"スクレイピングエラー: {e}")

    if not raw_items:
        raise HTTPException(status_code=404, detail="商品が見つかりませんでした。")

    if job is not None:
        _set_job_progress(job, stage="scryfall", processed=0, total=len(raw_items))

    # Scryfall 補完: シリアル処理でレート制限を回避
    enriched: list[dict] = []
    for index, item in enumerate(raw_items, start=1):
        result = await enrich_card_number(item)
        enriched.append(result)
        if job is not None:
            _set_job_progress(job, stage="scryfall", processed=index, total=len(raw_items))
        await asyncio.sleep(_SCRYFALL_INTERVAL)

    # 複数候補が残ったものだけ AI で disambiguate
    ai_total = sum(1 for item in enriched if item.get("candidates", []))
    if job is not None:
        _set_job_progress(job, stage="ai", processed=0, total=ai_total)

    final: list[dict] = []
    ai_processed = 0
    for item in enriched:
        candidates = item.get("candidates", [])
        if not candidates:
            final.append({**item, "disambiguated_by_ai": False})
            continue

        number = await disambiguate_card_number(
            raw_name=item.get("raw_name", ""),
            candidates=candidates,
        )
        final.append({**item, "card_number": number, "disambiguated_by_ai": True, "candidates": []})
        ai_processed += 1
        if job is not None:
            _set_job_progress(job, stage="ai", processed=ai_processed, total=ai_total)

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

    if job is not None:
        _set_job_progress(job, stage="done", processed=len(items), total=len(items))

    return ScrapeResponse(total=len(items), items=items)


@router.post("/scrape/start", response_model=ScrapeStartResponse, summary="スクレイプ処理を開始する")
async def scrape_start(request: ScrapeRequest) -> ScrapeStartResponse:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "state": "queued",
        "stage": "queued",
        "processed": 0,
        "total": 0,
        "error": None,
        "started_at": None,
        "finished_at": None,
        "result": None,
    }
    _JOBS[job_id] = job

    async def _worker() -> None:
        job["state"] = "running"
        job["started_at"] = _now()
        try:
            result = await _run_scrape(str(request.url), job)
            job["result"] = result.model_dump()
            job["state"] = "completed"
            job["finished_at"] = _now()
        except HTTPException as e:
            job["state"] = "failed"
            job["error"] = str(e.detail)
            job["finished_at"] = _now()
        except Exception as e:
            job["state"] = "failed"
            job["error"] = f"Internal error: {e}"
            job["finished_at"] = _now()

    asyncio.create_task(_worker())
    return ScrapeStartResponse(job_id=job_id)


@router.get("/scrape/status/{job_id}", response_model=ScrapeStatusResponse, summary="スクレイプ進捗を取得する")
async def scrape_status(job_id: str) -> ScrapeStatusResponse:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")

    return ScrapeStatusResponse(
        job_id=job_id,
        state=job["state"],
        stage=job["stage"],
        processed=job["processed"],
        total=job["total"],
        elapsed_seconds=_elapsed_seconds(job),
        error=job.get("error"),
    )


@router.get("/scrape/result/{job_id}", response_model=ScrapeResponse, summary="スクレイプ結果を取得する")
async def scrape_result(job_id: str) -> ScrapeResponse:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")

    if job["state"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error") or "スクレイプに失敗しました。")

    if job["state"] != "completed":
        raise HTTPException(status_code=202, detail="処理中です。")

    result = job.get("result")
    if not result:
        raise HTTPException(status_code=500, detail="結果データがありません。")

    return ScrapeResponse(**result)


@router.post("/scrape", response_model=ScrapeResponse, summary="シングルスターの商品ページをスクレイプする")
async def scrape(request: ScrapeRequest) -> ScrapeResponse:
    """
    シングルスターの商品グループURLを受け取り、全商品の情報を返す。

    1. ページ全体をスクレイピング（ページネーション対応）
    2. 各商品の英語名 + セットコードで Scryfall 検索してカード番号を補完（シリアル処理）
    3. 複数候補が残った場合は AI で絞り込む
    """
    return await _run_scrape(str(request.url))
