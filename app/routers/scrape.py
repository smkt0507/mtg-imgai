import asyncio
import json
import os
import time
import uuid
import httpx
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas import (
    ScrapeRequest,
    ScrapeResponse,
    CardItem,
    ScrapeStartResponse,
    ScrapeStatusResponse,
)
from app.services.scraper import scrape_product_group
from app.services.scryfall import enrich_card_number
from app.services.vision import disambiguate_card_number, infer_card_number_from_product

router = APIRouter(prefix="/api", tags=["scrape"])

_JOBS: dict[str, dict] = {}
_MAX_RUNNING_JOBS = 1
_JOB_TTL_SECONDS = 1800
_JOB_DIR = "/tmp/mtg_imgai_jobs"

os.makedirs(_JOB_DIR, exist_ok=True)


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


def _job_result_path(job_id: str) -> str:
    return os.path.join(_JOB_DIR, f"{job_id}.json")


def _cleanup_jobs() -> None:
    now = _now()
    delete_ids: list[str] = []
    for job_id, job in _JOBS.items():
        finished_at = job.get("finished_at")
        if finished_at and now - finished_at > _JOB_TTL_SECONDS:
            delete_ids.append(job_id)

    for job_id in delete_ids:
        _JOBS.pop(job_id, None)
        result_path = _job_result_path(job_id)
        if os.path.exists(result_path):
            try:
                os.remove(result_path)
            except OSError:
                pass


def _running_jobs_count() -> int:
    return sum(1 for job in _JOBS.values() if job.get("state") == "running")


class _IntervalLimiter:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.0, interval_seconds)
        self._lock = asyncio.Lock()
        self._next_available = 0.0

    async def wait(self) -> None:
        if self.interval_seconds <= 0:
            return

        async with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_available - now)
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()
            self._next_available = now + self.interval_seconds


def _enrich_cache_key(item: dict) -> tuple[str | None, str | None, bool, str | None]:
    return (
        item.get("search_name_en") or item.get("card_name_en"),
        item.get("set_code"),
        bool(item.get("foil", False)),
        item.get("number_hint"),
    )


def _disambiguation_cache_key(item: dict, candidates: list[dict]) -> tuple[object, ...]:
    return (
        item.get("search_name_en") or item.get("card_name_en") or item.get("raw_name"),
        item.get("variant_signature"),
        item.get("set_code"),
        bool(item.get("foil", False)),
        tuple(
            (
                candidate.get("collector_number"),
                candidate.get("name"),
                tuple(candidate.get("frame_effects") or []),
                candidate.get("border_color"),
                bool(candidate.get("full_art")),
                bool(candidate.get("promo")),
            )
            for candidate in candidates
        ),
    )


def _ai_fallback_cache_key(item: dict) -> tuple[object, ...]:
    return (
        item.get("search_name_en") or item.get("card_name_en") or item.get("raw_name"),
        item.get("variant_signature"),
        item.get("set_code"),
        bool(item.get("foil", False)),
        item.get("number_hint"),
    )


async def _run_scrape(url: str, job: dict | None = None) -> ScrapeResponse:
    try:
        raw_items = await scrape_product_group(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"スクレイピングエラー: {e}")

    if not raw_items:
        raise HTTPException(status_code=404, detail="商品が見つかりませんでした。")

    if job is not None:
        _set_job_progress(job, stage="processing", processed=0, total=len(raw_items))

    final: list[dict | None] = [None] * len(raw_items)
    processed_count = 0
    progress_lock = asyncio.Lock()
    cache_lock = asyncio.Lock()
    process_semaphore = asyncio.Semaphore(settings.scrape_concurrency)
    ai_semaphore = asyncio.Semaphore(settings.ai_disambiguation_concurrency)
    scryfall_limiter = _IntervalLimiter(settings.scryfall_interval_seconds)
    enrich_tasks: dict[tuple[str | None, str | None, bool, str | None], asyncio.Task] = {}
    disambiguation_tasks: dict[tuple[object, ...], asyncio.Task] = {}
    ai_fallback_tasks: dict[tuple[object, ...], asyncio.Task] = {}
    http_limits = httpx.Limits(
        max_connections=max(8, settings.scrape_concurrency * 2),
        max_keepalive_connections=max(4, settings.scrape_concurrency),
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), limits=http_limits) as client:
        async def _get_enriched_item(raw_item: dict) -> dict:
            key = _enrich_cache_key(raw_item)
            if not key[0] or not key[1]:
                return await enrich_card_number(raw_item)

            async with cache_lock:
                task = enrich_tasks.get(key)
                if task is None:
                    async def _run_enrich() -> dict:
                        await scryfall_limiter.wait()
                        return await enrich_card_number(raw_item, client=client)

                    task = asyncio.create_task(_run_enrich())
                    enrich_tasks[key] = task

            return await task

        async def _get_card_number(item: dict, candidates: list[dict]) -> tuple[str | None, bool]:
            key = _disambiguation_cache_key(item, candidates)
            async with cache_lock:
                task = disambiguation_tasks.get(key)
                if task is None:
                    async def _run_disambiguation() -> tuple[str | None, bool]:
                        async with ai_semaphore:
                            return await disambiguate_card_number(
                                raw_name=item.get("raw_name", ""),
                                candidates=candidates,
                                client_http=client,
                            )

                    task = asyncio.create_task(_run_disambiguation())
                    disambiguation_tasks[key] = task

            return await task

        async def _infer_number_with_ai(item: dict) -> tuple[str | None, bool]:
            key = _ai_fallback_cache_key(item)
            async with cache_lock:
                task = ai_fallback_tasks.get(key)
                if task is None:
                    async def _run_infer() -> tuple[str | None, bool]:
                        async with ai_semaphore:
                            number = await infer_card_number_from_product(item, client_http=client)
                            return number, bool(number)

                    task = asyncio.create_task(_run_infer())
                    ai_fallback_tasks[key] = task

            return await task

        async def _process_item(index: int, raw_item: dict) -> None:
            nonlocal processed_count

            async with process_semaphore:
                item = await _get_enriched_item(raw_item)
                candidates = item.get("candidates", [])
                if item.get("card_number"):
                    resolved = {**item, "disambiguated_by_ai": False, "candidates": []}
                elif not candidates:
                    number, used_ai = await _infer_number_with_ai(item)
                    resolved = {
                        **item,
                        "card_number": number,
                        "disambiguated_by_ai": used_ai,
                        "candidates": [],
                    }
                else:
                    number, used_ai = await _get_card_number(item, candidates)
                    resolved = {
                        **item,
                        "card_number": number,
                        "disambiguated_by_ai": used_ai,
                        "candidates": [],
                    }

                final[index] = resolved

                if job is not None:
                    async with progress_lock:
                        processed_count += 1
                        _set_job_progress(job, stage="processing", processed=processed_count, total=len(raw_items))

        tasks = [
            asyncio.create_task(_process_item(index, raw_item))
            for index, raw_item in enumerate(raw_items)
        ]
        await asyncio.gather(*tasks)

    if any(item is None for item in final):
        raise HTTPException(status_code=500, detail="一部の結果を組み立てられませんでした。")

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
        if i is not None
    ]

    if job is not None:
        _set_job_progress(job, stage="done", processed=len(items), total=len(items))

    return ScrapeResponse(total=len(items), items=items)


@router.post("/scrape/start", response_model=ScrapeStartResponse, summary="スクレイプ処理を開始する")
async def scrape_start(request: ScrapeRequest) -> ScrapeStartResponse:
    _cleanup_jobs()
    if _running_jobs_count() >= _MAX_RUNNING_JOBS:
        raise HTTPException(status_code=429, detail="現在処理中のジョブがあります。しばらく待ってから再実行してください。")

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
    }
    _JOBS[job_id] = job

    async def _worker() -> None:
        job["state"] = "running"
        job["started_at"] = _now()
        try:
            result = await _run_scrape(str(request.url), job)
            with open(_job_result_path(job_id), "w", encoding="utf-8") as fp:
                json.dump(result.model_dump(), fp, ensure_ascii=False)
            job["state"] = "completed"
            job["finished_at"] = _now()
        except HTTPException as e:
            job["state"] = "failed"
            job["error"] = str(e.detail)
            job["finished_at"] = _now()
        except Exception as e:
            job["state"] = "failed"
            exc_name = type(e).__name__
            exc_msg = str(e).strip() or "no message"
            job["error"] = f"Internal error ({exc_name}): {exc_msg}"
            job["finished_at"] = _now()

    asyncio.create_task(_worker())
    return ScrapeStartResponse(job_id=job_id)


@router.get("/scrape/status/{job_id}", response_model=ScrapeStatusResponse, summary="スクレイプ進捗を取得する")
async def scrape_status(job_id: str) -> ScrapeStatusResponse:
    _cleanup_jobs()
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
    _cleanup_jobs()
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")

    if job["state"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error") or "スクレイプに失敗しました。")

    if job["state"] != "completed":
        raise HTTPException(status_code=202, detail="処理中です。")

    result_path = _job_result_path(job_id)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=500, detail="結果データがありません。")

    try:
        with open(result_path, "r", encoding="utf-8") as fp:
            result = json.load(fp)
    except Exception:
        raise HTTPException(status_code=500, detail="結果データの読み込みに失敗しました。")

    return ScrapeResponse(**result)


@router.post("/scrape", response_model=ScrapeResponse, summary="シングルスターの商品ページをスクレイプする")
async def scrape(request: ScrapeRequest) -> ScrapeResponse:
    """
    シングルスターの商品グループURLを受け取り、全商品の情報を返す。

    1. ページ全体をスクレイピング（ページネーション対応）
    2. 各商品の英語名 + セットコードで Scryfall 検索してカード番号を補完（少量並列）
    3. 複数候補が残った場合は AI で絞り込む
    """
    return await _run_scrape(str(request.url))
