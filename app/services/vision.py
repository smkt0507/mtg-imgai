import asyncio
import base64
import json
import re
import time
import httpx
from openai import AsyncOpenAI

from app.config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)


class AIQuotaExceededError(Exception):
    """AI プロバイダーの利用上限超過を表す例外。"""


class AIProviderError(Exception):
    """AI プロバイダー側の一般的な失敗を表す例外。"""


GEMINI_API_VERSIONS = ["v1beta", "v1"]
_GEMINI_MODELS_CACHE: list[tuple[str, str]] = []
_GEMINI_MODELS_CACHE_EXPIRES_AT = 0.0
_GEMINI_MODELS_CACHE_LOCK = asyncio.Lock()

SYSTEM_PROMPT = """You are an expert Magic: The Gathering card identifier.
When given an image of an MTG card, extract the set code and collector number printed at the bottom of the card.
The set code is typically a 3-5 character uppercase abbreviation (e.g., OTJ, MKM, BRO, ONE).
The collector number is the number printed at the bottom (e.g., 123, 45, 300/384).
Also extract the card name printed at the top.

Respond ONLY with valid JSON in this exact format, no explanation:
{
  "set_code": "XXX",
  "collector_number": "123",
  "card_name": "Card Name Here"
}

If you cannot determine a value, use null for that field.
"""


def _extract_json(raw: str) -> dict:
    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"AI response did not contain valid JSON: {raw!r}")
    return json.loads(json_match.group())


def _normalize_model_name(name: str) -> str:
    if name.startswith("models/"):
        return name.split("models/", 1)[1]
    return name


async def _discover_gemini_models(client_http: httpx.AsyncClient) -> list[tuple[str, str]]:
    """
    利用可能な Gemini モデルを API から取得し、(api_version, model_name) の候補一覧を返す。
    generateContent 対応モデルのみを対象にする。
    """
    discovered: list[tuple[str, str]] = []

    for api_version in GEMINI_API_VERSIONS:
        endpoint = f"https://generativelanguage.googleapis.com/{api_version}/models"
        resp = await client_http.get(endpoint, params={"key": settings.gemini_api_key})
        if resp.status_code == 404:
            continue
        resp.raise_for_status()

        models = resp.json().get("models", [])
        for model in models:
            methods = model.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue

            raw_name = model.get("name", "")
            if not raw_name:
                continue

            model_name = _normalize_model_name(raw_name)
            if not model_name.startswith("gemini"):
                continue

            discovered.append((api_version, model_name))

    # 速度優先で flash 系を先に試す
    discovered.sort(key=lambda x: ("flash" not in x[1], x[1]))
    return discovered


async def _get_gemini_model_candidates(client_http: httpx.AsyncClient) -> list[tuple[str, str]]:
    global _GEMINI_MODELS_CACHE
    global _GEMINI_MODELS_CACHE_EXPIRES_AT

    now = time.monotonic()
    if _GEMINI_MODELS_CACHE and now < _GEMINI_MODELS_CACHE_EXPIRES_AT:
        return list(_GEMINI_MODELS_CACHE)

    async with _GEMINI_MODELS_CACHE_LOCK:
        now = time.monotonic()
        if _GEMINI_MODELS_CACHE and now < _GEMINI_MODELS_CACHE_EXPIRES_AT:
            return list(_GEMINI_MODELS_CACHE)

        discovered = await _discover_gemini_models(client_http)
        _GEMINI_MODELS_CACHE = discovered
        _GEMINI_MODELS_CACHE_EXPIRES_AT = now + settings.gemini_model_cache_ttl_seconds
        return list(discovered)


async def _analyze_with_openai(image_url: str) -> dict:
    if not settings.openai_api_key:
        raise AIProviderError("OPENAI_API_KEY が未設定です。")

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Identify this MTG card and return only the JSON.",
                        },
                    ],
                },
            ],
            max_tokens=256,
            temperature=0,
        )
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "Error code: 429" in msg:
            raise AIQuotaExceededError(
                "OpenAI のクォータ上限に達しました。Billing を有効化するか、別の API キーを設定してください。"
            ) from e
        raise AIProviderError(msg) from e

    raw = response.choices[0].message.content or ""
    return _extract_json(raw)


async def _analyze_with_gemini(image_url: str) -> dict:
    if not settings.gemini_api_key:
        raise AIProviderError("GEMINI_API_KEY が未設定です。")

    last_error = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client_http:
            image_resp = await client_http.get(image_url)
            image_resp.raise_for_status()
            mime_type = image_resp.headers.get("content-type", "image/jpeg").split(";")[0]
            image_b64 = base64.b64encode(image_resp.content).decode("ascii")

            model_candidates = await _get_gemini_model_candidates(client_http)
            if not model_candidates:
                raise AIProviderError(
                    "Gemini の利用可能モデル取得に失敗しました。"
                    " APIキーとGenerative Language APIの有効化を確認してください。"
                )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": SYSTEM_PROMPT + "\nReturn only JSON."},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image_b64,
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 256,
                },
            }

            for api_version, model_name in model_candidates:
                endpoint = (
                    f"https://generativelanguage.googleapis.com/{api_version}/models/"
                    f"{model_name}:generateContent"
                )
                gemini_resp = await client_http.post(
                    endpoint,
                    params={"key": settings.gemini_api_key},
                    json=payload,
                )

                if gemini_resp.status_code == 429:
                    raise AIQuotaExceededError(
                        "Gemini の無料枠上限に達しました。時間を置いて再試行してください。"
                    )
                if gemini_resp.status_code == 404:
                    last_error = f"model not found: {model_name} on {api_version}"
                    continue

                gemini_resp.raise_for_status()
                payload_resp = gemini_resp.json()
                candidates = payload_resp.get("candidates") or []
                if not candidates:
                    raise ValueError(f"Gemini response is empty: {payload_resp}")

                parts = candidates[0].get("content", {}).get("parts", [])
                text = "\n".join([p.get("text", "") for p in parts if p.get("text")]).strip()
                return _extract_json(text)

            raise AIProviderError(
                "Gemini の利用可能モデルが見つかりませんでした。"
                f" 最終エラー: {last_error or 'unknown'}"
            )
    except AIQuotaExceededError:
        raise
    except Exception as e:
        raise AIProviderError(str(e)) from e


async def analyze_card_image(image_url: str) -> dict:
    """
    MTG カード画像からセットコードとコレクター番号を抽出する。
    """
    provider = settings.ai_provider.lower().strip()
    if provider == "openai":
        return await _analyze_with_openai(image_url)
    if provider == "gemini":
        return await _analyze_with_gemini(image_url)
    raise AIProviderError("AI_PROVIDER は 'gemini' か 'openai' を指定してください。")


DISAMBIGUATE_PROMPT = """You are an expert Magic: The Gathering card identifier.

Given a product title from a Japanese card shop and a list of Scryfall card candidates,
choose the ONE candidate whose collector_number best matches the product.

Use these clues:
- frame_effects: e.g. "showcase", "extendedart", "borderless" — these usually mean higher collector numbers
- border_color: "borderless" means a borderless treatment
- full_art: true means full-art basic land or special treatment
- promo: true means promo card
- If the product title contains words like ショーケース, ボーダーレス, エクステンデッドアート, then prefer those frame effects
- If no special treatment is mentioned in the title, prefer the plain/lowest collector number

Respond ONLY with valid JSON:
{"collector_number": "123"}

If you truly cannot decide, return {"collector_number": null}
"""


def _candidate_sort_key(candidate: dict) -> str:
    return str(candidate.get("collector_number") or "").zfill(10)


def _is_special_treatment(candidate: dict) -> bool:
    return bool(
        candidate.get("frame_effects")
        or candidate.get("border_color") == "borderless"
        or candidate.get("full_art")
        or candidate.get("promo")
    )


def _score_candidate_against_title(raw_name: str, candidate: dict) -> tuple[int, bool]:
    title = raw_name.lower()
    frame_effects = {str(effect).lower() for effect in candidate.get("frame_effects", [])}
    score = 0
    matched_keyword = False

    if any(keyword in raw_name for keyword in ("ショーケース",)) or "showcase" in title:
        matched_keyword = True
        if "showcase" in frame_effects:
            score += 4

    if any(keyword in raw_name for keyword in ("ボーダーレス",)) or "borderless" in title:
        matched_keyword = True
        if candidate.get("border_color") == "borderless" or "borderless" in frame_effects:
            score += 4

    if (
        any(keyword in raw_name for keyword in ("エクステンデッドアート", "拡張アート"))
        or "extended art" in title
        or "extendedart" in title
    ):
        matched_keyword = True
        if "extendedart" in frame_effects:
            score += 4

    if any(keyword in raw_name for keyword in ("フルアート",)) or "full art" in title:
        matched_keyword = True
        if candidate.get("full_art"):
            score += 3

    if any(keyword in raw_name for keyword in ("プロモ", "PROMO")) or "promo" in title:
        matched_keyword = True
        if candidate.get("promo"):
            score += 3

    return score, matched_keyword


def _choose_candidate_without_ai(raw_name: str, candidates: list[dict]) -> str | None:
    scored: list[tuple[int, dict]] = []
    saw_keyword = False
    for candidate in candidates:
        score, matched_keyword = _score_candidate_against_title(raw_name, candidate)
        if matched_keyword:
            saw_keyword = True
        scored.append((score, candidate))

    if saw_keyword:
        best_score = max(score for score, _ in scored)
        if best_score > 0:
            best = [candidate for score, candidate in scored if score == best_score]
            if len(best) == 1:
                return best[0]["collector_number"]
        return None

    plain_candidates = [candidate for candidate in candidates if not _is_special_treatment(candidate)]
    if len(plain_candidates) == 1:
        return plain_candidates[0]["collector_number"]

    return None


async def disambiguate_card_number(
    raw_name: str,
    candidates: list[dict],
    client_http: httpx.AsyncClient | None = None,
) -> tuple[str | None, bool]:
    """
    商品名と Scryfall 候補リストから最適なコレクター番号を返す。
    戻り値は (collector_number, used_ai)。
    """
    if not candidates:
        return None, False
    if len(candidates) == 1:
        return candidates[0]["collector_number"], False

    heuristic_number = _choose_candidate_without_ai(raw_name, candidates)
    if heuristic_number is not None:
        return heuristic_number, False

    prompt_user = (
        f"Product title: {raw_name}\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
        "Which collector_number matches this product?"
    )

    provider = settings.ai_provider.lower().strip()
    try:
        if provider == "openai":
            result = await _disambiguate_openai(prompt_user)
        else:
            result = await _disambiguate_gemini(prompt_user, client_http=client_http)
    except Exception:
        # AI が失敗したら最小番号を返す（フォールバック）
        nums = sorted(candidates, key=_candidate_sort_key)
        return nums[0]["collector_number"], False

    return result.get("collector_number"), True


async def _disambiguate_openai(prompt_user: str) -> dict:
    if not settings.openai_api_key:
        raise AIProviderError("OPENAI_API_KEY が未設定です。")
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": DISAMBIGUATE_PROMPT},
            {"role": "user", "content": prompt_user},
        ],
        max_tokens=64,
        temperature=0,
    )
    return _extract_json(response.choices[0].message.content or "")


async def _disambiguate_gemini(prompt_user: str, client_http: httpx.AsyncClient | None = None) -> dict:
    if not settings.gemini_api_key:
        raise AIProviderError("GEMINI_API_KEY が未設定です。")

    if client_http is not None:
        return await _disambiguate_gemini_with_client(client_http, prompt_user)

    async with httpx.AsyncClient(timeout=30.0) as local_client:
        return await _disambiguate_gemini_with_client(local_client, prompt_user)


async def _disambiguate_gemini_with_client(client_http: httpx.AsyncClient, prompt_user: str) -> dict:
    model_candidates = await _get_gemini_model_candidates(client_http)
    if not model_candidates:
        raise AIProviderError("Gemini の利用可能モデルが見つかりませんでした。")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": DISAMBIGUATE_PROMPT + "\n\n" + prompt_user},
                ]
            }
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 64},
    }

    for api_version, model_name in model_candidates:
        endpoint = (
            f"https://generativelanguage.googleapis.com/{api_version}/models/"
            f"{model_name}:generateContent"
        )
        resp = await client_http.post(
            endpoint,
            params={"key": settings.gemini_api_key},
            json=payload,
        )
        if resp.status_code in (404, 429):
            continue
        resp.raise_for_status()
        parts = resp.json()["candidates"][0]["content"]["parts"]
        text = "\n".join(p.get("text", "") for p in parts).strip()
        return _extract_json(text)

    raise AIProviderError("Gemini の呼び出しに失敗しました。")
