import base64
import json
import re
import httpx
from openai import AsyncOpenAI

from app.config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)


class AIQuotaExceededError(Exception):
    """AI プロバイダーの利用上限超過を表す例外。"""


class AIProviderError(Exception):
    """AI プロバイダー側の一般的な失敗を表す例外。"""


GEMINI_API_VERSIONS = ["v1beta", "v1"]

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

            model_candidates = await _discover_gemini_models(client_http)
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
