import json
import re
from openai import AsyncOpenAI

from app.config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)


class AIQuotaExceededError(Exception):
    """AI プロバイダーの利用上限超過を表す例外。"""


class AIProviderError(Exception):
    """AI プロバイダー側の一般的な失敗を表す例外。"""

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


async def analyze_card_image(image_url: str) -> dict:
    """
    GPT-4o Vision を使用して MTG カード画像からセットコードとコレクター番号を抽出する。
    """
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

    # JSON ブロックのみ抽出（```json ... ``` 形式も対応）
    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"AI response did not contain valid JSON: {raw!r}")

    return json.loads(json_match.group())
