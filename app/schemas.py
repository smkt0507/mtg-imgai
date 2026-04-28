from pydantic import BaseModel, HttpUrl, Field
from typing import Optional


class AnalyzeRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="解析対象のMTGカード画像URL")


class ScryfallCard(BaseModel):
    id: str
    name: str
    set: str
    collector_number: str
    scryfall_uri: str
    image_uris: Optional[dict] = None


class AnalyzeResponse(BaseModel):
    set_code: str = Field(..., description="セットコード (例: OTJ, MKM)")
    collector_number: str = Field(..., description="コレクター番号 (例: 123)")
    card_name: Optional[str] = Field(None, description="カード名")
    scryfall_uri: Optional[str] = Field(None, description="Scryfall カードページURL")
    scryfall_image_uri: Optional[str] = Field(None, description="Scryfall 正規画像URL")
    validated: bool = Field(..., description="Scryfallで検証済みかどうか")


class ErrorResponse(BaseModel):
    detail: str


# --- スクレイプ関連 ---

class ScrapeRequest(BaseModel):
    url: HttpUrl = Field(..., description="スクレイプ対象のシングルスター商品グループURL")


class CardItem(BaseModel):
    raw_name: str = Field(..., description="元の商品名テキスト")
    card_name_ja: Optional[str] = Field(None, description="日本語カード名")
    card_name_en: Optional[str] = Field(None, description="英語カード名")
    lang: Optional[str] = Field(None, description="言語 (jp / en)")
    foil: bool = Field(..., description="FOILかどうか")
    set_code: Optional[str] = Field(None, description="セットコード")
    card_number: Optional[str] = Field(None, description="コレクター番号")
    price: Optional[int] = Field(None, description="価格（円）")
    disambiguated_by_ai: bool = Field(False, description="AIで番号を特定したか")


class ScrapeResponse(BaseModel):
    total: int = Field(..., description="取得件数")
    items: list[CardItem]


class ScrapeStartResponse(BaseModel):
    job_id: str = Field(..., description="進捗確認に使うジョブID")


class ScrapeStatusResponse(BaseModel):
    job_id: str = Field(..., description="ジョブID")
    state: str = Field(..., description="queued/running/completed/failed")
    stage: str = Field(..., description="現在の処理段階")
    processed: int = Field(..., description="処理済み件数")
    total: int = Field(..., description="全体件数")
    elapsed_seconds: int = Field(..., description="経過秒数")
    error: Optional[str] = Field(None, description="失敗時のエラーメッセージ")
