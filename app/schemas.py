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
