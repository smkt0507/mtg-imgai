from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ai_provider: str = "gemini"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    scryfall_interval_seconds: float = Field(0.15, ge=0.0)
    scrape_concurrency: int = Field(4, ge=1, le=16)
    ai_disambiguation_concurrency: int = Field(1, ge=1, le=4)
    scraper_page_concurrency: int = Field(6, ge=1, le=16)
    gemini_model_cache_ttl_seconds: int = Field(3600, ge=60)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
