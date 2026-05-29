"""Runtime configuration for the Alpha backend.

All credentials are optional. With none set, the stack still runs on REAL live
on-chain data (Subtensor RPC) for the core economic + identity signals; pillars that
need an external feed are honestly marked n/a until their key is provided.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Alpha"
    environment: str = "development"

    # Storage
    database_url: str = "sqlite:///./alpha.db"

    # CORS
    cors_origins: str = "*"

    # Data source credentials (all optional)
    taostats_api_key: str | None = None
    github_token: str | None = None
    huggingface_token: str | None = None
    x_bearer_token: str | None = None
    discord_bot_token: str | None = None
    subtensor_endpoint: str | None = None

    # AI layer
    openrouter_api_key: str | None = None
    anthropic_api_key: str | None = None
    oracle_model: str = "anthropic/claude-3.5-sonnet"

    # Telegram alerts (optional)
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    # Scan behaviour
    scan_interval_minutes: int = 15
    scan_on_startup: bool = True
    subnet_count: int = 128

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def has_llm(self) -> bool:
        return bool(self.openrouter_api_key or self.anthropic_api_key)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@lru_cache
def get_settings() -> Settings:
    return Settings()
