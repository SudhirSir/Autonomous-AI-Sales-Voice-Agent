"""Application configuration and runtime settings."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration loaded from environment or .env file."""

    app_name: str = "Fortivex Autonomous AI Sales Voice Agent"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    fortivex_db_path: str = str(BASE_DIR / "data" / "fortivex.db")
    knowledge_dir: str = str(BASE_DIR / "knowledge_base")
    crm_webhook_url: str = ""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def get_runtime_gemini_key() -> str:
    """Return active Gemini API key from env or settings."""
    return os.environ.get("GEMINI_API_KEY", "").strip() or settings.gemini_api_key.strip()
