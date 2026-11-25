"""Application configuration and runtime settings."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env file."""

    app_name: str = "Fortivex Autonomous AI Sales Voice Agent"

    # LLM & Embeddings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "text-embedding-004"

    # Pluggable Voice Providers (STT & TTS)
    stt_provider: str = "auto"  # auto | gemini | deepgram | whisper | browser
    tts_provider: str = "auto"  # auto | elevenlabs | google | local-wav | browser
    deepgram_api_key: str = ""
    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    google_tts_api_key: str = ""

    # Storage & Knowledge Base
    fortivex_db_path: str = str(BASE_DIR / "data" / "fortivex.db")
    knowledge_dir: str = str(BASE_DIR / "knowledge_base")
    backup_dir: str = str(BASE_DIR / "data" / "backups")

    # External CRM Integrations
    crm_webhook_url: str = ""
    crm_webhook_secret: str = ""
    hubspot_access_token: str = ""
    salesforce_instance_url: str = ""
    salesforce_access_token: str = ""
    crm_mock_mode: bool = False

    # Business Hours (UTC)
    business_hours_start: int = 9
    business_hours_end: int = 18

    # Security & Reliability
    fortivex_api_token: str = ""
    cors_origins: str = "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8765"
    rate_limit_rpm: int = 120
    pii_masking_enabled: bool = False

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def get_runtime_secret(env_name: str, default_val: str = "") -> str:
    """Return secret dynamically from environment overrides or settings."""
    return os.environ.get(env_name, "").strip() or default_val.strip()


def get_runtime_gemini_key() -> str:
    """Return active Gemini API key from env or settings."""
    return get_runtime_secret("GEMINI_API_KEY", settings.gemini_api_key)
