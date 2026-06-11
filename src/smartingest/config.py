"""Application configuration.

Settings are loaded from environment variables (and an optional ``.env`` file)
using ``pydantic-settings`` so that every value is typed and validated at
startup rather than failing deep inside a request.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Gemini ---
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    # --- LangSmith ---
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "smartingest"

    # --- Application paths ---
    smartingest_db_path: str = "data/jobs.db"
    smartingest_upload_dir: str = "data/uploads"
    smartingest_rules_path: str = "config/rules.yaml"

    # --- Pipeline behaviour ---
    # Default to mock mode so the project runs end-to-end with no API key.
    smartingest_mock_llm: bool = True
    smartingest_min_confidence: float = 0.75
    smartingest_max_retries: int = 2

    # --- Frontend ---
    smartingest_api_url: str = "http://localhost:8000"

    @property
    def use_mock_llm(self) -> bool:
        """Whether to use deterministic stub responses instead of Gemini.

        Mock mode is forced on when no API key is configured so the pipeline
        never hard-fails purely for lack of credentials.
        """
        return self.smartingest_mock_llm or not self.gemini_api_key


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
