"""Configuration management for the Psychometric Domain Mapping application.

Loads settings from environment variables and .env files. Provides typed,
validated access to all configuration with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Attempt to load .env — optional dependency
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path, override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Project root helper
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_FILENAME = "psychometric_role_domain_mapping_corrected_master.csv"
XLSX_FILENAME = "psychometric_role_domain_mapping_corrected_master.xlsx"


def _csv_list(raw: str) -> List[str]:
    """Split a comma-separated env var into a trimmed list of non-empty strings."""
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass(frozen=True)
class GroqConfig:
    """Groq API configuration — loaded from environment variables."""

    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    base_url: str = "https://api.groq.com/openai/v1"
    model_preference: List[str] = field(
        default_factory=lambda: _csv_list(
            os.getenv(
                "GROQ_MODEL_PREFERENCE",
                "llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-120b,openai/gpt-oss-20b",
            )
        )
    )
    fallback_models: List[str] = field(
        default_factory=lambda: _csv_list(
            os.getenv(
                "GROQ_FALLBACK_MODELS",
                "llama-3.1-8b-instant,openai/gpt-oss-20b",
            )
        )
    )
    catalog_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("GROQ_MODEL_CATALOG_TTL_SECONDS", "3600"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("GROQ_MAX_RETRIES", "2"))
    )
    max_fallbacks: int = field(
        default_factory=lambda: int(os.getenv("GROQ_MAX_FALLBACKS", "3"))
    )
    max_retry_wait_seconds: float = field(
        default_factory=lambda: float(os.getenv("GROQ_MAX_RETRY_WAIT_SECONDS", "8"))
    )
    timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
    )
    excluded_model_patterns: List[str] = field(
        default_factory=lambda: _csv_list(
            os.getenv(
                "GROQ_EXCLUDED_MODEL_PATTERNS",
                "whisper,orpheus,tts,prompt-guard,playai,distil-whisper,llama-guard,safeguard",
            )
        )
    )
    min_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("GROQ_MIN_CONTEXT_TOKENS", "4096"))
    )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def all_candidate_models(self) -> List[str]:
        """Ordered unique list: preferences first, then fallbacks."""
        seen: set[str] = set()
        result: List[str] = []
        for m in self.model_preference + self.fallback_models:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    groq: GroqConfig = field(default_factory=GroqConfig)
    data_dir: Path = DATA_DIR
    csv_path: Path = field(default_factory=lambda: DATA_DIR / CSV_FILENAME)
    xlsx_path: Path = field(default_factory=lambda: DATA_DIR / XLSX_FILENAME)
    default_top_n: int = 10
    max_top_n: int = 50


def get_config() -> AppConfig:
    """Factory that builds configuration from the current environment."""
    return AppConfig()
