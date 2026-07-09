"""
Application configuration — all values come from environment variables.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── App ────────────────────────────────────────────────────────────────────
    APP_NAME: str = "SignAI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ── Security ───────────────────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE_THIS_IN_PRODUCTION_USE_A_LONG_RANDOM_STRING"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── Database  ──────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/signai"

    # ── Redis ─────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"

    # ── ML Model ───────────────────────────────────────────────────────────────
    MODEL_PATH: str = "ml_training/saved_models/sign_model_v1.h5"
    CONFIDENCE_THRESHOLD: float = 0.70
    MAX_SEQUENCE_LENGTH: int = 120   # 4 seconds at 30fps — hard cap
    MIN_SEQUENCE_LENGTH: int = 12    # 0.4 seconds at 30fps — noise floor
    MOTION_THRESHOLD: float = 0.035  # tune against your webcam if needed
    NUM_LANDMARKS: int = 21
    NUM_FEATURES: int = 63  # 21 landmarks × 3 (x, y, z)

    # ── CORS ───────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Production build preview
    ]

    # ── Translation ──────────────────────────────────────────────────
    GOOGLE_TRANSLATE_API_KEY: str = ""
    DEFAULT_LANGUAGE: str = "en"
    SUPPORTED_LANGUAGES: list[str] = ["en", "ta", "hi", "es", "fr"]


# Singleton — import this everywhere
settings = Settings()
