"""Configuration settings for CacheSense."""


from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core Cache Settings
    SIMILARITY_THRESHOLD: float = 0.92
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    FAISS_INDEX_PATH: str = "data/faiss.index"
    SQLITE_DB_PATH: str = "data/cache.db"
    MAX_CACHE_ENTRIES: int = 50000
    DEFAULT_TTL_SECONDS: int = 86400  # 24 hours
    MAX_CACHEABLE_TEMPERATURE: float = 0.3
    CACHE_NAMESPACE_BY_KEY: bool = False
    SINGLE_FLIGHT_ENABLED: bool = True

    # PII & Privacy Filter
    CACHE_PII_FILTER_ENABLED: bool = True
    CACHE_EXCLUDE_REGEX_PATTERNS: list[str] = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",  # Email
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b",  # Credit Card
        r"\bsk-[a-zA-Z0-9]{32,}\b",  # OpenAI API key pattern
        r"\bAKIA[0-9A-Z]{16}\b",  # AWS Access Key
    ]
    LOG_RAW_PROMPTS: bool = False

    # Concurrency & Performance
    EMBEDDING_THREADS: int = 4
    EMBEDDING_BATCH_SIZE: int = 32
    TTL_SWEEP_INTERVAL_SECONDS: int = 60
    SNAPSHOT_INTERVAL_SECONDS: int = 300

    # Upstream Provider Configuration
    UPSTREAM_OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    UPSTREAM_ANTHROPIC_BASE_URL: str = "https://api.anthropic.com/v1"
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    UPSTREAM_TIMEOUT_SECONDS: float = 60.0
    UPSTREAM_MAX_RETRIES: int = 2

    # Mock Upstream (for tests & zero-cost benchmarks)
    MOCK_UPSTREAM_ENABLED: bool = False
    MOCK_UPSTREAM_BASE_URL: str = "http://127.0.0.1:8080"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
