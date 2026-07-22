"""
Application configuration
"""

from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "QuOptuna Next"

    # CORS
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative frontend port
        "http://localhost:8000",  # Backend port
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS_ORIGINS from comma-separated string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # Database
    DATABASE_URL: str = "sqlite:///./db/quoptuna_app.db"
    # Optuna may use the same database, or a separate one, in deployments.
    OPTUNA_DATABASE_URL: str = ""
    OPTUNA_DB_SCHEMA: str = "optuna"
    ARTIFACT_STORAGE: str = "local"
    ARTIFACT_ROOT: str = "db/analysis"
    S3_ENDPOINT_URL: str = ""
    S3_BUCKET: str = ""
    S3_REGION: str = ""
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: str = ""
    S3_PREFIX: str = "quoptuna"
    S3_SIGNED_URL_TTL: int = 900

    # File Upload
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB

    # Optimization
    DEFAULT_N_TRIALS: int = 100
    DEFAULT_TIMEOUT: int = 3600  # 1 hour

    # LLM Settings
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # Auth0 (auth is enforced only when these are configured)
    AUTH0_DOMAIN: str = ""
    AUTH0_CLIENT_ID: str = ""
    AUTH0_CLIENT_SECRET: str = ""
    AUTH0_SECRET: str = ""  # 64-char hex, generate with: openssl rand -hex 32
    APP_BASE_URL: str = "http://localhost:8000"

    @property
    def AUTH_ENABLED(self) -> bool:  # noqa: N802 - matches env-style settings naming
        return bool(
            self.AUTH0_DOMAIN
            and self.AUTH0_CLIENT_ID
            and self.AUTH0_CLIENT_SECRET
            and self.AUTH0_SECRET
        )

    class Config:
        env_file = ".env"
        case_sensitive = True
        # .env may hold keys for other tools (e.g. PORT); don't crash on them.
        extra = "ignore"


settings = Settings()
