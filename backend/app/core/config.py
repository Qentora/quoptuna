"""
Application configuration
"""

from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "QuOptuna Next"

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative frontend port
        "http://localhost:8000",  # Backend port
    ]

    # Database
    DATABASE_URL: str = "sqlite:///./quoptuna.db"

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

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
