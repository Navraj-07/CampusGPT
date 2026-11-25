"""Configuration and environment loader for CampusGPT."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    """Application configuration with sensible local defaults."""

    BASE_DIR = BASE_DIR
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or os.getenv(
        "SECRET_KEY", "campusgpt-local-dev-secret-key-change-me"
    )

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

    DATA_DIR = BASE_DIR / "data"
    DATABASE_PATH = str(DATA_DIR / "campusgpt.db")
    UPLOAD_FOLDER = str(DATA_DIR / "uploads")
    VECTOR_STORE_FOLDER = str(DATA_DIR / "vector_stores")
    MODEL_CACHE_FOLDER = str(DATA_DIR / "model_cache")

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB
    ALLOWED_EXTENSIONS = {"pdf", "docx", "pptx", "txt", "md"}

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

    @classmethod
    def ensure_directories(cls) -> None:
        """Create required runtime directories if they do not exist."""
        for folder in (
            cls.DATA_DIR,
            Path(cls.UPLOAD_FOLDER),
            Path(cls.VECTOR_STORE_FOLDER),
            Path(cls.MODEL_CACHE_FOLDER),
        ):
            folder.mkdir(parents=True, exist_ok=True)
