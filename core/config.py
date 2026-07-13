from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


class LLMProvider(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"
    AZURE = "azure"


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # LLM Provider
    # ------------------------------------------------------------------
    LLM_PROVIDER: LLMProvider = LLMProvider.GROQ

    # Groq  (https://console.groq.com — free tier)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    # Ollama  (fully local, no API key required)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # Azure OpenAI  (production path)
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4"
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "Main_dataset")))
    PROCESSED_DIR: Path = Path(os.getenv("PROCESSED_DIR", str(BASE_DIR / "Data Cleaning + EDA")))
    VECTOR_DB_DIR: Path = Path(os.getenv("VECTOR_DB_DIR", str(BASE_DIR / "vector_db")))

    # Anomaly detection artifacts (Milestone 2 — Engineer 2, Isolation Forest)
    # Anomaly detection artifacts
    ANOMALY_DIR: Path = Path(os.getenv("ANOMALY_DIR", str(BASE_DIR / "ml_core" / "artifacts"))) # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR}/mlflow.db")
    MLFLOW_EXPERIMENT_NAME: str = "olist_forecasting"

    # ------------------------------------------------------------------
    # Embeddings  (local HuggingFace model, no API key needed)
    # ------------------------------------------------------------------
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
