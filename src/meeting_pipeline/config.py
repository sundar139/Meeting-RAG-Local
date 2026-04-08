from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "meeting-pipeline"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = False

    data_root: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    interim_data_dir: Path = Path("data/interim")
    processed_data_dir: Path = Path("data/processed")
    eval_data_dir: Path = Path("data/eval")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "meeting_pipeline"
    postgres_user: str = "meeting_user"
    postgres_password: SecretStr = Field(default=SecretStr("change-me"))
    pgvector_collection: str = "meeting_chunks"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text-v2-moe"
    ollama_chat_model: str = "llama3.2:3b-instruct"

    whisperx_model: str = "large-v3"
    whisperx_device: Literal["cuda", "cpu"] = "cuda"
    whisperx_compute_type: str = "float16"
    whisperx_batch_size: int = 8

    pyannote_auth_token: SecretStr = Field(default=SecretStr(""))
    huggingface_token: SecretStr = Field(default=SecretStr(""))

    streamlit_server_port: int = 8501

    @property
    def postgres_dsn(self) -> str:
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
