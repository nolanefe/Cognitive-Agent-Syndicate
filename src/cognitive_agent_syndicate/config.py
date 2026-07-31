"""Application configuration loaded from environment variables."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognitive_agent_syndicate.paths import normalize_relative_posix_path


class Settings(BaseSettings):
    """Typed runtime configuration for the delivery pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = Field(default="mock", min_length=1, max_length=50)
    model: str = Field(default="mock-model", min_length=1, max_length=100)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_chars: int = Field(default=50_000, ge=100, le=500_000)
    max_generated_files: int = Field(default=20, ge=1, le=100)
    max_repair_attempts: int = Field(default=1, ge=0, le=1)
    artifact_output_dir: str = Field(default="generated_artifacts", min_length=1, max_length=260)
    api_key: str | None = Field(default=None)

    @field_validator("api_key", mode="before")
    @classmethod
    def normalize_api_key(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return value

    @field_validator("provider", "model", mode="before")
    @classmethod
    def strip_required_string(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("artifact_output_dir")
    @classmethod
    def validate_artifact_output_dir(cls, value: str) -> str:
        return normalize_relative_posix_path(value)
