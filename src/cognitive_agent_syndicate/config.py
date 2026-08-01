"""Application configuration loaded from environment variables."""

from enum import StrEnum

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognitive_agent_syndicate.paths import normalize_relative_posix_path


class ProviderName(StrEnum):
    """Supported model provider identifiers."""

    MOCK = "mock"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Typed runtime configuration for the delivery pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    provider: ProviderName = Field(default=ProviderName.MOCK)
    model: str = Field(default="mock-model", min_length=1, max_length=100)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_chars: int = Field(default=50_000, ge=100, le=500_000)
    max_generated_files: int = Field(default=20, ge=1, le=100)
    max_repair_attempts: int = Field(default=1, ge=0, le=1)
    artifact_output_dir: str = Field(default="generated_artifacts", min_length=1, max_length=260)
    # Legacy API key setting kept for backward compatibility.
    api_key: SecretStr | None = Field(default=None)
    # Preferred OpenAI key; takes precedence over ``api_key`` when both are set.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )
    request_timeout: float = Field(default=60.0, ge=1.0, le=600.0)
    max_output_tokens: int = Field(default=4096, ge=1, le=100_000)

    @field_validator("api_key", "openai_api_key", mode="before")
    @classmethod
    def normalize_api_key(cls, value: object) -> SecretStr | None:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            raw = value.get_secret_value().strip()
            return SecretStr(raw) if raw else None
        if isinstance(value, str):
            stripped = value.strip()
            return SecretStr(stripped) if stripped else None
        raise ValueError("API key must be a string.")

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: object) -> object:
        if isinstance(value, ProviderName):
            return value
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("model", mode="before")
    @classmethod
    def strip_required_string(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("artifact_output_dir")
    @classmethod
    def validate_artifact_output_dir(cls, value: str) -> str:
        return normalize_relative_posix_path(value)

    def resolved_openai_api_key(self) -> SecretStr | None:
        """Return the configured OpenAI API key with deterministic precedence.

        ``OPENAI_API_KEY`` (``openai_api_key``) takes precedence over the legacy
        ``API_KEY`` (``api_key``) setting when both are configured.
        """
        return self.openai_api_key or self.api_key


def build_settings(**overrides: object) -> Settings:
    """Construct settings without loading a local .env file."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg, arg-type]


def apply_settings_overrides(
    base: Settings | None = None,
    **overrides: object,
) -> Settings:
    """Construct settings while preserving resolved OpenAI credentials across overrides."""
    merged: dict[str, object] = dict(overrides)
    if "openai_api_key" not in merged and "api_key" not in merged:
        source = base if base is not None else build_settings()
        resolved = source.resolved_openai_api_key()
        if resolved is not None:
            merged["openai_api_key"] = resolved.get_secret_value()
    return build_settings(**merged)
