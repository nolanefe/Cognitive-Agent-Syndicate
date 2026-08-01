"""Scoped credential and live-environment management."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from getpass import getpass

from pydantic import SecretStr

from cognitive_agent_syndicate.providers.errors import ProviderConfigurationError

_LIVE_ENV_VARS = ("RUN_LIVE_BENCHMARKS", "RUN_LIVE_TESTS")
_CREDENTIAL_ENV_VARS = ("OPENAI_API_KEY", "API_KEY")


@dataclass(frozen=True)
class CredentialEnvironmentSnapshot:
    """Captured credential-related environment state."""

    values: dict[str, str | None]


def snapshot_credential_environment() -> CredentialEnvironmentSnapshot:
    """Capture current credential and live opt-in environment variables."""
    return CredentialEnvironmentSnapshot(
        values={name: os.environ.get(name) for name in (*_CREDENTIAL_ENV_VARS, *_LIVE_ENV_VARS)}
    )


def restore_credential_environment(snapshot: CredentialEnvironmentSnapshot) -> None:
    """Restore credential environment to a prior snapshot exactly."""
    for name, prior in snapshot.values.items():
        if prior is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prior


def resolve_existing_api_key_from_env() -> SecretStr | None:
    """Return an existing API key from the environment without prompting."""
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        return SecretStr(openai_key)
    legacy_key = os.environ.get("API_KEY", "").strip()
    if legacy_key:
        return SecretStr(legacy_key)
    return None


def prompt_for_openai_api_key(
    *,
    prompt_fn: Callable[[str], str] | None = None,
) -> SecretStr:
    """Securely prompt for an OpenAI API key without echoing input."""
    reader = prompt_fn or getpass
    value = reader("OpenAI API key: ")
    stripped = value.strip()
    if not stripped:
        raise ProviderConfigurationError("OpenAI API key is required for live validation.")
    return SecretStr(stripped)


@contextmanager
def scoped_live_environment(
    *,
    prompt_if_missing: bool = True,
    prompt_fn: Callable[[str], str] | None = None,
) -> Iterator[SecretStr]:
    """Temporarily configure live credential and opt-in environment variables."""
    snapshot = snapshot_credential_environment()
    try:
        existing = resolve_existing_api_key_from_env()
        if existing is not None:
            api_key = existing
        elif prompt_if_missing:
            api_key = prompt_for_openai_api_key(prompt_fn=prompt_fn)
            os.environ["OPENAI_API_KEY"] = api_key.get_secret_value()
        else:
            raise ProviderConfigurationError("OpenAI provider requires a non-empty API key.")

        os.environ["RUN_LIVE_BENCHMARKS"] = "1"
        yield api_key
    finally:
        restore_credential_environment(snapshot)
