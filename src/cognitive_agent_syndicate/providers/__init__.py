"""Model provider abstractions."""

from cognitive_agent_syndicate.providers.base import GenerationResult, ModelProvider
from cognitive_agent_syndicate.providers.mock import (
    MockModelProvider,
    MockResponseNotConfiguredError,
)

__all__ = [
    "GenerationResult",
    "MockModelProvider",
    "MockResponseNotConfiguredError",
    "ModelProvider",
]
