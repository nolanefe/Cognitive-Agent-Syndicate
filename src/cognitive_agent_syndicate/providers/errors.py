"""Typed provider exceptions with safe, concise messages."""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for model provider failures."""

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.request_id = request_id
        self.status_code = status_code
        super().__init__(message)


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is invalid or incomplete."""


class ProviderAuthenticationError(ProviderError):
    """Raised when provider authentication fails."""


class ProviderRateLimitError(ProviderError):
    """Raised when the provider rate-limits a request."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""


class ProviderConnectionError(ProviderError):
    """Raised when the provider cannot be reached."""


class ProviderRefusalError(ProviderError):
    """Raised when the model refuses to complete the request."""

    def __init__(
        self,
        message: str,
        *,
        refusal_category: str | None = None,
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.refusal_category = refusal_category
        super().__init__(message, request_id=request_id, status_code=status_code)


class ProviderCancelledResponseError(ProviderError):
    """Raised when the provider cancels a response."""


class ProviderFailedResponseError(ProviderError):
    """Raised when the provider reports a failed response."""


class ProviderIncompleteResponseError(ProviderError):
    """Raised when the provider returns an incomplete response."""

    def __init__(
        self,
        message: str,
        *,
        incomplete_reason: str | None = None,
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.incomplete_reason = incomplete_reason
        super().__init__(message, request_id=request_id, status_code=status_code)


class ProviderMalformedResponseError(ProviderError):
    """Raised when the provider response cannot be parsed into the expected type."""
