"""Classify SDK failures for validation and retry handling."""

from enum import Enum


class ErrorCategory(Enum):
    VALIDATION = "validation"
    RETRY = "retry"
    FATAL = "fatal"


def classify_error(error, provider) -> ErrorCategory:
    """Return an ErrorCategory using SDK error details."""
    import httpx

    if isinstance(error, ValueError):
        return ErrorCategory.VALIDATION
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return ErrorCategory.RETRY

    if provider == "bedrock":
        from botocore.exceptions import (
            ClientError,
            ConnectTimeoutError,
            ParamValidationError,
            ReadTimeoutError,
        )

        if isinstance(error, ParamValidationError):
            return ErrorCategory.VALIDATION
        if isinstance(error, (ConnectTimeoutError, ReadTimeoutError)):
            return ErrorCategory.RETRY
        if isinstance(error, ClientError):
            error_code = error.response["Error"]["Code"]
            if error_code == "ValidationException":
                return ErrorCategory.VALIDATION
            if error_code == "ThrottlingException":
                return ErrorCategory.RETRY
    elif provider in ("openai", "anthropic"):
        if provider == "openai":
            from openai import APITimeoutError, RateLimitError
        else:
            from anthropic import APITimeoutError, RateLimitError
        if isinstance(error, (RateLimitError, APITimeoutError)):
            return ErrorCategory.RETRY

    # Google APIError.code and Ollama/OpenAI/Anthropic status_code are HTTP codes.
    status_code = getattr(error, "status_code", getattr(error, "code", None))
    if status_code in (400, 422):
        return ErrorCategory.VALIDATION
    if status_code in (408, 429, 500, 502, 503, 504):
        return ErrorCategory.RETRY
    return ErrorCategory.FATAL
