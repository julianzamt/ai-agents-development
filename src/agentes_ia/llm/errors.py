"""Classify SDK failures for validation and retry handling."""

def classify_error(error, provider):
    """Return a validation, retry or fatal category using SDK error details."""
    import httpx

    if isinstance(error, ValueError):
        return "validation"
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return "retry"

    if provider == "bedrock":
        from botocore.exceptions import (
            ClientError,
            ConnectTimeoutError,
            ParamValidationError,
            ReadTimeoutError,
        )

        if isinstance(error, ParamValidationError):
            return "validation"
        if isinstance(error, (ConnectTimeoutError, ReadTimeoutError)):
            return "retry"
        if isinstance(error, ClientError):
            error_code = error.response["Error"]["Code"]
            if error_code == "ValidationException":
                return "validation"
            if error_code == "ThrottlingException":
                return "retry"
    elif provider in ("openai", "anthropic"):
        if provider == "openai":
            from openai import APITimeoutError, RateLimitError
        else:
            from anthropic import APITimeoutError, RateLimitError
        if isinstance(error, (RateLimitError, APITimeoutError)):
            return "retry"

    # Google APIError.code and Ollama/OpenAI/Anthropic status_code are HTTP codes.
    status_code = getattr(error, "status_code", getattr(error, "code", None))
    if status_code in (400, 422):
        return "validation"
    if status_code in (408, 429, 500, 502, 503, 504):
        return "retry"
    return "fatal"
