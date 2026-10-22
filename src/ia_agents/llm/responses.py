"""Response formatting and provider-native conversation messages."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class GenerationResult:
    """The same result shape for plain, tool and structured generation."""
    response: Any
    history: list
    structured: BaseModel | None
    latency: float
    report: str


def response_details(provider, response):
    """Extract text and token counts from a provider response."""
    def field(value, name, default=None):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    if provider == "bedrock":
        usage = field(response, "usage")
        input_tokens = field(usage, "inputTokens")
        output_tokens = field(usage, "outputTokens")
        content = field(field(field(response, "output"), "message"), "content", [])
        text = "".join(field(part, "text", "") for part in content)
    elif provider == "anthropic":
        usage = field(response, "usage")
        # Cached input is reported separately by Anthropic.
        counts = [field(usage, name) for name in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"
        )]
        input_tokens = sum(count or 0 for count in counts) if any(
            count is not None for count in counts
        ) else None
        output_tokens = field(usage, "output_tokens")
        text = "".join(field(part, "text", "") for part in
                       field(response, "content", []) if field(part, "type") == "text")
    elif provider == "openai":
        usage = field(response, "usage")
        input_tokens = field(usage, "prompt_tokens")
        output_tokens = field(usage, "completion_tokens")
        text = "\n".join(field(field(choice, "message"), "content") or ""
                         for choice in field(response, "choices", []))
    elif provider == "google":
        usage = field(response, "usage_metadata")
        input_tokens = field(usage, "prompt_token_count")
        output_tokens = field(usage, "candidates_token_count")
        text = "\n".join(
            "".join(field(part, "text") or "" for part in
                    field(field(candidate, "content"), "parts", [])
                    if not field(part, "thought", False))
            for candidate in field(response, "candidates", [])
        )
    elif provider == "ollama":
        input_tokens = field(response, "prompt_eval_count")
        output_tokens = field(response, "eval_count")
        text = field(field(response, "message"), "content") or ""
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return text, input_tokens, output_tokens


def format_response(provider, response, latency, *, structured=None):
    """Format SDK text and token usage; missing counts remain unknown."""
    text, input_tokens, output_tokens = response_details(provider, response)
    if structured is not None:
        text = structured.model_dump_json(indent=2)
    return (
        f"Latency: {latency:.3f} s\n"
        f"Input tokens: {input_tokens if input_tokens is not None else 'N/A'}\n"
        f"Output tokens: {output_tokens if output_tokens is not None else 'N/A'}\n"
        f"Response: {text or '(no text)'}"
    )


def conversation_turn(provider, prompt, response):
    """Return a standalone turn using the shared conversation formatter."""
    from .conversation import Conversation

    conversation = Conversation(provider)
    conversation.add_user_message(prompt)
    conversation.add_assistant_response(response)
    return conversation.messages
