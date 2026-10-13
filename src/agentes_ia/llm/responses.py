"""Response formatting and provider-native conversation messages."""

def format_response(provider, response, latency):
    """Format SDK text and token usage; missing counts remain unknown."""
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

    return (
        f"Latency: {latency:.3f} s\n"
        f"Input tokens: {input_tokens if input_tokens is not None else 'N/A'}\n"
        f"Output tokens: {output_tokens if output_tokens is not None else 'N/A'}\n"
        f"Response: {text or '(sin texto)'}"
    )


def conversation_turn(provider, prompt, response):
    if provider == "google":
        candidates = response.candidates or []
        if not candidates or candidates[0].content is None:
            raise ValueError("Google returned no content to save in history.")
        return [{"role": "user", "parts": [{"text": prompt}]},
                candidates[0].content.model_dump(exclude_none=True)]
    if provider == "bedrock":
        return [{"role": "user", "content": [{"text": prompt}]},
                response["output"]["message"]]
    if provider == "anthropic":
        return [{"role": "user", "content": prompt},
                {"role": "assistant", "content": [part.model_dump(exclude_none=True)
                                                  for part in response.content]}]
    if provider == "openai":
        message = response.choices[0].message
        return [{"role": "user", "content": prompt},
                message.model_dump(exclude_none=True, exclude={"parsed", "annotations"})]
    return [{"role": "user", "content": prompt},
            response.message.model_dump(exclude_none=True)]
