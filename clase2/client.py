"""Reusable LLM client for the Clase 2 laboratory.

Select bedrock, anthropic, openai, google or ollama with LLM_PROVIDER in .env.
Importing this module never sends a request. Run it to try the success criterion.
"""

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


def call_bedrock(prompt, system=None, temperature=0.7, max_tokens=500):
    import boto3
    from botocore.config import Config

    # Keep retries in LLMClient instead of multiplying them with SDK retries.
    with boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(
            connect_timeout=10,
            read_timeout=60,
            retries={"total_max_attempts": 1},
        ),
    ) as client:
        request = {
            "modelId": os.environ.get(
                "BEDROCK_MODEL_ID_CLAUDE", "global.anthropic.claude-sonnet-4-6"
            ),
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        if system is not None:
            request["system"] = [{"text": system}]
        response = client.converse(**request)
    return "".join(
        part["text"]
        for part in response["output"]["message"]["content"]
        if "text" in part
    )


def call_anthropic(prompt, system=None, temperature=0.7, max_tokens=500):
    from anthropic import Anthropic

    with Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0, timeout=60
    ) as client:
        request = {
            "model": os.environ.get("ANTHROPIC_MODEL_ID", "claude-sonnet-4-6"),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # As in the notebook, omit temperature: some Claude models reject it.
        if system is not None:
            request["system"] = system
        response = client.messages.create(**request)
    return "".join(part.text for part in response.content if part.type == "text")


def call_openai(prompt, system=None, temperature=0.7, max_tokens=500):
    from openai import OpenAI

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    with OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=60
    ) as client:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_ID", "gpt-4o"),
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    return response.choices[0].message.content


def call_google(prompt, system=None, temperature=0.7, max_tokens=500):
    from google import genai
    from google.genai import types

    with genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options=types.HttpOptions(
            timeout=60_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        # A new chat makes each generate() independent of previous prompts.
        chat = client.chats.create(
            model=os.environ.get("GOOGLE_MODEL_ID", "gemini-3.5-flash-lite"),
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        response = chat.send_message(prompt)
    return response.text


def call_ollama(prompt, system=None, temperature=0.7, max_tokens=500):
    from ollama import Client

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    client = Client(host=os.environ.get("OLLAMA_HOST"), timeout=60)
    try:
        response = client.chat(
            model=os.environ.get("OLLAMA_MODEL_ID", "llama3.1"),
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response.message.content
    finally:
        client.close()


def _classify_error(error, provider):
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


class LLMClient:
    def __init__(self):
        load_dotenv()
        self.provider = os.environ.get("LLM_PROVIDER", "bedrock")
        credentials = {
            "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "ollama": None,
        }
        if self.provider not in credentials:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{self.provider}'. "
                f"Choose one of: {', '.join(credentials)}."
            )
        credential = credentials[self.provider]
        if credential and not os.environ.get(credential):
            raise ValueError(
                f"Missing {credential} for provider '{self.provider}'. "
                "Set it in your environment or .env file."
            )

    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Return text, with up to three attempts and 1s/2s retry delays."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        if system is not None and not isinstance(system, str):
            raise ValueError("system must be a string or None.")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be a finite, non-negative number.")

        provider_functions = {
            "bedrock": call_bedrock,
            "anthropic": call_anthropic,
            "openai": call_openai,
            "google": call_google,
            "ollama": call_ollama,
        }
        call_provider = provider_functions[self.provider]
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                text = call_provider(prompt, system, temperature, max_tokens)
            except ImportError as error:
                raise RuntimeError(
                    f"Missing SDK for '{self.provider}'. "
                    "Install dependencies with pip install -r requirements.txt."
                ) from error
            except Exception as error:
                category = _classify_error(error, self.provider)
                if category == "validation":
                    raise ValueError(
                        f"Invalid request for '{self.provider}': {error}"
                    ) from error
                if category != "retry":
                    raise
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Provider '{self.provider}' failed after "
                        f"{max_attempts} attempts due to rate limiting, "
                        "a timeout or a temporary service error."
                    ) from error
                delay = 2 ** attempt
                logger.warning(
                    "Temporary error from %s; retrying in %ss (attempt %s/%s).",
                    self.provider, delay, attempt + 2, max_attempts,
                )
                time.sleep(delay)
                continue

            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(
                    f"Provider '{self.provider}' returned no text. "
                    "Check content restrictions and the output token limit."
                )
            return text

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate independent answers with three workers, preserving order."""
        with ThreadPoolExecutor(max_workers=3) as pool:
            return list(pool.map(self.generate, prompts))


if __name__ == "__main__":
    try:
        print(LLMClient().generate("¿Cuál es la capital de Francia?"))
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
