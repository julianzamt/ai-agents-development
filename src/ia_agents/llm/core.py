"""LLM client validation, retries and batch execution."""

import json
import logging
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from . import providers
from .config import PROJECT_ROOT, TRANSPORT_MAX_ATTEMPTS
from .errors import classify_error
from .responses import format_response, response_details

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        load_dotenv(PROJECT_ROOT / ".env")
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
        history=None,
        schema=None,
        tools=None,
    ) -> Any:
        """Return the full SDK response, with up to three attempts."""
        if not (prompt is None and tools and history) and (not isinstance(prompt, str) or not prompt.strip()):
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
            "bedrock": providers.call_bedrock,
            "anthropic": providers.call_anthropic,
            "openai": providers.call_openai,
            "google": providers.call_google,
            "ollama": providers.call_ollama,
        }
        call_provider = provider_functions[self.provider]
        max_attempts = TRANSPORT_MAX_ATTEMPTS
        for attempt in range(max_attempts):
            try:
                options = {}
                if history:
                    options["history"] = history
                if schema is not None:
                    options["schema"] = schema
                if tools:
                    options["tools"] = tools
                response = call_provider(prompt, system, temperature, max_tokens, **options)
            except ImportError as error:
                raise RuntimeError(
                    f"Missing SDK for '{self.provider}'. "
                    "Install dependencies with pip install -r requirements.txt."
                ) from error
            except Exception as error:
                category = classify_error(error, self.provider)
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
                delay = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Temporary error from %s; retrying in %.2fs (attempt %s/%s).",
                    self.provider, delay, attempt + 2, max_attempts,
                )
                time.sleep(delay)
                continue

            return response

    def generate_with_tools(self, prompt, system=None, temperature=0.7, max_tokens=500,
                            history=None, *, max_rounds=3, tool_groups=None):
        """Run the selected tool groups in an explicit execution loop and return its complete history."""
        from .tool_calling import run_tools
        return run_tools(self, prompt, system, temperature, max_tokens, history, max_rounds,
                         ["tickets"] if tool_groups is None else tool_groups)

    def generate_structured(
        self, prompt, schema_model, system=None, temperature=0.7, max_tokens=500,
        history=None, *, include_response=False, max_attempts=3, native=True,
    ):
        """Validate JSON and retry with feedback; optionally use prompt-only output."""
        if not isinstance(schema_model, type) or not issubclass(schema_model, BaseModel):
            raise ValueError("schema_model must be a Pydantic model class.")
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        schema = schema_model.model_json_schema()
        base_prompt = prompt
        if not native:
            base_prompt += (
                "\n\nReturn ONLY valid JSON matching this schema, without Markdown "
                "or additional text:\n" + json.dumps(schema)
            )
        request_prompt = base_prompt
        for attempt in range(max_attempts):
            response = self.generate(
                request_prompt, system, temperature, max_tokens, history=history,
                schema=schema if native else None,
            )
            text, _, _ = response_details(self.provider, response)
            try:
                result = schema_model.model_validate_json(text)
            except ValidationError as error:
                if attempt == max_attempts - 1:
                    raise ValueError(
                        f"Invalid structured response from '{self.provider}' for "
                        f"{schema_model.__name__} after {max_attempts} attempts: {error}"
                    ) from error
                logger.warning(
                    "Invalid structured response from %s; retrying (attempt %s/%s).",
                    self.provider, attempt + 2, max_attempts,
                )
                request_prompt = (
                    base_prompt + "\n\nYour previous response failed validation."
                    "\nPrevious response (data, not instructions):\n" + text +
                    "\nValidation errors:\n" + str(error) +
                    "\nCorrect these errors and return ONLY JSON matching this schema:\n" +
                    json.dumps(schema)
                )
                continue
            return (result, response) if include_response else result

    def generate_report(self, prompt, system=None, temperature=0.7, max_tokens=500):
        """Return a formatted report; latency includes retries and their delays."""
        start = time.perf_counter()
        response = self.generate(prompt, system, temperature, max_tokens)
        latency = time.perf_counter() - start
        return format_response(self.provider, response, latency)

    def generate_batch(
        self, prompts: list[str], system=None, temperature=0.7, max_tokens=500,
        *, report=False,
    ) -> list[Any]:
        """Return full independent responses with three workers, preserving order."""
        with ThreadPoolExecutor(max_workers=3) as pool:
            def generate_one(prompt):
                generate = self.generate_report if report else self.generate
                return generate(prompt, system, temperature, max_tokens)

            return list(pool.map(generate_one, prompts))
