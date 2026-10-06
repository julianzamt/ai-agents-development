"""Offline checks for the laboratory's success criteria and retry behavior."""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import httpx2
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import RateLimitError as AnthropicRateLimitError
from botocore.exceptions import ClientError, ReadTimeoutError
from google.genai.errors import ClientError as GoogleClientError
from ollama import ResponseError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import RateLimitError as OpenAIRateLimitError

from clase2 import client


PROVIDERS = ("bedrock", "anthropic", "openai", "google", "ollama")
TEST_ENVIRONMENT = {
    "LLM_PROVIDER": "google",
    "GOOGLE_API_KEY": "test-placeholder",
    "ANTHROPIC_API_KEY": "test-placeholder",
    "OPENAI_API_KEY": "test-placeholder",
    "AWS_BEARER_TOKEN_BEDROCK": "test-placeholder",
}


def sdk_error(provider, status):
    if provider == "bedrock":
        code = {429: "ThrottlingException", 400: "ValidationException"}.get(
            status, "AccessDeniedException"
        )
        return ClientError({"Error": {"Code": code, "Message": "test error"}}, "Converse")
    if provider == "google":
        return GoogleClientError(status, {"error": {"message": "test error"}})
    if provider == "ollama":
        return ResponseError("test error", status)
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://example.com"))
    if provider == "openai":
        error_class = OpenAIRateLimitError if status == 429 else OpenAIAPIStatusError
    else:
        error_class = AnthropicRateLimitError if status == 429 else AnthropicAPIStatusError
    return error_class("test error", response=response, body=None)


class ClientTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, TEST_ENVIRONMENT, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        dotenv = patch.object(client, "load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)
        sleep = patch.object(client.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        logging = patch.object(client.logger, "warning")
        logging.start()
        self.addCleanup(logging.stop)

    def select(self, provider):
        os.environ["LLM_PROVIDER"] = provider
        return client.LLMClient()

    def test_each_provider_returns_only_text_and_forwards_parameters(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                with patch.object(client, f"call_{provider}", return_value="Paris") as generate:
                    result = self.select(provider).generate("Capital?", "Be concise", 0, 123)
                    self.assertEqual(result, "Paris")
                    generate.assert_called_once_with("Capital?", "Be concise", 0, 123)

    def test_invalid_max_tokens_never_calls_provider(self):
        with patch.object(client, "call_google") as generate:
            for max_tokens in (-10, 0, 1.5, True, "500", None):
                with self.subTest(max_tokens=max_tokens):
                    with self.assertRaisesRegex(ValueError, "max_tokens.*positive integer"):
                        client.LLMClient().generate("Hello", max_tokens=max_tokens)
            generate.assert_not_called()
            self.sleep.assert_not_called()

    def test_invalid_prompt_system_and_temperature_never_call_provider(self):
        invalid_arguments = [
            {"prompt": ""}, {"prompt": "   "}, {"prompt": None},
            {"system": 123}, {"temperature": -1}, {"temperature": float("nan")},
            {"temperature": float("inf")}, {"temperature": True}, {"temperature": "0"},
        ]
        with patch.object(client, "call_google") as generate:
            for arguments in invalid_arguments:
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    client.LLMClient().generate(**({"prompt": "Hello"} | arguments))
            generate.assert_not_called()

    def test_each_sdk_rate_limit_retries_with_exponential_backoff(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.sleep.reset_mock()
                error = sdk_error(provider, 429)
                with patch.object(
                    client, f"call_{provider}", side_effect=[error, error, "Paris"]
                ) as generate:
                    self.assertEqual(self.select(provider).generate("Capital?"), "Paris")
                    self.assertEqual(generate.call_count, 3)
                    self.assertEqual(self.sleep.call_args_list, [call(1), call(2)])

    def test_exhausted_retries_are_bounded_and_preserve_cause(self):
        error = sdk_error("google", 429)
        with patch.object(client, "call_google", side_effect=error) as generate:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts") as raised:
                client.LLMClient().generate("Hello")
            self.assertIs(raised.exception.__cause__, error)
            self.assertEqual(generate.call_count, 3)
            self.assertEqual(self.sleep.call_args_list, [call(1), call(2)])

    def test_each_sdk_validation_error_is_clear_and_not_retried(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                error = sdk_error(provider, 400)
                with patch.object(client, f"call_{provider}", side_effect=error) as generate:
                    with self.assertRaisesRegex(ValueError, "Invalid request") as raised:
                        self.select(provider).generate("Hello")
                    self.assertIs(raised.exception.__cause__, error)
                    generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_permission_errors_are_propagated_without_retry(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                error = sdk_error(provider, 403)
                with patch.object(client, f"call_{provider}", side_effect=error) as generate:
                    with self.assertRaises(type(error)) as raised:
                        self.select(provider).generate("Hello")
                    self.assertIs(raised.exception, error)
                    generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_network_timeouts_retry(self):
        request = httpx2.Request("POST", "https://example.com")
        errors = {
            "bedrock": ReadTimeoutError(endpoint_url="https://example.com"),
            "google": TimeoutError("test timeout"),
            "openai": OpenAIAPITimeoutError(request),
            "anthropic": AnthropicAPITimeoutError(request),
        }
        for provider, error in errors.items():
            with self.subTest(provider=provider):
                self.sleep.reset_mock()
                with patch.object(client, f"call_{provider}", side_effect=[error, "Paris"]):
                    self.assertEqual(self.select(provider).generate("Capital?"), "Paris")
                    self.sleep.assert_called_once_with(1)

    def test_missing_credentials_and_unsupported_provider_are_clear(self):
        os.environ.pop("GOOGLE_API_KEY")
        with self.assertRaisesRegex(ValueError, "Missing GOOGLE_API_KEY"):
            client.LLMClient()
        os.environ["LLM_PROVIDER"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported LLM_PROVIDER"):
            client.LLMClient()

    def test_default_is_bedrock_and_ollama_needs_no_credentials(self):
        os.environ.pop("LLM_PROVIDER")
        self.assertEqual(client.LLMClient().provider, "bedrock")
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            self.assertEqual(client.LLMClient().provider, "ollama")

    def test_empty_response_is_explained_without_retry(self):
        for text in (None, "", "  "):
            with self.subTest(text=text):
                with patch.object(client, "call_google", return_value=text) as generate:
                    with self.assertRaisesRegex(RuntimeError, "returned no text"):
                        client.LLMClient().generate("Hello")
                    generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_batch_runs_concurrently_and_preserves_order(self):
        barrier = threading.Barrier(3)

        def answer(prompt, system, temperature, max_tokens):
            barrier.wait(timeout=5)
            return prompt.upper()

        prompts = [f"question {index}" for index in range(6)]
        with patch.object(client, "call_google", side_effect=answer):
            self.assertEqual(client.LLMClient().generate_batch(prompts), [p.upper() for p in prompts])
        self.assertEqual(client.LLMClient().generate_batch([]), [])

    def test_google_adapter_uses_independent_chats_and_text(self):
        with patch("google.genai.Client") as constructor:
            sdk = constructor.return_value.__enter__.return_value
            sdk.chats.create.return_value.send_message.return_value.text = "Paris"
            for _ in range(2):
                self.assertEqual(client.call_google("Capital?", "Be concise", 0, 123), "Paris")
            self.assertEqual(sdk.chats.create.call_count, 2)
            config = sdk.chats.create.call_args.kwargs["config"]
            self.assertEqual(config.system_instruction, "Be concise")
            self.assertEqual(config.temperature, 0)
            self.assertEqual(config.max_output_tokens, 123)
            self.assertEqual(constructor.call_args.kwargs["http_options"].retry_options.attempts, 1)

    def test_bedrock_adapter_passes_system_and_joins_text_blocks(self):
        with patch("boto3.client") as constructor:
            sdk = constructor.return_value.__enter__.return_value
            sdk.converse.return_value = {"output": {"message": {"content": [
                {"text": "Pa"}, {"reasoningContent": {}}, {"text": "ris"}
            ]}}}
            self.assertEqual(client.call_bedrock("Capital?", "Be concise", 0, 123), "Paris")
            arguments = sdk.converse.call_args.kwargs
            self.assertEqual(arguments["system"], [{"text": "Be concise"}])
            self.assertEqual(arguments["inferenceConfig"], {"temperature": 0, "maxTokens": 123})

    def test_anthropic_adapter_passes_system_and_joins_only_text(self):
        with patch("anthropic.Anthropic") as constructor:
            sdk = constructor.return_value.__enter__.return_value
            sdk.messages.create.return_value.content = [
                SimpleNamespace(type="thinking"),
                SimpleNamespace(type="text", text="Paris"),
            ]
            self.assertEqual(client.call_anthropic("Capital?", "Be concise", 0, 123), "Paris")
            arguments = sdk.messages.create.call_args.kwargs
            self.assertEqual(arguments["system"], "Be concise")
            self.assertEqual(arguments["max_tokens"], 123)

    def test_openai_adapter_passes_system_temperature_and_token_limit(self):
        with patch("openai.OpenAI") as constructor:
            sdk = constructor.return_value.__enter__.return_value
            sdk.chat.completions.create.return_value.choices = [
                SimpleNamespace(message=SimpleNamespace(content="Paris"))
            ]
            self.assertEqual(client.call_openai("Capital?", "Be concise", 0, 123), "Paris")
            arguments = sdk.chat.completions.create.call_args.kwargs
            self.assertEqual(arguments["messages"][0], {"role": "system", "content": "Be concise"})
            self.assertEqual(arguments["temperature"], 0)
            self.assertEqual(arguments["max_completion_tokens"], 123)

    def test_ollama_adapter_passes_options_and_closes_connection(self):
        with patch("ollama.Client") as constructor:
            sdk = constructor.return_value
            sdk.chat.return_value.message.content = "Paris"
            self.assertEqual(client.call_ollama("Capital?", "Be concise", 0, 123), "Paris")
            arguments = sdk.chat.call_args.kwargs
            self.assertEqual(arguments["messages"][0], {"role": "system", "content": "Be concise"})
            self.assertEqual(arguments["options"], {"temperature": 0, "num_predict": 123})
            sdk.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
