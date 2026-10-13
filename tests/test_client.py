"""Offline checks for the laboratory's success criteria and retry behavior."""

import os
import tempfile
from pathlib import Path
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import httpx
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import RateLimitError as AnthropicRateLimitError
from botocore.exceptions import ClientError, ReadTimeoutError
from google.genai.errors import ClientError as GoogleClientError
from ollama import ResponseError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import RateLimitError as OpenAIRateLimitError

from agentes_ia.llm import core as client
from agentes_ia import cli
from agentes_ia.llm import providers, state
from agentes_ia.llm import responses as response_utils
from agentes_ia.llm.config import DEFAULT_PARAMS


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
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.com"))
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

    def test_each_provider_returns_full_response_and_forwards_parameters(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                response = SimpleNamespace(content="Paris", usage={"tokens": 10})
                with patch.object(providers, f"call_{provider}", return_value=response) as generate:
                    result = self.select(provider).generate("Capital?", "Be concise", 0, 123)
                    self.assertIs(result, response)
                    generate.assert_called_once_with("Capital?", "Be concise", 0, 123)

    def test_reports_for_all_providers(self):
        responses = {
            "bedrock": {"usage": {"inputTokens": 4, "outputTokens": 2},
                        "output": {"message": {"content": [{"text": "Paris"}]}}},
            "anthropic": SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, cache_read_input_tokens=3,
                                      output_tokens=2),
                content=[SimpleNamespace(type="text", text="Paris")]),
            "openai": SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
                choices=[SimpleNamespace(message=SimpleNamespace(content="Paris"))]),
            "google": SimpleNamespace(
                usage_metadata=SimpleNamespace(prompt_token_count=4, candidates_token_count=2),
                candidates=[SimpleNamespace(content=SimpleNamespace(
                    parts=[SimpleNamespace(text="Paris")]))]),
            "ollama": SimpleNamespace(prompt_eval_count=4, eval_count=2,
                                      message=SimpleNamespace(content="Paris")),
        }
        for provider, response in responses.items():
            with self.subTest(provider=provider):
                self.assertEqual(response_utils.format_response(provider, response, 0.125),
                                 "Latency: 0.125 s\nInput tokens: 4\n"
                                 "Output tokens: 2\nResponse: Paris")

    def test_report_handles_missing_usage_and_text(self):
        self.assertEqual(response_utils.format_response("google", SimpleNamespace(), 1),
                         "Latency: 1.000 s\nInput tokens: N/A\n"
                         "Output tokens: N/A\nResponse: (no text)")

    def test_generate_report_measures_elapsed_time(self):
        llm = self.select("ollama")
        response = SimpleNamespace(prompt_eval_count=0, eval_count=0,
                                   message=SimpleNamespace(content="Paris"))
        with patch.object(llm, "generate", return_value=response), patch.object(
            client.time, "perf_counter", side_effect=[10, 10.5]
        ):
            self.assertEqual(llm.generate_report("Capital?"),
                             "Latency: 0.500 s\nInput tokens: 0\n"
                             "Output tokens: 0\nResponse: Paris")

    def test_invalid_max_tokens_never_calls_provider(self):
        with patch.object(providers, "call_google") as generate:
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
        with patch.object(providers, "call_google") as generate:
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
                    providers, f"call_{provider}", side_effect=[error, error, "Paris"]
                ) as generate:
                    self.assertEqual(self.select(provider).generate("Capital?"), "Paris")
                    self.assertEqual(generate.call_count, 3)
                    self.assertEqual(self.sleep.call_args_list, [call(1), call(2)])

    def test_exhausted_retries_are_bounded_and_preserve_cause(self):
        error = sdk_error("google", 429)
        with patch.object(providers, "call_google", side_effect=error) as generate:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts") as raised:
                client.LLMClient().generate("Hello")
            self.assertIs(raised.exception.__cause__, error)
            self.assertEqual(generate.call_count, 3)
            self.assertEqual(self.sleep.call_args_list, [call(1), call(2)])

    def test_each_sdk_validation_error_is_clear_and_not_retried(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                error = sdk_error(provider, 400)
                with patch.object(providers, f"call_{provider}", side_effect=error) as generate:
                    with self.assertRaisesRegex(ValueError, "Invalid request") as raised:
                        self.select(provider).generate("Hello")
                    self.assertIs(raised.exception.__cause__, error)
                    generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_permission_errors_are_propagated_without_retry(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                error = sdk_error(provider, 403)
                with patch.object(providers, f"call_{provider}", side_effect=error) as generate:
                    with self.assertRaises(type(error)) as raised:
                        self.select(provider).generate("Hello")
                    self.assertIs(raised.exception, error)
                    generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_network_timeouts_retry(self):
        request = httpx.Request("POST", "https://example.com")
        errors = {
            "bedrock": ReadTimeoutError(endpoint_url="https://example.com"),
            "google": TimeoutError("test timeout"),
            "openai": OpenAIAPITimeoutError(request),
            "anthropic": AnthropicAPITimeoutError(request),
        }
        for provider, error in errors.items():
            with self.subTest(provider=provider):
                self.sleep.reset_mock()
                with patch.object(providers, f"call_{provider}", side_effect=[error, "Paris"]):
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

    def test_response_without_text_is_preserved_for_inspection(self):
        response = SimpleNamespace(text=None, prompt_feedback={"block_reason": "SAFETY"})
        with patch.object(providers, "call_google", return_value=response) as generate:
            self.assertIs(client.LLMClient().generate("Hello"), response)
            generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_batch_runs_concurrently_and_preserves_order(self):
        barrier = threading.Barrier(3)

        def answer(prompt, system, temperature, max_tokens):
            barrier.wait(timeout=5)
            return prompt.upper()

        prompts = [f"question {index}" for index in range(6)]
        with patch.object(providers, "call_google", side_effect=answer):
            self.assertEqual(client.LLMClient().generate_batch(prompts), [p.upper() for p in prompts])
        self.assertEqual(client.LLMClient().generate_batch([]), [])

    def test_google_adapter_uses_generate_content_and_full_response(self):
        with patch("google.genai.Client") as constructor:
            sdk = constructor.return_value.__enter__.return_value
            sdk.models.generate_content.return_value.text = "Paris"
            for _ in range(2):
                self.assertIs(providers.call_google("Capital?", "Be concise", 0, 123), sdk.models.generate_content.return_value)
            self.assertEqual(sdk.models.generate_content.call_count, 2)
            sdk.chats.create.assert_not_called()
            self.assertEqual(sdk.models.generate_content.call_args.kwargs["contents"], "Capital?")
            config = sdk.models.generate_content.call_args.kwargs["config"]
            self.assertTrue(config.automatic_function_calling.disable)
            self.assertEqual(config.system_instruction, "Be concise")
            self.assertEqual(config.temperature, 0)
            self.assertEqual(config.max_output_tokens, 123)
            self.assertEqual(constructor.call_args.kwargs["http_options"].retry_options.attempts, 1)

    def test_bedrock_adapter_passes_system_and_returns_full_response(self):
        with patch("boto3.client") as constructor:
            sdk = constructor.return_value.__enter__.return_value
            sdk.converse.return_value = {"output": {"message": {"content": [
                {"text": "Pa"}, {"reasoningContent": {}}, {"text": "ris"}
            ]}}}
            self.assertIs(providers.call_bedrock("Capital?", "Be concise", 0, 123), sdk.converse.return_value)
            arguments = sdk.converse.call_args.kwargs
            self.assertEqual(arguments["system"], [{"text": "Be concise"}])
            self.assertEqual(arguments["inferenceConfig"], {"temperature": 0, "maxTokens": 123})

    def test_anthropic_adapter_passes_system_and_returns_full_response(self):
        with patch("anthropic.Anthropic") as constructor:
            sdk = constructor.return_value.__enter__.return_value
            sdk.messages.create.return_value.content = [
                SimpleNamespace(type="thinking"),
                SimpleNamespace(type="text", text="Paris"),
            ]
            self.assertIs(providers.call_anthropic("Capital?", "Be concise", 0, 123), sdk.messages.create.return_value)
            arguments = sdk.messages.create.call_args.kwargs
            self.assertEqual(arguments["system"], "Be concise")
            self.assertEqual(arguments["max_tokens"], 123)

    def test_openai_adapter_passes_system_temperature_and_token_limit(self):
        with patch("openai.OpenAI") as constructor:
            sdk = constructor.return_value.__enter__.return_value
            sdk.chat.completions.create.return_value.choices = [
                SimpleNamespace(message=SimpleNamespace(content="Paris"))
            ]
            self.assertIs(providers.call_openai("Capital?", "Be concise", 0, 123), sdk.chat.completions.create.return_value)
            arguments = sdk.chat.completions.create.call_args.kwargs
            self.assertEqual(arguments["messages"][0], {"role": "system", "content": "Be concise"})
            self.assertEqual(arguments["temperature"], 0)
            self.assertEqual(arguments["max_completion_tokens"], 123)

    def test_ollama_adapter_passes_options_and_closes_connection(self):
        with patch("ollama.Client") as constructor:
            sdk = constructor.return_value
            sdk.chat.return_value.message.content = "Paris"
            self.assertIs(providers.call_ollama("Capital?", "Be concise", 0, 123), sdk.chat.return_value)
            arguments = sdk.chat.call_args.kwargs
            self.assertEqual(arguments["messages"][0], {"role": "system", "content": "Be concise"})
            self.assertEqual(arguments["options"], {"temperature": 0, "num_predict": 123})
            sdk.close.assert_called_once()


class CLITests(unittest.TestCase):
    def test_positionals_and_flag_precedence(self):
        args = cli.parse_cli_args(["Hello", "Be brief", "0.7", "100", "--temp", "0.3"])
        self.assertEqual(args.prompt, "Hello")
        self.assertEqual(args.system, "Be brief")
        self.assertEqual(args.temperature, 0.3)
        self.assertEqual(args.max_tokens, 100)

    def test_invalid_arguments_fail_before_creating_client(self):
        for argv in ([], ["Hello", "--temp", "nan"], ["Hello", "--temp", "-1"],
                     ["Hello", "--max-tokens", "0"], ["Hello", "--max-tokens", "1.5"]):
            with self.subTest(argv=argv), patch("sys.stderr"), patch.object(cli, "LLMClient") as constructor:
                with self.assertRaises(SystemExit):
                    cli.main(argv)
                constructor.assert_not_called()

    def test_settings_history_overrides_and_reset(self):
        from google.genai import types
        response = SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(
            role="model", parts=[types.Part(text="Hello", thought_signature=b"\xff\x00")]))],
            usage_metadata=None)
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"), patch.object(cli, "LLMClient") as constructor:
            path = Path(directory) / "state.json"
            flags = ["--history", str(path)]
            cli.main(flags + ["--system", "Be brief", "--temp", "0.3", "--max_tokens", "100"])
            constructor.assert_not_called()
            self.assertEqual(state.load_state(path)["params"],
                             dict(system="Be brief", temperature=0.3, max_tokens=100))
            llm = constructor.return_value
            llm.provider = "google"
            llm.generate.return_value = response
            cli.main(flags + ["My name is Julian"])
            llm.generate.assert_called_with("My name is Julian", system="Be brief", temperature=0.3,
                                            max_tokens=100, history=[])
            history = state.load_state(path)["history"]["google"]
            self.assertEqual(history[1]["parts"][0]["thought_signature"], b"\xff\x00")
            # The SDK accepts the restored native content, including bytes.
            types.Content.model_validate(history[1])
            cli.main(flags + ["What is my name?", "Override", "0.5", "50"])
            llm.generate.assert_called_with("What is my name?", system="Override", temperature=0.5,
                                            max_tokens=50, history=history)
            self.assertEqual(state.load_state(path)["params"]["system"], "Be brief")
            self.assertEqual(len(state.load_state(path)["history"]["google"]), 4)
            config = state.load_state(path)["params"]
            cli.main(flags + ["--reset"])
            self.assertEqual(state.load_state(path), {"params": config, "history": {}})
            cli.main(flags + ["Hello"])
            history = state.load_state(path)["history"]
            cli.main(flags + ["--reset-config"])
            self.assertEqual(state.load_state(path),
                             {"params": DEFAULT_PARAMS, "history": history})
            cli.main(flags + ["--system", "Custom"])
            cli.main(flags + ["--reset-all"])
            self.assertEqual(state.load_state(path), state.new_state())

    def test_failed_request_does_not_change_history(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"), patch.object(cli, "LLMClient") as constructor:
            path = Path(directory) / "state.json"
            state.save_state(path, state.new_state())
            before = path.read_text()
            constructor.return_value.provider = "google"
            constructor.return_value.generate.side_effect = RuntimeError("failed")
            with self.assertRaises(SystemExit):
                cli.main(["Hello", "--history", str(path)])
            self.assertEqual(path.read_text(), before)

    def test_adapters_forward_history_in_native_format(self):
        with patch.dict(os.environ, TEST_ENVIRONMENT, clear=True):
            for provider, target, method in (
                ("bedrock", "boto3.client", "converse"),
                ("anthropic", "anthropic.Anthropic", "messages.create"),
                ("openai", "openai.OpenAI", "chat.completions.create"),
                ("google", "google.genai.Client", "models.generate_content"),
                ("ollama", "ollama.Client", "chat"),
            ):
                with self.subTest(provider=provider), patch(target) as constructor:
                    sdk = constructor.return_value if provider == "ollama" else constructor.return_value.__enter__.return_value
                    operation = sdk
                    for name in method.split("."):
                        operation = getattr(operation, name)
                    history = [{"role": "user", "parts": [{"text": "Before"}]}] if provider == "google" else [{"role": "user", "content": "Before"}]
                    getattr(providers, f"call_{provider}")("Now", history=history)
                    messages = operation.call_args.kwargs["contents" if provider == "google" else "messages"]
                    self.assertEqual(messages[0], history[0])
                    self.assertEqual(len(messages), 2)
                    self.assertEqual(len(history), 1)

    def test_batch_uses_saved_config_and_leaves_history_unchanged(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print") as output, patch.object(cli, "LLMClient") as constructor:
            path = Path(directory) / "state.json"
            saved = state.new_state()
            saved["params"] = dict(system="Brief", temperature=0.3, max_tokens=100)
            state.save_state(path, saved)
            before = path.read_text()
            constructor.return_value.generate_batch.return_value = ["report1", "report2"]
            cli.main(["--history", str(path), "--batch", "First", "Second"])
            constructor.return_value.generate_batch.assert_called_once_with(
                ["First", "Second"], system="Brief", temperature=0.3,
                max_tokens=100, report=True)
            self.assertEqual(path.read_text(), before)
            self.assertEqual(output.call_count, 2)

    def test_batch_rejects_invalid_combinations(self):
        for argv in (["--batch"], ["--batch", ""],
                     ["--batch", "a", "b", "c", "d"],
                     ["--prompt", "a", "--batch", "b"]):
            with self.subTest(argv=argv), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    cli.parse_cli_args(argv)

    def test_batch_forwards_config_and_returns_ordered_reports(self):
        llm = client.LLMClient.__new__(client.LLMClient)
        with patch.object(llm, "generate_report", side_effect=lambda prompt, *args: prompt) as generate:
            self.assertEqual(llm.generate_batch(["a", "b", "c"], "Brief", 0.3, 100,
                                                report=True), ["a", "b", "c"])
            generate.assert_has_calls([call(p, "Brief", 0.3, 100) for p in ["a", "b", "c"]],
                                      any_order=True)

    def test_corrupt_state_requires_explicit_reset(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            path = Path(directory) / "state.json"
            path.write_text("broken")
            with self.assertRaises(ValueError):
                state.load_state(path)
            cli.main(["--reset-all", "--history", str(path)])
            self.assertEqual(state.load_state(path), state.new_state())


if __name__ == "__main__":
    unittest.main()
