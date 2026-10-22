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
from ollama import Message, ResponseError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import RateLimitError as OpenAIRateLimitError

from ia_agents.llm import core as client
from ia_agents import cli
from ia_agents.llm import providers, state
from ia_agents.llm import responses as response_utils
from ia_agents.llm.config import DEFAULT_PARAMS


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
        jitter = patch.object(client.random, "uniform", return_value=0)
        self.jitter = jitter.start()
        self.addCleanup(jitter.stop)
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
                    result = self.select(provider)._request("Capital?", "Be concise", 0, 123)
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

    def test_generate_measures_elapsed_time(self):
        llm = self.select("ollama")
        response = SimpleNamespace(prompt_eval_count=0, eval_count=0,
                                   message=Message(role="assistant", content="Paris"))
        with patch.object(llm, "_request", return_value=response), patch.object(
            client.time, "perf_counter", side_effect=[10, 10.5]
        ):
            self.assertEqual(llm.generate("Capital?").report,
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
                    self.assertEqual(self.select(provider)._request("Capital?", None, 0.7, 500), "Paris")
                    self.assertEqual(generate.call_count, 3)
                    self.assertEqual(self.sleep.call_args_list, [call(1), call(2)])

    def test_exhausted_retries_are_bounded_and_preserve_cause(self):
        error = sdk_error("google", 429)
        with patch.object(providers, "call_google", side_effect=error) as generate:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts") as raised:
                client.LLMClient()._request("Hello", None, 0.7, 500)
            self.assertIs(raised.exception.__cause__, error)
            self.assertEqual(generate.call_count, 3)
            self.assertEqual(self.sleep.call_args_list, [call(1), call(2)])

    def test_each_sdk_validation_error_is_clear_and_not_retried(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                error = sdk_error(provider, 400)
                with patch.object(providers, f"call_{provider}", side_effect=error) as generate:
                    with self.assertRaisesRegex(ValueError, "Invalid request") as raised:
                        self.select(provider)._request("Hello", None, 0.7, 500)
                    self.assertIs(raised.exception.__cause__, error)
                    generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_permission_errors_are_propagated_without_retry(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                error = sdk_error(provider, 403)
                with patch.object(providers, f"call_{provider}", side_effect=error) as generate:
                    with self.assertRaises(type(error)) as raised:
                        self.select(provider)._request("Hello", None, 0.7, 500)
                    self.assertIs(raised.exception, error)
                    generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_retry_delays_include_jitter_and_stop_after_success(self):
        self.jitter.side_effect = [0.25, 0.75]
        with patch.object(providers, "call_google", side_effect=[TimeoutError(), TimeoutError(), "OK"]):
            self.assertEqual(self.select("google")._request("Hello", None, 0.7, 500), "OK")
        self.assertEqual(self.sleep.call_args_list, [call(1.25), call(2.75)])
        self.assertEqual(self.jitter.call_args_list, [call(0, 1), call(0, 1)])

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
                    self.assertEqual(self.select(provider)._request("Capital?", None, 0.7, 500), "Paris")
                    self.sleep.assert_called_once_with(1)

    def test_missing_credentials_and_unsupported_provider_are_clear(self):
        os.environ.pop("GOOGLE_API_KEY")
        with self.assertRaisesRegex(ValueError, "Missing GOOGLE_API_KEY"):
            client.LLMClient()
        os.environ["LLM_PROVIDER"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported LLM_PROVIDER"):
            client.LLMClient()

    def test_configured_default_and_ollama_needs_no_credentials(self):
        os.environ.pop("LLM_PROVIDER")
        self.assertEqual(client.LLMClient().provider, client.DEFAULT_PROVIDER)
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            self.assertEqual(client.LLMClient().provider, "ollama")

    def test_response_without_text_is_preserved_for_inspection(self):
        response = SimpleNamespace(text=None, prompt_feedback={"block_reason": "SAFETY"})
        with patch.object(providers, "call_google", return_value=response) as generate:
            self.assertIs(client.LLMClient()._request("Hello", None, 0.7, 500), response)
            generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_batch_runs_concurrently_and_preserves_order(self):
        barrier = threading.Barrier(3)

        def answer(prompt, system, temperature, max_tokens):
            barrier.wait(timeout=5)
            return prompt.upper()

        prompts = [f"question {index}" for index in range(6)]
        with patch.object(client.LLMClient, "generate", side_effect=answer):
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
            self.assertEqual(constructor.call_args.kwargs["http_options"].timeout, 60_000)

    def test_bedrock_adapter_passes_system_and_returns_full_response(self):
        with patch("boto3.client") as constructor:
            sdk = constructor.return_value.__enter__.return_value
            sdk.converse.return_value = {"output": {"message": {"content": [
                {"text": "Pa"}, {"reasoningContent": {}}, {"text": "ris"}
            ]}}}
            self.assertIs(providers.call_bedrock("Capital?", "Be concise", 0, 123), sdk.converse.return_value)
            config = constructor.call_args.kwargs["config"]
            self.assertEqual(config.connect_timeout, 10)
            self.assertEqual(config.read_timeout, 60)
            self.assertEqual(config.retries["total_max_attempts"], 1)
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
            self.assertEqual(constructor.call_args.kwargs["timeout"], 60)
            self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
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
            self.assertEqual(constructor.call_args.kwargs["timeout"], 60)
            self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
            arguments = sdk.chat.completions.create.call_args.kwargs
            self.assertEqual(arguments["messages"][0], {"role": "system", "content": "Be concise"})
            self.assertEqual(arguments["temperature"], 0)
            self.assertEqual(arguments["max_completion_tokens"], 123)

    def test_ollama_adapter_passes_options_and_closes_connection(self):
        with patch("ollama.Client") as constructor:
            sdk = constructor.return_value
            sdk.chat.return_value.message.content = "Paris"
            self.assertIs(providers.call_ollama("Capital?", "Be concise", 0, 123), sdk.chat.return_value)
            self.assertEqual(constructor.call_args.kwargs["timeout"], 60)
            arguments = sdk.chat.call_args.kwargs
            self.assertEqual(arguments["messages"][0], {"role": "system", "content": "Be concise"})
            self.assertEqual(arguments["options"], {"temperature": 0, "num_predict": 123})
            sdk.close.assert_called_once()


class CLITests(unittest.TestCase):
    def test_history_cannot_use_configuration_path(self):
        path = cli.CONFIG_PATH.parent / ".." / "data" / cli.CONFIG_PATH.name
        with patch("sys.stderr") as stderr, self.assertRaises(SystemExit) as raised:
            cli.parse_cli_args(["Hello", "--history", str(path)])
        self.assertEqual(raised.exception.code, 2)
        message = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("Configuration and history must use different files.", message)

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
            config_path = Path(directory) / "config.json"
            config_patch = patch.object(cli, "CONFIG_PATH", config_path)
            config_patch.start()
            self.addCleanup(config_patch.stop)
            flags = ["--history", str(path)]
            cli.main(flags + ["--system", "Be brief", "--temp", "0.3", "--max_tokens", "100"])
            constructor.assert_not_called()
            self.assertEqual(state.load_config(config_path),
                             dict(system="Be brief", temperature=0.3, max_tokens=100))
            llm = constructor.return_value
            llm.provider = "google"
            def generate(prompt, **kwargs):
                history = kwargs.get("history", [])
                return SimpleNamespace(response=response,
                    history=history + response_utils.conversation_turn("google", prompt, response),
                    structured=None, report="report")
            llm.generate.side_effect = generate
            cli.main(flags + ["My name is Julian"])
            llm.generate.assert_called_with("My name is Julian", system="Be brief", temperature=0.3,
                                            max_tokens=100, history=[])
            history = state.load_history(path)["google"]
            self.assertEqual(history[1]["parts"][0]["thought_signature"], b"\xff\x00")
            # The SDK accepts the restored native content, including bytes.
            types.Content.model_validate(history[1])
            cli.main(flags + ["What is my name?", "Override", "0.5", "50"])
            llm.generate.assert_called_with("What is my name?", system="Override", temperature=0.5,
                                            max_tokens=50, history=history)
            self.assertEqual(state.load_config(config_path)["system"], "Be brief")
            self.assertEqual(len(state.load_history(path)["google"]), 4)
            config = state.load_config(config_path)
            cli.main(flags + ["--reset-chat"])
            self.assertEqual(state.load_history(path), {})
            self.assertEqual(state.load_config(config_path), config)
            cli.main(flags + ["Hello"])
            history = state.load_history(path)
            cli.main(flags + ["--reset-config"])
            self.assertEqual(state.load_config(config_path), DEFAULT_PARAMS)
            self.assertEqual(state.load_history(path), history)
            cli.main(flags + ["--system", "Custom"])
            cli.main(flags + ["--reset-all"])
            self.assertEqual(state.load_history(path), {})
            self.assertEqual(state.load_config(config_path), DEFAULT_PARAMS)

    def test_config_option_is_not_supported(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            cli.parse_cli_args(["--config", "other.json", "--prompt", "Hello"])

    def test_config_falls_back_for_missing_file_and_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            self.assertEqual(state.load_config(path), DEFAULT_PARAMS)
            state.save_json(path, {"temperature": 0.2})
            self.assertEqual(state.load_config(path), DEFAULT_PARAMS | {"temperature": 0.2})
            state.save_json(path, {"max_tokens": 0})
            with self.assertRaises(ValueError):
                state.load_config(path)

    def test_config_changes_do_not_read_or_rewrite_history(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            config = Path(directory) / "config.json"
            history = Path(directory) / "history.json"
            history.write_text("broken history")
            config_patch = patch.object(cli, "CONFIG_PATH", config)
            config_patch.start()
            self.addCleanup(config_patch.stop)
            flags = ["--history", str(history)]
            cli.main(flags + ["--temperature", "0.2"])
            self.assertEqual(state.load_config(config)["temperature"], 0.2)
            cli.main(flags + ["--reset-config"])
            self.assertEqual(state.load_config(config), DEFAULT_PARAMS)
            self.assertEqual(history.read_text(), "broken history")

    def test_failed_request_does_not_change_history(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"), patch.object(cli, "LLMClient") as constructor:
            path = Path(directory) / "state.json"
            config_patch = patch.object(cli, "CONFIG_PATH", Path(directory) / "config.json")
            config_patch.start()
            self.addCleanup(config_patch.stop)
            state.save_json(path, {})
            before = path.read_text()
            constructor.return_value.provider = "google"
            constructor.return_value.generate.side_effect = RuntimeError("failed")
            with self.assertRaises(SystemExit):
                cli.main(["Hello", "--history", str(path)])
            self.assertEqual(path.read_text(), before)

    def test_resets_persist_before_single_and_batch_requests_fail(self):
        for batch in (False, True):
            for reset in ("--reset-chat", "--reset-config", "--reset-all"):
                with self.subTest(batch=batch, reset=reset), tempfile.TemporaryDirectory() as directory:
                    config = Path(directory) / "config.json"
                    history = Path(directory) / "history.json"
                    saved_config = DEFAULT_PARAMS | {"temperature": 0.2}
                    saved_history = {"google": [{"role": "user", "parts": [{"text": "Before"}]}]}
                    state.save_json(config, saved_config)
                    state.save_json(history, saved_history)
                    def fail(*args, **kwargs):
                        self.assertEqual(state.load_config(config),
                                         saved_config if reset == "--reset-chat" else DEFAULT_PARAMS)
                        self.assertEqual(state.load_history(history),
                                         saved_history if reset == "--reset-config" else {})
                        self.assertEqual(kwargs["temperature"], 0.9)
                        raise RuntimeError("Request failed")
                    with patch.object(cli, "CONFIG_PATH", config), patch.object(cli, "LLMClient") as constructor:
                        llm = constructor.return_value
                        llm.provider = "google"
                        llm.generate.side_effect = fail
                        llm.generate_batch.side_effect = fail
                        argv = ["--batch", "Hello"] if batch else ["Hello"]
                        with self.assertRaisesRegex(SystemExit, "Request failed"):
                            cli.main(argv + [reset, "--temperature", "0.9", "--history", str(history)])

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
                    getattr(providers, f"call_{provider}")("Now", None, 0.7, 500, history=history)
                    messages = operation.call_args.kwargs["contents" if provider == "google" else "messages"]
                    self.assertEqual(messages[0], history[0])
                    self.assertEqual(len(messages), 2)
                    self.assertEqual(len(history), 1)

    def test_batch_uses_saved_config_and_leaves_history_unchanged(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print") as output, patch.object(cli, "LLMClient") as constructor:
            path = Path(directory) / "state.json"
            config_path = Path(directory) / "config.json"
            config_patch = patch.object(cli, "CONFIG_PATH", config_path)
            config_patch.start()
            self.addCleanup(config_patch.stop)
            state.save_json(config_path, dict(system="Brief", temperature=0.3, max_tokens=100))
            state.save_json(path, {})
            before = path.read_text()
            constructor.return_value.generate_batch.return_value = [SimpleNamespace(report="report1"), SimpleNamespace(report="report2")]
            cli.main(["--history", str(path), "--batch", "First", "Second"])
            constructor.return_value.generate_batch.assert_called_once_with(
                ["First", "Second"], system="Brief", temperature=0.3,
                max_tokens=100)
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
        with patch.object(llm, "generate", side_effect=lambda prompt, *args: prompt) as generate:
            self.assertEqual(llm.generate_batch(["a", "b", "c"], "Brief", 0.3, 100), ["a", "b", "c"])
            generate.assert_has_calls([call(p, "Brief", 0.3, 100) for p in ["a", "b", "c"]],
                                      any_order=True)

    def test_corrupt_state_requires_explicit_reset(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            path = Path(directory) / "state.json"
            path.write_text("broken")
            with self.assertRaises(ValueError):
                state.load_history(path)
            config_path = Path(directory) / "config.json"
            config_patch = patch.object(cli, "CONFIG_PATH", config_path)
            config_patch.start()
            self.addCleanup(config_patch.stop)
            cli.main(["--reset-all", "--history", str(path)])
            self.assertEqual(state.load_history(path), {})
            self.assertEqual(state.load_config(config_path), DEFAULT_PARAMS)


if __name__ == "__main__":
    unittest.main()
