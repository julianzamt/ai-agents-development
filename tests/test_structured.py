"""Native structured request and validation contracts."""

import os
import tempfile
from pathlib import Path
from ia_agents import cli
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ia_agents.llm.core import LLMClient
from ia_agents.llm import providers
from ia_agents.schemas import SupportQuery, SCHEMAS
from pydantic import BaseModel


class StructuredTests(unittest.TestCase):
    def test_valid_response_and_invalid_response(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "ollama"
        data = SupportQuery(category="technical", urgency="high", summary="Outage",
                            requires_escalation=True)
        response = SimpleNamespace(message=SimpleNamespace(content=data.model_dump_json()))
        with patch.object(llm, "generate", return_value=response) as generate:
            self.assertEqual(llm.generate_structured("Classify", SupportQuery), data)
            self.assertEqual(generate.call_args.kwargs["schema"], SupportQuery.model_json_schema())
            response.message.content = '{"category": "unknown"}'
            with self.assertRaisesRegex(ValueError, "Invalid structured response"):
                llm.generate_structured("Classify", SupportQuery)

    def test_invalid_json_is_repaired_with_feedback(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "ollama"
        data = SupportQuery(category="technical", urgency="high", summary="Outage",
                            requires_escalation=True)
        invalid = SimpleNamespace(message=SimpleNamespace(content="not JSON"))
        valid = SimpleNamespace(message=SimpleNamespace(content=data.model_dump_json()))
        history = [{"role": "user", "content": "Earlier"}]
        with patch.object(llm, "generate", side_effect=[invalid, valid]) as generate:
            result, response = llm.generate_structured("Classify", SupportQuery,
                history=history, include_response=True)
            self.assertEqual(result, data)
            self.assertIs(response, valid)
            self.assertEqual(generate.call_count, 2)
            retry = generate.call_args.args[0]
            self.assertIn("not JSON", retry)
            self.assertIn("Validation errors", retry)
            self.assertIn("Classify", retry)
            self.assertEqual(history, [{"role": "user", "content": "Earlier"}])

    def test_exhaustion_is_bounded_and_preserves_validation_error(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "ollama"
        invalid = SimpleNamespace(message=SimpleNamespace(content="{}"))
        with patch.object(llm, "generate", return_value=invalid) as generate:
            with self.assertRaisesRegex(ValueError, "after 2 attempts") as raised:
                llm.generate_structured("Classify", SupportQuery, max_attempts=2)
            self.assertEqual(generate.call_count, 2)
            self.assertIsNotNone(raised.exception.__cause__)

    def test_prompt_only_mode_and_provider_errors(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "ollama"
        data = SupportQuery(category="technical", urgency="high", summary="Outage",
                            requires_escalation=True)
        valid = SimpleNamespace(message=SimpleNamespace(content=data.model_dump_json()))
        with patch.object(llm, "generate", return_value=valid) as generate:
            self.assertEqual(llm.generate_structured("Classify", SupportQuery, native=False), data)
            self.assertIsNone(generate.call_args.kwargs["schema"])
            self.assertIn('"requires_escalation"', generate.call_args.args[0])
        with patch.object(llm, "generate", side_effect=RuntimeError("Authentication failed")) as generate:
            with self.assertRaisesRegex(RuntimeError, "Authentication"):
                llm.generate_structured("Classify", SupportQuery)
            self.assertEqual(generate.call_count, 1)

    def test_invalid_attempt_limit_is_rejected_without_request(self):
        llm = LLMClient.__new__(LLMClient)
        with patch.object(llm, "generate") as generate:
            for count in (0, -1, True, 1.5):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    llm.generate_structured("Classify", SupportQuery, max_attempts=count)
            generate.assert_not_called()

    def test_structured_batch_retries_independently_and_preserves_order(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "ollama"
        attempts = {}
        def respond(prompt, *args, **kwargs):
            original = "First" if prompt.startswith("First") else "Second"
            attempts[original] = attempts.get(original, 0) + 1
            text = "invalid" if original == "First" and attempts[original] == 1 else SupportQuery(
                category="billing", urgency="low", summary=original,
                requires_escalation=False).model_dump_json()
            return SimpleNamespace(message=SimpleNamespace(content=text))
        with patch.object(llm, "generate", side_effect=respond):
            results = llm.generate_batch(["First", "Second"], schema_model=SupportQuery)
        self.assertEqual([result.summary for result in results], ["First", "Second"])
        self.assertEqual(attempts, {"First": 2, "Second": 1})

    def test_cli_structured_batch_reports_and_keeps_local_files(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "ollama"
        data = SupportQuery(category="billing", urgency="low", summary="Duplicate charge",
                            requires_escalation=False)
        response = SimpleNamespace(message=SimpleNamespace(content=data.model_dump_json()))
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            history = Path(directory) / "history.json"
            config.write_text('{"temperature": 0.2}')
            history.write_text("History must not be read")
            for mode in ("native", "prompt"):
                with patch.object(cli, "CONFIG_PATH", config), patch.object(cli, "LLMClient", return_value=llm), patch.object(
                    llm, "generate", return_value=response
                ) as generate, patch("builtins.print") as output:
                    cli.main(["--history", str(history), "--batch", "First", "Second",
                              "--structured", "support", "--structured-mode", mode,
                              "--structured-attempts", "2"])
                    self.assertEqual(generate.call_count, 2)
                    for request in generate.call_args_list:
                        self.assertEqual(request.args[2], 0.2)
                        self.assertEqual(request.kwargs["schema"],
                                         SupportQuery.model_json_schema() if mode == "native" else None)
                    self.assertIn("[1] First", output.call_args_list[0].args[0])
                    self.assertIn("[2] Second", output.call_args_list[1].args[0])
                    self.assertIn('"summary": "Duplicate charge"', output.call_args_list[0].args[0])
            self.assertEqual(config.read_text(), '{"temperature": 0.2}')
            self.assertEqual(history.read_text(), "History must not be read")

    def test_structured_batch_propagates_validation_failure(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "ollama"
        response = SimpleNamespace(message=SimpleNamespace(content="{}"))
        with patch.object(llm, "generate", return_value=response) as generate:
            with self.assertRaisesRegex(ValueError, "after 2 attempts"):
                llm.generate_batch(["First"], schema_model=SupportQuery, max_attempts=2)
            self.assertEqual(generate.call_count, 2)

    def test_cli_selects_registered_schema_in_every_execution_path(self):
        class OtherSchema(BaseModel):
            answer: str

        result = OtherSchema(answer="Selected alternate schema")
        response = SimpleNamespace(message=SimpleNamespace(
            content=result.model_dump_json(),
            model_dump=lambda **kwargs: {"role": "assistant", "content": result.model_dump_json()},
        ))
        for batch in (False, True):
            for tools in (False, True):
                with self.subTest(batch=batch, tools=tools), tempfile.TemporaryDirectory() as directory:
                    with patch.dict(SCHEMAS, other=OtherSchema), patch.object(
                        cli, "CONFIG_PATH", Path(directory) / "config.json"
                    ), patch.object(cli, "LLMClient") as constructor, patch("builtins.print"):
                        llm = constructor.return_value
                        llm.provider = "ollama"
                        llm.generate_batch.return_value = ["report"]
                        llm.generate_structured.return_value = (result, response)
                        llm.generate_with_tools.return_value = (response, [])
                        argv = ["--batch", "Hello"] if batch else ["Hello"]
                        argv += ["--structured", "other", "--history", str(Path(directory) / "history.json")]
                        if tools:
                            argv += ["--tools", "tickets"]
                        cli.main(argv)
                        if batch:
                            self.assertIs(llm.generate_batch.call_args.kwargs["schema_model"], OtherSchema)
                        elif tools:
                            self.assertIs(llm.generate_with_tools.call_args.kwargs["schema_model"], OtherSchema)
                        else:
                            self.assertIs(llm.generate_structured.call_args.args[1], OtherSchema)

    def test_each_adapter_sends_native_schema(self):
        schema = SupportQuery.model_json_schema()
        targets = [("bedrock", "boto3.client", "converse", "outputConfig"),
                   ("anthropic", "anthropic.Anthropic", "messages.create", "output_config"),
                   ("openai", "openai.OpenAI", "chat.completions.create", "response_format"),
                   ("google", "google.genai.Client", "models.generate_content", "config"),
                   ("ollama", "ollama.Client", "chat", "format")]
        with patch.dict(os.environ, GOOGLE_API_KEY="test", OPENAI_API_KEY="test",
                        ANTHROPIC_API_KEY="test"):
            for provider, target, method, key in targets:
                with self.subTest(provider=provider), patch(target) as constructor:
                    sdk = constructor.return_value if provider == "ollama" else constructor.return_value.__enter__.return_value
                    operation = sdk
                    for name in method.split("."):
                        operation = getattr(operation, name)
                    getattr(providers, f"call_{provider}")("Classify", schema=schema)
                    value = operation.call_args.kwargs[key]
                    if provider == "google":
                        self.assertEqual(value.response_json_schema, schema)
                        self.assertEqual(value.response_mime_type, "application/json")
                    elif provider == "ollama":
                        self.assertEqual(value, schema)
                    else:
                        self.assertIn("json_schema", str(value))


if __name__ == "__main__":
    unittest.main()
