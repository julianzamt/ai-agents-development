"""Explicit tool execution, validation and native message round trips."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types
from pydantic import ValidationError

from ia_agents.llm.core import LLMClient
from ia_agents.llm.tool_calling import result_messages
from ia_agents.tools import TICKET_TOOLS, TOOL_GROUPS
from ia_agents import cli
from ia_agents.schemas import SupportQuery
from ia_agents.llm.state import load_history


def google_response(name=None, arguments=None):
    part = types.Part(function_call=types.FunctionCall(name=name, args=arguments)) if name else types.Part(text="Not found")
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[part]))])


class ToolTests(unittest.TestCase):
    def test_missing_ticket_and_mock_escalation(self):
        self.assertEqual(TICKET_TOOLS[0].execute({"ticket_id": "T-999"})["status"], "Not found")
        self.assertEqual(TICKET_TOOLS[1].execute({"ticket_id": "T-001", "reason": "Outage"})["status"], "Escalated (mock)")
        with self.assertRaises(ValidationError):
            TICKET_TOOLS[0].execute({"ticket_id": 123})

    def test_google_loop_keeps_signature_and_results(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "google"
        first = google_response("get_ticket_status", {"ticket_id": "T-999"})
        first.candidates[0].content.parts[0].thought_signature = b"\xff"
        final = google_response()
        with patch.object(llm, "generate", side_effect=[first, final]) as generate:
            response, history = llm.generate_with_tools("Find T-999")
            self.assertIs(response, final)
            self.assertEqual(len(history), 4)
            self.assertEqual(history[1]["parts"][0]["thought_signature"], b"\xff")
            self.assertEqual(history[2]["parts"][0]["function_response"]["response"]["status"], "Not found")
            types.Content.model_validate(history[2])
            self.assertIsNone(generate.call_args.args[0])

    def test_unknown_invalid_and_unbounded_requests(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "google"
        for name, args in [("unknown", {}), ("get_ticket_status", {"ticket_id": "bad"})]:
            with patch.object(llm, "generate", return_value=google_response(name, args)):
                with self.assertRaises(ValueError):
                    llm.generate_with_tools("Find ticket")
        with patch.object(llm, "generate", return_value=google_response("get_ticket_status", {"ticket_id": "T-001"})) as generate:
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                llm.generate_with_tools("Find ticket", max_rounds=1)
            self.assertEqual(generate.call_count, 2)

    def test_multiple_groups_reach_model_and_execute(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "google"
        groups = {"status": TICKET_TOOLS[:1], "escalation": TICKET_TOOLS[1:]}
        with patch.dict(TOOL_GROUPS, groups), patch.object(llm, "generate", side_effect=[
            google_response("escalate_ticket", {"ticket_id": "T-001", "reason": "Outage"}),
            google_response(),
        ]) as generate:
            args = cli.parse_cli_args(["Hello", "--tools", "status", "escalation", "status"])
            self.assertEqual(args.tools, ["status", "escalation", "status"])
            _, history = llm.generate_with_tools("Escalate", tool_groups=args.tools)
            self.assertEqual(generate.call_args.kwargs["tools"], TICKET_TOOLS)
            result = history[2]["parts"][0]["function_response"]["response"]
            self.assertEqual(result["status"], "Escalated (mock)")

    def test_groups_are_validated_before_model_call(self):
        llm = LLMClient.__new__(LLMClient)
        with patch.object(llm, "generate") as generate:
            for groups in ([], ["unknown"]):
                with self.subTest(groups=groups), self.assertRaises(ValueError):
                    llm.generate_with_tools("Hello", tool_groups=groups)
            generate.assert_not_called()
        for argv in (["Hello", "--tools"], ["Hello", "--tools", "unknown"]):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                cli.parse_cli_args(argv)

    def test_cli_batch_tools_and_structured_are_independent(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "google"
        for mode in (None, "native", "prompt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "config.json"
                history = Path(directory) / "history.json"
                history.write_text("Do not read or change batch history")
                tool_queries = []
                formatting_queries = []
                attempts = {}
                def respond(prompt, *args, **kwargs):
                    if kwargs.get("tools"):
                        messages = kwargs["history"]
                        ticket = messages[0]["parts"][0]["text"]
                        tool_queries.append((ticket, len(messages)))
                        if len(messages) == 1:
                            return google_response("get_ticket_status", {"ticket_id": ticket})
                        self.assertEqual(messages[2]["parts"][0]["function_response"]["response"]["ticket_id"], ticket)
                        return google_response()
                    # Retry prompts retain the original transcript and ticket.
                    ticket = "T-001" if '"text": "T-001"' in prompt else "T-002"
                    attempts[ticket] = attempts.get(ticket, 0) + 1
                    formatting_queries.append(kwargs)
                    response = google_response()
                    response.candidates[0].content.parts[0].text = (
                        "invalid" if attempts[ticket] == 1 else SupportQuery(
                            category="technical", urgency="low", summary=ticket,
                            requires_escalation=False).model_dump_json()
                    )
                    return response
                with patch.object(cli, "CONFIG_PATH", config), patch.object(cli, "LLMClient", return_value=llm), patch.object(
                    llm, "generate", side_effect=respond
                ), patch("builtins.print") as output:
                    argv = ["--history", str(history), "--batch", "T-001", "T-002", "--tools", "tickets"]
                    if mode:
                        argv += ["--structured", "support", "--structured-mode", mode, "--structured-attempts", "2"]
                    cli.main(argv)
                self.assertCountEqual(tool_queries, [(ticket, size) for ticket in ("T-001", "T-002") for size in (1, 3)])
                self.assertIn("[1] T-001", output.call_args_list[0].args[0])
                self.assertIn("[2] T-002", output.call_args_list[1].args[0])
                if mode:
                    self.assertEqual(attempts, {"T-001": 2, "T-002": 2})
                    for request in formatting_queries:
                        self.assertIsNone(request["history"])
                        self.assertEqual(request["schema"], SupportQuery.model_json_schema() if mode == "native" else None)
                    self.assertIn('"summary": "T-001"', output.call_args_list[0].args[0])
                self.assertFalse(config.exists())
                self.assertEqual(history.read_text(), "Do not read or change batch history")

    def test_single_tools_structured_saves_complete_exchange(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "google"
        result = SupportQuery(category="technical", urgency="low", summary="Ticket checked",
                              requires_escalation=False)
        final = google_response()
        final.candidates[0].content.parts[0].text = result.model_dump_json()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            history = Path(directory) / "history.json"
            with patch.object(cli, "CONFIG_PATH", config), patch.object(cli, "LLMClient", return_value=llm), patch.object(
                llm, "generate", side_effect=[
                    google_response("get_ticket_status", {"ticket_id": "T-001"}),
                    google_response(), final,
                ]
            ), patch("builtins.print"):
                cli.main(["T-001", "--tools", "tickets", "--structured", "support", "--history", str(history)])
            messages = load_history(history)["google"]
            self.assertEqual(len(messages), 6)
            self.assertEqual(messages[2]["parts"][0]["function_response"]["response"]["ticket_id"], "T-001")
            self.assertEqual(messages[-1]["parts"][0]["text"], result.model_dump_json())

    def test_structured_failure_does_not_repeat_tool_execution(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "google"
        with patch.object(llm, "generate", side_effect=[
            google_response("get_ticket_status", {"ticket_id": "T-001"}),
            google_response(), google_response(), google_response(),
        ]) as generate:
            with self.assertRaisesRegex(ValueError, "after 2 attempts"):
                llm.generate_batch(["T-001"], tool_groups=["tickets"],
                                   schema_model=SupportQuery, max_attempts=2)
            self.assertEqual(generate.call_count, 4)
            self.assertEqual(sum(bool(request.kwargs.get("tools")) for request in generate.call_args_list), 2)

    def test_batch_tool_failure_propagates(self):
        llm = LLMClient.__new__(LLMClient)
        llm.provider = "google"
        with patch.object(llm, "generate", return_value=google_response("unknown", {})):
            with self.assertRaisesRegex(ValueError, "Unknown tool"):
                llm.generate_batch(["Check ticket"], tool_groups=["tickets"])

    def test_result_ids_are_preserved(self):
        results = [("call-1", "get_ticket_status", {"status": "Resolved"})]
        self.assertEqual(result_messages("openai", results)[0]["tool_call_id"], "call-1")
        self.assertEqual(result_messages("anthropic", results)[0]["content"][0]["tool_use_id"], "call-1")
        self.assertEqual(result_messages("bedrock", results)[0]["content"][0]["toolResult"]["toolUseId"], "call-1")


if __name__ == "__main__":
    unittest.main()
