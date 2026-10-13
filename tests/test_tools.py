"""Explicit tool execution, validation and native message round trips."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types
from pydantic import ValidationError

from agentes_ia.llm.core import LLMClient
from agentes_ia.llm.tool_calling import result_messages
from agentes_ia.tools import TICKET_TOOLS


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

    def test_result_ids_are_preserved(self):
        results = [("call-1", "get_ticket_status", {"status": "Resolved"})]
        self.assertEqual(result_messages("openai", results)[0]["tool_call_id"], "call-1")
        self.assertEqual(result_messages("anthropic", results)[0]["content"][0]["tool_use_id"], "call-1")
        self.assertEqual(result_messages("bedrock", results)[0]["content"][0]["toolResult"]["toolUseId"], "call-1")


if __name__ == "__main__":
    unittest.main()
