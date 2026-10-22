"""Conversation ownership across saved and transient request state."""

import unittest

from ia_agents.llm.conversation import Conversation, request_messages


class ConversationTests(unittest.TestCase):
    def test_request_state_does_not_change_saved_conversation(self):
        history = [{"role": "user", "content": [{"text": "Earlier"}]}]
        conversation = Conversation("bedrock", history)
        conversation.add_user_message("New prompt")

        request = request_messages("bedrock", conversation.messages, "Correction")
        request[0]["content"][0]["text"] = "Changed"

        self.assertEqual(history[0]["content"][0]["text"], "Earlier")
        self.assertEqual(conversation.messages[0]["content"][0]["text"], "Earlier")
        self.assertEqual(len(conversation.messages), 2)
        self.assertEqual(len(request), 3)

    def test_continuation_keeps_existing_prompt_without_adding_another(self):
        for provider in ("google", "bedrock", "anthropic", "openai", "ollama"):
            with self.subTest(provider=provider):
                conversation = Conversation(provider)
                conversation.add_user_message("Hello")
                request = request_messages(provider, conversation.messages, None)
                self.assertEqual(request, conversation.messages)
                self.assertEqual(len(request), 1)
                self.assertIsNot(request, conversation.messages)
