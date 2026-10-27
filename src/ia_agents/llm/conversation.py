"""Own conversation updates and provider-native message formatting."""

import copy
import json


def user_message(provider, prompt):
    if provider == "google":
        return {"role": "user", "parts": [{"text": prompt}]}
    if provider == "bedrock":
        return {"role": "user", "content": [{"text": prompt}]}
    return {"role": "user", "content": prompt}


def assistant_message(provider, response):
    if provider == "google":
        candidates = response.candidates or []
        if not candidates or candidates[0].content is None:
            raise ValueError("Google returned no content to save in history.")
        return candidates[0].content.model_dump(exclude_none=True)
    if provider == "bedrock":
        return response["output"]["message"]
    if provider == "anthropic":
        return {"role": "assistant", "content": [part.model_dump(exclude_none=True)
                                                  for part in response.content]}
    if provider == "openai":
        message = response.choices[0].message
        return message.model_dump(exclude_none=True, exclude={"parsed", "annotations"})
    return response.message.model_dump(exclude_none=True)


def result_messages(provider, results):
    if provider == "google":
        return [{"role": "user", "parts": [{"function_response": {
            "name": name, "response": result, **({"id": call_id} if call_id else {})
        }} for call_id, name, result in results]}]
    if provider == "bedrock":
        return [{"role": "user", "content": [{"toolResult": {
            "toolUseId": call_id, "content": [{"json": result}]
        }} for call_id, _, result in results]}]
    if provider == "anthropic":
        return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id,
                    "content": json.dumps(result)} for call_id, _, result in results]}]
    return [{"role": "tool", "content": json.dumps(result),
             **({"tool_call_id": call_id} if provider == "openai" else {"tool_name": name})}
            for call_id, name, result in results]


class Conversation:
    """Simple chat history tracking. No trimming, summarizing or context limiting yet"""

    def __init__(self, provider, history=None):
        self.provider = provider
        self.messages = copy.deepcopy(history or [])

    def add_user_message(self, prompt):
        self.messages.append(user_message(self.provider, prompt))

    def add_assistant_response(self, response):
        self.messages.append(assistant_message(self.provider, response))

    def add_tool_results(self, results):
        self.messages.extend(result_messages(self.provider, results))


def request_messages(provider, history, prompt):
    """Build transient request state without changing persistent history."""
    conversation = Conversation(provider, history)
    if prompt is not None:
        conversation.add_user_message(prompt)
    return conversation.messages
