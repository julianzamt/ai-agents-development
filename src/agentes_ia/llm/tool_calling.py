"""Provider-native tool messages and a bounded application-owned execution loop."""

import copy
import json
import logging

from ..tools import TICKET_TOOLS

logger = logging.getLogger(__name__)


def user_message(provider, prompt):
    if provider == "google":
        return {"role": "user", "parts": [{"text": prompt}]}
    if provider == "bedrock":
        return {"role": "user", "content": [{"text": prompt}]}
    return {"role": "user", "content": prompt}


def tool_requests(provider, response):
    """Return the native assistant message and (id, name, arguments) requests."""
    if provider == "google":
        if not response.candidates or response.candidates[0].content is None:
            raise ValueError("Google returned no content for the tool conversation.")
        content = response.candidates[0].content
        return content.model_dump(exclude_none=True), [
            (getattr(part.function_call, "id", None), part.function_call.name, part.function_call.args)
            for part in content.parts or [] if part.function_call is not None]
    if provider == "bedrock":
        message = response["output"]["message"]
        return message, [(p["toolUse"]["toolUseId"], p["toolUse"]["name"], p["toolUse"]["input"])
                         for p in message["content"] if "toolUse" in p]
    if provider == "anthropic":
        return {"role": "assistant", "content": [p.model_dump(exclude_none=True) for p in response.content]}, [
            (p.id, p.name, p.input) for p in response.content if p.type == "tool_use"]
    message = response.choices[0].message if provider == "openai" else response.message
    calls = []
    for call in message.tool_calls or []:
        arguments = call.function.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as error:
                raise ValueError(f"Invalid JSON arguments for tool '{call.function.name}'.") from error
        calls.append((getattr(call, "id", None), call.function.name, arguments))
    return message.model_dump(exclude_none=True, exclude={"parsed", "annotations"}), calls


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


def run_tools(llm, prompt, system, temperature, max_tokens, history, max_rounds):
    if type(max_rounds) is not int or max_rounds <= 0:
        raise ValueError("max_rounds must be a positive integer.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    messages = copy.deepcopy(history or [])
    messages.append(user_message(llm.provider, prompt))
    registry = {tool.name: tool for tool in TICKET_TOOLS}
    # Allow a final model response after the last permitted execution round.
    for round_index in range(max_rounds + 1):
        response = llm.generate(None, system, temperature, max_tokens,
                                history=messages, tools=TICKET_TOOLS)
        assistant, calls = tool_requests(llm.provider, response)
        messages.append(assistant)
        if not calls:
            return response, messages
        if round_index == max_rounds:
            raise RuntimeError(f"Tool calling exceeded {max_rounds} execution rounds.")
        # Validate the whole round before running any registered function.
        validated = []
        for call_id, name, arguments in calls:
            if name not in registry:
                raise ValueError(f"Unknown tool '{name}'.")
            tool = registry[name]
            values = tool.arguments.model_validate(arguments).model_dump()
            validated.append((call_id, tool, values))
        results = []
        for call_id, tool, arguments in validated:
            logger.warning("Executing mock tool %s with arguments %s", tool.name, arguments)
            results.append((call_id, tool.name, tool.function(**arguments)))
        messages.extend(result_messages(llm.provider, results))
