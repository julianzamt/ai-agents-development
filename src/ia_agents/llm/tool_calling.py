"""Provider-native tool messages and a bounded application-owned execution loop."""

import json
import logging

from ..tools import select_tools
from .conversation import result_messages

logger = logging.getLogger(__name__)


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


def run_tools(llm, system, temperature, max_tokens, conversation, max_tools_rounds, tool_groups):
    if type(max_tools_rounds) is not int or max_tools_rounds <= 0:
        raise ValueError("max_tools_rounds must be a positive integer.")
    
    tools = select_tools(tool_groups)
    messages = conversation.messages
    
    registry = {tool.name: tool for tool in tools}
    # Allow a final model response after the last permitted execution round.
    for round_index in range(max_tools_rounds + 1):
        response = llm._request(None, system, temperature, max_tokens,
                                history=messages, tools=tools)
        _, calls = tool_requests(llm.provider, response)
        conversation.add_assistant_response(response)
        if not calls:
            return response
        if round_index == max_tools_rounds:
            raise RuntimeError(f"Tool calling exceeded {max_tools_rounds} execution rounds.")
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
            logger.warning("Executing tool %s with arguments %s", tool.name, arguments)
            results.append((call_id, tool.name, tool.function(**arguments)))
        conversation.add_tool_results(results)
