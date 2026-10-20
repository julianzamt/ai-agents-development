"""Provider SDK adapters; each returns the original response."""

import json
import os

from .config import CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS

def call_bedrock(prompt, system=None, temperature=0.7, max_tokens=500, history=None, schema=None, tools=None):
    import boto3
    from botocore.config import Config

    # Keep retries in LLMClient instead of multiplying them with SDK retries.
    with boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            read_timeout=READ_TIMEOUT_SECONDS,
            retries={"total_max_attempts": 1},
        ),
    ) as client:
        request = {
            "modelId": os.environ.get(
                "BEDROCK_MODEL_ID_CLAUDE", "global.anthropic.claude-sonnet-4-6"
            ),
            "messages": list(history or []) + ([{"role": "user", "content": [{"text": prompt}]}] if prompt is not None else []),
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        if system is not None:
            request["system"] = [{"text": system}]
        if schema is not None:
            request["outputConfig"] = {"textFormat": {"type": "json_schema", "structure": {
                "jsonSchema": {"name": "structured_response", "schema": json.dumps(schema)}
            }}}
        if tools:
            request["toolConfig"] = {"tools": [{"toolSpec": {"name": t.name,
                "description": t.description, "inputSchema": {"json": t.schema}}} for t in tools]}
        response = client.converse(**request)
    return response


def call_anthropic(prompt, system=None, temperature=0.7, max_tokens=500, history=None, schema=None, tools=None):
    from anthropic import Anthropic

    with Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0, timeout=READ_TIMEOUT_SECONDS
    ) as client:
        request = {
            "model": os.environ.get("ANTHROPIC_MODEL_ID", "claude-sonnet-4-6"),
            "max_tokens": max_tokens,
            "messages": list(history or []) + ([{"role": "user", "content": prompt}] if prompt is not None else []),
        }
        # As in the notebook, omit temperature: some Claude models reject it.
        if system is not None:
            request["system"] = system
        if schema is not None:
            request["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        if tools:
            request["tools"] = [{"name": t.name, "description": t.description,
                                 "input_schema": t.schema} for t in tools]
        response = client.messages.create(**request)
    return response


def call_openai(prompt, system=None, temperature=0.7, max_tokens=500, history=None, schema=None, tools=None):
    from openai import OpenAI

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.extend(history or [])
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
    with OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=READ_TIMEOUT_SECONDS
    ) as client:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_ID", "gpt-4o"),
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **({"tools": [{"type": "function", "function": {"name": t.name,
                "description": t.description, "parameters": t.schema}} for t in tools]} if tools else {}),
            **({"response_format": {"type": "json_schema", "json_schema": {
                "name": "structured_response", "strict": True, "schema": schema
            }}} if schema is not None else {}),
        )
    return response


def call_google(prompt, system=None, temperature=0.7, max_tokens=500, history=None, schema=None, tools=None):
    from google import genai
    from google.genai import types

    with genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options=types.HttpOptions(
            timeout=READ_TIMEOUT_SECONDS * 1_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        response = client.models.generate_content(
            model=os.environ.get("GOOGLE_MODEL_ID", "gemini-3.5-flash-lite"),
            contents=(list(history or []) + ([{"role": "user", "parts": [{"text": prompt}]}]
                      if prompt is not None else [])) if history or prompt is None else prompt,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
                tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
                    name=t.name, description=t.description, parameters_json_schema=t.schema
                ) for t in tools])] if tools else None,
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
                **({"response_mime_type": "application/json", "response_json_schema": schema}
                   if schema is not None else {}),
            ),
        )
    return response


def call_ollama(prompt, system=None, temperature=0.7, max_tokens=500, history=None, schema=None, tools=None):
    from ollama import Client

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.extend(history or [])
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
    client = Client(host=os.environ.get("OLLAMA_HOST"), timeout=READ_TIMEOUT_SECONDS)
    try:
        response = client.chat(
            model=os.environ.get("OLLAMA_MODEL_ID", "llama3.1"),
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
            **({"tools": [{"type": "function", "function": {"name": t.name,
                "description": t.description, "parameters": t.schema}} for t in tools]} if tools else {}),
            **({"format": schema} if schema is not None else {}),
        )
        return response
    finally:
        client.close()
