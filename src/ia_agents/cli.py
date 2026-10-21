import argparse
import math
import time
from pathlib import Path

from .schemas import SCHEMAS
from .tools import TOOL_GROUPS
from .llm.config import DEFAULT_PARAMS, CONFIG_PATH, HISTORY_PATH
from .llm.core import LLMClient
from .llm.responses import conversation_turn, format_response, response_details
from .llm.state import load_config, load_history, save_json


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="Query an LLM with persistent history and configuration.")

    # Positional args
    for name, kind in (("prompt", str), ("system", str), ("temperature", float), ("max_tokens", int)):
        parser.add_argument(f"{name}_pos", nargs="?", type=kind, metavar=name)

    # Optional named args
    parser.add_argument("--prompt")
    parser.add_argument("--system")
    parser.add_argument("--temp", "--temperature", dest="temperature", type=float)
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int)

    # Batched queries
    parser.add_argument("--batch", nargs="+", metavar="PROMPT",
                            help="Run 1 to 3 independent queries using the current configuration")

    # Use a different chat history file than the deafult
    parser.add_argument("--history", type=Path, default=HISTORY_PATH,
                            help="Conversation history JSON (default: data/chat_history.json)")

    # Gives the LLM access to the specified sets of tools
    parser.add_argument("--tools", nargs="+", choices=list(TOOL_GROUPS),
                        help="Enable the selected tool groups")

    # Structured response opts. --structured-attempts and --structured-mode are silently ignored if --structured is not set.
    parser.add_argument("--structured", choices=list(SCHEMAS),
                        help="Return structured output validated against the selected schema")
    parser.add_argument("--structured-attempts", type=int, default=3,
                        help="Maximum structured validation attempts (default: 3)")
    parser.add_argument("--structured-mode", choices=["native", "prompt"], default="native",
                        help="Native schema enforcement or universal prompt-only JSON")

    # Local store management
    resets = parser.add_mutually_exclusive_group()
    resets.add_argument("--reset-chat", action="store_true", help="Clear history only")
    resets.add_argument("--reset-all", action="store_true", help="Clear history and restore default configuration")
    resets.add_argument("--reset-config", action="store_true", help="Restore default configuration only")

    # For passing args into tests
    args = parser.parse_args(argv)

    # Create the values dict. At this stage, missing values stay None; saved configuration and defaults are applied later.
    values = {name: getattr(args, name) if getattr(args, name) is not None
              else getattr(args, f"{name}_pos")
              for name in ("prompt", "system", "temperature", "max_tokens")}

    # Sanity checks
    if CONFIG_PATH.resolve() == args.history.resolve():
        parser.error("Configuration and history must use different files.")

    if values["prompt"] is not None and not values["prompt"].strip(): # No empty prompts
        parser.error("The prompt must not be empty.")

    if args.batch is not None:
        if len(args.batch) > 3 or any(not prompt.strip() for prompt in args.batch):
            parser.error("--batch accepts 1 to 3 non-empty prompts.")
        if values["prompt"] is not None:
            parser.error("--batch cannot be combined with an individual prompt.")

    if args.structured and not args.batch and values["prompt"] is None:
        parser.error("--structured requires a prompt or --batch.")
    if args.structured_attempts <= 0:
        parser.error("--structured-attempts must be positive.")

    if args.tools and not args.batch and values["prompt"] is None:
        parser.error("--tools requires a prompt or --batch.")

    temp = values["temperature"]
    if temp is not None and (not math.isfinite(temp) or temp < 0):
        parser.error("Temperature must be finite and non-negative.")

    if values["max_tokens"] is not None and values["max_tokens"] <= 0:
        parser.error("max_tokens must be a positive integer.")

    if not (args.batch or args.reset_chat or args.reset_all or args.reset_config) and all(value is None for value in values.values()):
        parser.error("Provide a prompt, --batch, parameters to save, or a reset option.")

    return argparse.Namespace(**values, history=args.history, reset_chat=args.reset_chat,
                              reset_all=args.reset_all, reset_config=args.reset_config, batch=args.batch, structured=args.structured,
                              structured_attempts=args.structured_attempts,
                              structured_mode=args.structured_mode, tools=args.tools)

def main(argv=None):
    args = parse_cli_args(argv)
    try:
        # Loads last chat history from each configured providers, if exist
        history_by_provider = {} if args.reset_all or args.reset_chat else (
            load_history(args.history) if args.prompt is not None and not args.batch else {}
        )


        # Overrides system config params with explicit user's requests if provided
        config_params = dict(DEFAULT_PARAMS) if args.reset_all or args.reset_config else load_config(CONFIG_PATH)
        explicit_params = {name: getattr(args, name) for name in DEFAULT_PARAMS
                    if getattr(args, name) is not None}
        params = config_params | explicit_params

        if args.batch:
            llm = LLMClient()
            structured_options = {}
            if args.structured:
                structured_options = dict(schema_model=SCHEMAS[args.structured],
                                          max_attempts=args.structured_attempts,
                                          native=args.structured_mode == "native")
            if args.tools:
                structured_options["tool_groups"] = args.tools

            # Api call
            reports = llm.generate_batch(args.batch, **params, report=True, **structured_options)

            if args.reset_chat or args.reset_all:
                save_json(args.history, history_by_provider)
            if args.reset_config or args.reset_all:
                save_json(CONFIG_PATH, config_params)
            for index, (prompt, report) in enumerate(zip(args.batch, reports), start=1):
                print(f"[{index}] {prompt}\n{report}\n")
            return

        if args.prompt is None:
            if explicit_params or args.reset_config or args.reset_all:
                save_json(CONFIG_PATH, params)
            if args.reset_chat or args.reset_all:
                save_json(args.history, history_by_provider)
            if explicit_params or args.reset_config or args.reset_all:
                print(f"Configuration saved to {CONFIG_PATH}: {params}")
            if args.reset_chat or args.reset_all:
                print(f"History cleared in {args.history}")
            return
        # Parameters alongside a prompt override only this request.
        llm = LLMClient()
        history = history_by_provider.get(llm.provider, [])
        start = time.perf_counter()
        if args.tools:
            structured_options = {}
            if args.structured:
                structured_options = dict(schema_model=SCHEMAS[args.structured],
                                          max_attempts=args.structured_attempts,
                                          native=args.structured_mode == "native")
            response, updated_history = llm.generate_with_tools(
                args.prompt, **params, history=history, tool_groups=args.tools, **structured_options
            )
        elif args.structured:
            result, response = llm.generate_structured(
                args.prompt, SCHEMAS[args.structured], **params, history=history, include_response=True,
                max_attempts=args.structured_attempts, native=args.structured_mode == "native"
            )
        else:
            response = llm.generate(args.prompt, **params, history=history)
        latency = time.perf_counter() - start
        if args.tools:
            history_by_provider[llm.provider] = updated_history
        else:
            turn = conversation_turn(llm.provider, args.prompt, response)
            history_by_provider[llm.provider] = history + turn
        save_json(args.history, history_by_provider)
        if args.reset_config or args.reset_all:
            save_json(CONFIG_PATH, config_params)
        report = format_response(llm.provider, response, latency)
        if args.structured:
            if args.tools:
                text, _, _ = response_details(llm.provider, response)
                result = SCHEMAS[args.structured].model_validate_json(text)
            report = report.split("Response: ", 1)[0] + "Response: " + result.model_dump_json(indent=2)
        print(report)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
