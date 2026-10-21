import argparse
import math
from pathlib import Path

from .schemas import SCHEMAS
from .tools import TOOL_GROUPS
from .llm.config import DEFAULT_PARAMS, CONFIG_PATH, HISTORY_PATH
from .llm.core import LLMClient
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
    parser.add_argument("--history", dest="history_path", type=Path, default=HISTORY_PATH,
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
    if CONFIG_PATH.resolve() == args.history_path.resolve():
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

    return argparse.Namespace(**values, history_path=args.history_path, reset_chat=args.reset_chat,
                              reset_all=args.reset_all, reset_config=args.reset_config, batch=args.batch, structured=args.structured,
                              structured_attempts=args.structured_attempts,
                              structured_mode=args.structured_mode, tools=args.tools)

def prepare_execution(args):
    """Prepare request parameters and persist explicit local changes before calling the API."""
    config_params = dict(DEFAULT_PARAMS) if args.reset_all or args.reset_config else load_config(CONFIG_PATH)
    explicit_params = {name: getattr(args, name) for name in DEFAULT_PARAMS
                       if getattr(args, name) is not None}
    params = config_params | explicit_params

    has_query = args.prompt is not None or args.batch is not None

    # Reset or save new default params
    if args.reset_config or args.reset_all or (explicit_params and not has_query):
        # Overrides supplied with a query apply only to that query.
        saved_params = config_params if has_query else params
        save_json(CONFIG_PATH, saved_params)
        if not has_query:
            print(f"Configuration saved to {CONFIG_PATH}: {saved_params}")

    # Load if needed (single prompts, not used on batches yet), or reset chat history
    if args.reset_chat or args.reset_all:
        history_by_provider = {}
        save_json(args.history_path, history_by_provider)
        if not has_query:
            print(f"History cleared in {args.history_path}")
    elif args.prompt is not None:
        history_by_provider = load_history(args.history_path)
    else:
        # Batches are independent; configuration-only commands do not need history.
        history_by_provider = {}

    return params, history_by_provider


def build_generation_options(args):
    """Build tool and schema options shared by batch and individual requests."""
    options = {}
    if args.structured:
        options.update(schema_model=SCHEMAS[args.structured],
                       max_attempts=args.structured_attempts,
                       native=args.structured_mode == "native")
    if args.tools:
        options["tool_groups"] = args.tools
    return options


def main(argv=None):
    args = parse_cli_args(argv)
    try:
        params, history_by_provider = prepare_execution(args)
        if args.prompt is None and not args.batch:
            return

        llm = LLMClient()
        generation_options = build_generation_options(args)

        # Batch api call
        if args.batch:
            results = llm.generate_batch(args.batch, **params, **generation_options)

            for index, (prompt, result) in enumerate(zip(args.batch, results), start=1):
                print(f"[{index}] {prompt}\n{result.report}\n")
            return

        # Single prompt api call
        history = history_by_provider.get(llm.provider, [])
        result = llm.generate(args.prompt, **params, history=history, **generation_options)
        history_by_provider[llm.provider] = result.history
        save_json(args.history_path, history_by_provider)
        print(result.report)

    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
