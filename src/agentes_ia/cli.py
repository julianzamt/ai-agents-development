"""Command-line parsing and persistent conversation workflow."""

import argparse
import math
import time
from pathlib import Path

from .schemas import SupportQuery
from .llm.config import DEFAULT_PARAMS, STATE_PATH
from .llm.core import LLMClient
from .llm.responses import conversation_turn, format_response
from .llm.state import load_state, new_state, save_state


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="Query an LLM with persistent history and configuration.")
    for name, kind in (("prompt", str), ("system", str), ("temperature", float), ("max_tokens", int)):
        parser.add_argument(f"{name}_pos", nargs="?", type=kind, metavar=name)
    parser.add_argument("--prompt")
    parser.add_argument("--structured", choices=["support"],
                        help="Return a validated SupportQuery using native structured output")
    parser.add_argument("--structured-attempts", type=int, default=3,
                        help="Maximum structured validation attempts (default: 3)")
    parser.add_argument("--structured-mode", choices=["native", "prompt"], default="native",
                        help="Native schema enforcement or universal prompt-only JSON")
    parser.add_argument("--batch", nargs="+", metavar="PROMPT",
                        help="Run 1 to 3 independent queries using the current configuration")
    parser.add_argument("--system")
    parser.add_argument("--temp", "--temperature", dest="temperature", type=float)
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int)
    parser.add_argument("--history", type=Path, default=STATE_PATH,
                        help="Configuration and history JSON (default: data/client_state.json)")
    resets = parser.add_mutually_exclusive_group()
    resets.add_argument("--reset", action="store_true", help="Clear history only")
    resets.add_argument("--reset-all", action="store_true", help="Clear history and restore default configuration")
    resets.add_argument("--reset-config", action="store_true", help="Restore default configuration only")
    args = parser.parse_args(argv)
    values = {name: getattr(args, name) if getattr(args, name) is not None
              else getattr(args, f"{name}_pos")
              for name in ("prompt", "system", "temperature", "max_tokens")}
    if values["prompt"] is not None and not values["prompt"].strip():
        parser.error("The prompt must not be empty.")
    if args.batch is not None:
        if len(args.batch) > 3 or any(not prompt.strip() for prompt in args.batch):
            parser.error("--batch accepts 1 to 3 non-empty prompts.")
        if values["prompt"] is not None:
            parser.error("--batch cannot be combined with an individual prompt.")
    if args.structured and (args.batch or values["prompt"] is None):
        parser.error("--structured requires an individual prompt and cannot be used with --batch.")
    if args.structured_attempts <= 0:
        parser.error("--structured-attempts must be positive.")
    temp = values["temperature"]
    if temp is not None and (not math.isfinite(temp) or temp < 0):
        parser.error("Temperature must be finite and non-negative.")
    if values["max_tokens"] is not None and values["max_tokens"] <= 0:
        parser.error("max_tokens must be a positive integer.")
    if not (args.batch or args.reset or args.reset_all or args.reset_config) and all(value is None for value in values.values()):
        parser.error("Provide a prompt, --batch, parameters to save, or a reset option.")
    return argparse.Namespace(**values, history=args.history, reset=args.reset,
                              reset_all=args.reset_all, reset_config=args.reset_config, batch=args.batch, structured=args.structured,
                              structured_attempts=args.structured_attempts,
                              structured_mode=args.structured_mode)


def main(argv=None):
    args = parse_cli_args(argv)
    try:
        state = new_state() if args.reset_all else load_state(args.history)
        if args.reset:
            state["history"] = {}
        if args.reset_config:
            state["params"] = dict(DEFAULT_PARAMS)
        explicit = {name: getattr(args, name) for name in DEFAULT_PARAMS
                    if getattr(args, name) is not None}
        params = state["params"] | explicit
        if args.batch:
            llm = LLMClient()
            reports = llm.generate_batch(args.batch, **params, report=True)
            if args.reset or args.reset_all or args.reset_config:
                save_state(args.history, state)
            for index, (prompt, report) in enumerate(zip(args.batch, reports), start=1):
                print(f"[{index}] {prompt}\n{report}\n")
            return
        if args.prompt is None:
            state["params"] = params
            save_state(args.history, state)
            print(f"Configuration saved to {args.history}: {params}")
            return
        # Parameters alongside a prompt override only this request.
        llm = LLMClient()
        history = state["history"].get(llm.provider, [])
        start = time.perf_counter()
        if args.structured:
            result, response = llm.generate_structured(
                args.prompt, SupportQuery, **params, history=history, include_response=True,
                max_attempts=args.structured_attempts, native=args.structured_mode == "native"
            )
        else:
            response = llm.generate(args.prompt, **params, history=history)
        latency = time.perf_counter() - start
        turn = conversation_turn(llm.provider, args.prompt, response)
        state["history"][llm.provider] = history + turn
        save_state(args.history, state)
        report = format_response(llm.provider, response, latency)
        if args.structured:
            report = report.split("Response: ", 1)[0] + "Response: " + result.model_dump_json(indent=2)
        print(report)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
