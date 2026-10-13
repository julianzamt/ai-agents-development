"""Command-line parsing and persistent conversation workflow."""

import argparse
import math
import time
from pathlib import Path

from .llm.config import DEFAULT_PARAMS, STATE_PATH
from .llm.core import LLMClient
from .llm.responses import conversation_turn, format_response
from .llm.state import load_state, new_state, save_state


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="Consultá un LLM con historial y configuración persistentes.")
    for name, kind in (("prompt", str), ("system", str), ("temperature", float), ("max_tokens", int)):
        parser.add_argument(f"{name}_pos", nargs="?", type=kind, metavar=name)
    parser.add_argument("--prompt")
    parser.add_argument("--batch", nargs="+", metavar="PROMPT",
                        help="De 1 a 3 consultas independientes con la configuración actual")
    parser.add_argument("--system")
    parser.add_argument("--temp", "--temperature", dest="temperature", type=float)
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int)
    parser.add_argument("--history", type=Path, default=STATE_PATH,
                        help="JSON con configuración e historial (default: data/client_state.json)")
    resets = parser.add_mutually_exclusive_group()
    resets.add_argument("--reset", action="store_true", help="Vaciar solo el historial")
    resets.add_argument("--reset-all", action="store_true", help="Vaciar historial y restaurar configuración")
    resets.add_argument("--reset-config", action="store_true", help="Restaurar solo la configuración")
    args = parser.parse_args(argv)
    values = {name: getattr(args, name) if getattr(args, name) is not None
              else getattr(args, f"{name}_pos")
              for name in ("prompt", "system", "temperature", "max_tokens")}
    if values["prompt"] is not None and not values["prompt"].strip():
        parser.error("El prompt no puede estar vacío.")
    if args.batch is not None:
        if len(args.batch) > 3 or any(not prompt.strip() for prompt in args.batch):
            parser.error("--batch acepta de 1 a 3 prompts no vacíos.")
        if values["prompt"] is not None:
            parser.error("--batch no puede combinarse con un prompt individual.")
    temp = values["temperature"]
    if temp is not None and (not math.isfinite(temp) or temp < 0):
        parser.error("La temperatura debe ser finita y no negativa.")
    if values["max_tokens"] is not None and values["max_tokens"] <= 0:
        parser.error("max_tokens debe ser un entero positivo.")
    if not (args.batch or args.reset or args.reset_all or args.reset_config) and all(value is None for value in values.values()):
        parser.error("Indicá un prompt, parámetros para guardar o --reset.")
    return argparse.Namespace(**values, history=args.history, reset=args.reset,
                              reset_all=args.reset_all, reset_config=args.reset_config, batch=args.batch)


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
            print(f"Configuración guardada en {args.history}: {params}")
            return
        # Parameters alongside a prompt override only this request.
        llm = LLMClient()
        history = state["history"].get(llm.provider, [])
        start = time.perf_counter()
        response = llm.generate(args.prompt, **params, history=history)
        latency = time.perf_counter() - start
        turn = conversation_turn(llm.provider, args.prompt, response)
        state["history"][llm.provider] = history + turn
        save_state(args.history, state)
        print(format_response(llm.provider, response, latency))
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
