"""Diagnose a RAG pipeline: check one query, measure retrieval metrics, cost, latency, failing stage or judge scores."""

import argparse
import os
import statistics
import sys
from pathlib import Path

import chromadb
from chromadb.errors import ChromaError

from ..llm import LLMClient
from ..llm.config import PROJECT_ROOT
from .config import DEFAULT_CONFIG_PATH, add_config_argument, config_for
from .diagnose import (
    abstention_report, cost_report, diagnose_retrieval, judge_runs, latency_report, load_eval_dataset, retrieval_metrics,
    run_cases, stage_report,
)


def print_result(query: str, expected_chunk_id: str, result: dict) -> None:
    print(f"\nQuery: {query}")
    if result["ok"]:
        print(f"OK: {expected_chunk_id} is rank {result['rank']} of {result['k']} "
              f"(distance={result['distance']:.4f})")
    else:
        print(f"FAIL: {expected_chunk_id} is not in the top {result['k']}")
    for rank, chunk in enumerate(result["chunks"], start=1):
        marker = "*" if chunk["chunk_id"] == expected_chunk_id else " "
        print(f"  {marker}{rank}. distance={chunk['distance']:.4f} id={chunk['chunk_id']}")


STYLES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36",
          "orange": "38;5;208"}


def paint(text: str, *styles: str) -> str:
    """Wrap text in ANSI styles when stdout is a terminal and NO_COLOR is unset."""
    if not styles or "NO_COLOR" in os.environ or not sys.stdout.isatty():
        return text
    return f"\033[{';'.join(STYLES[s] for s in styles)}m{text}\033[0m"


def beyond(distance, max_distance) -> bool:
    return max_distance is not None and distance is not None and distance > max_distance


def print_chunks(case: dict, max_distance: float | None, relevant=()) -> None:
    """Retrieved chunks with rank and distance; * marks relevant, x marks beyond max_distance."""
    for rank, (chunk_id, distance) in enumerate(zip(case["retrieved"], case["distances"]), start=1):
        is_relevant, far = chunk_id in relevant, beyond(distance, max_distance)
        marker = paint("*", "green", "bold") if is_relevant else " "
        shown = "   n/a" if distance is None else f"{distance:.4f}"
        shown = paint(shown, "yellow") if far else shown
        flag = paint(" x", "yellow") if far else "  "
        print(f"      {marker}{rank}. {shown}{flag} {paint(chunk_id, 'green' if is_relevant else 'dim')}")


def print_cases(result: dict, only_failures: bool = False, max_distance: float | None = None) -> None:
    print(paint(f"\nk={result['k']}", "bold", "cyan")
          + paint(f"  (distance; x = beyond max_distance={max_distance})", "dim"))
    shown = 0
    for number, case in enumerate(result["per_case"], start=1):
        if only_failures and case["recall"] == 1:
            continue
        shown += 1
        status = paint("HIT ", "green", "bold") if case["hit"] else paint("MISS", "red", "bold")
        print(f"\n{number:>3}. " + paint(case["question"], "bold"))
        print(f"     {status} recall={case['recall']:.2f}  precision={case['precision']:.2f}  "
              f"rr={case['rr']:.2f}  " + paint(f"[{case.get('kind', 'unspecified')}]", "dim"))
        print_chunks(case, max_distance, case["relevant"])
        for chunk_id in case["relevant"]:
            if chunk_id not in case["retrieved"]:
                print("       " + paint(f"missing: {chunk_id}", "red"))
    if only_failures and not shown:
        print(paint("  No failures: every relevant chunk was retrieved.", "green"))
    unanswerable = [case for case in result.get("unanswerable", [])
                    if not only_failures or not all(beyond(d, max_distance) for d in case["distances"][:1])]
    if unanswerable:
        print(paint("\nMust not answer" + (" (top-1 within max_distance)" if only_failures else "") + ":",
                    "bold", "cyan"))
    for case in unanswerable:
        print("\n  - " + paint(case["question"], "bold") + "  " + paint(f"[{case['kind']}]", "dim"))
        print_chunks(case, max_distance)
    print()


def describe(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (f"n={len(values):<3} min={min(values):.4f}  median={statistics.median(values):.4f}  "
            f"max={max(values):.4f}")


def print_distances(result: dict, max_distance: float | None) -> None:
    """Where relevant chunks and must-not-answer questions fall relative to the threshold."""
    relevant = [d for case in result["per_case"]
                for chunk_id, d in zip(case["retrieved"], case["distances"])
                if chunk_id in case["relevant"] and d is not None]
    top1 = [case["distances"][0] for case in result.get("unanswerable", [])
            if case["distances"] and case["distances"][0] is not None]
    print(paint(f"\nDistances at k={result['k']} (max_distance={max_distance}):", "bold", "cyan"))
    line = f"  relevant chunks retrieved:  {describe(relevant)}"
    if max_distance is not None and relevant:
        count = sum(d > max_distance for d in relevant)
        line += "  " + paint(f"beyond threshold: {count}", "yellow" if count else "green")
    print(line)
    if result.get("unanswerable"):
        line = f"  top-1 of must-not-answer:   {describe(top1)}"
        if max_distance is not None and top1:
            count = sum(d <= max_distance for d in top1)
            line += "  " + paint(f"within threshold: {count}", "yellow" if count else "green")
        print(line)


def print_table(rows: list[tuple[str, dict]], label: str) -> None:
    print(paint(f"{label:<15} {'cases':>5} {'hit_rate':>9} {'recall':>8} {'precision':>10} {'mrr':>7}", "bold"))
    for name, r in rows:
        print(f"{name:<15} {r['cases']:>5} {r['hit_rate']:>9.3f} {r['recall']:>8.3f} "
              f"{r['precision']:>10.3f} {r['mrr']:>7.3f}")


def print_metrics(results: list[dict], verbose: bool = False, only_failures: bool = False,
                  max_distance: float | None = None, threshold_source: str | None = None) -> None:
    if verbose or only_failures:
        for result in results:
            print_cases(result, only_failures=only_failures, max_distance=max_distance)
    skipped = len(results[0].get("unanswerable", []))
    print(f"Cases with relevant chunks: {results[0]['cases']}"
          + (f" ({skipped} must-not-answer cases not scored)" if skipped else ""))
    source = f" (from {threshold_source})" if threshold_source else ""
    print(paint(f"Distance threshold (max_distance): {max_distance}", "bold") + paint(source, "dim"))
    print_table([(f"k={r['k']}", r) for r in results], "all")
    if "by_kind" in results[0]:
        print(paint("\nBy kind:", "bold", "cyan"))
        for kind in results[0]["by_kind"]:
            print_table([(f"k={r['k']}", r["by_kind"][kind]) for r in results], kind)
    if "distances" in results[0]["per_case"][0]:
        print_distances(max(results, key=lambda r: r["k"]), max_distance)


# Subcommands that answer every answerable dataset question with rag().
ANSWERING = ("cost", "latency", "stages", "judge", "abstention")


def print_abstention(result: dict, verbose: bool = False) -> None:
    labels = {"no_evidence": paint("NO EVIDENCE", "green"), "abstained": paint("ABSTAINED  ", "green"),
              "answered": paint("ANSWERED   ", "red", "bold")}
    if verbose:
        for number, case in enumerate(result["per_case"], start=1):
            distance = "n/a" if case["top_distance"] is None else f"{case['top_distance']:.4f}"
            print(f"{number:>3}. {labels.get(case['outcome'], case['outcome'])} top-1={distance}  "
                  + paint(case["question"], "bold") + "  " + paint(f"[{case['kind']}]", "dim"))
            if case["outcome"] == "answered":
                answer = " ".join(case["answer"].split())
                print("       " + paint(answer[:300] + ("…" if len(answer) > 300 else ""), "dim"))
                if case["cited"]:
                    print("       " + paint("cited: " + ", ".join(case["cited"]), "dim"))
        print()
    outcomes = result["outcomes"]
    rate = result["abstention_rate"]
    print(f"Cases that must not be answered: {result['cases']} | "
          + ", ".join(f"{name}={count}" for name, count in outcomes.items()))
    print(paint(f"abstention rate: {rate:.3f}", "bold", "green" if rate == 1 else "yellow"))
    for kind, entry in result["by_kind"].items():
        print(f"  {kind:<15} {entry['cases']:>3} cases  abstention rate: {entry['abstention_rate']:.3f}")


def print_block(title: str, text: str, style: str | None = None) -> None:
    """A titled section with its text indented below it."""
    print("\n     " + paint(f"── {title} " + "─" * max(0, 60 - len(title)), "bold", "cyan"))
    for line in text.splitlines() or [""]:
        print("       " + (paint(line, style) if style else line))


def score_style(score: int) -> str:
    """Green for 4-5, orange for 3, red for 1-2."""
    return "green" if score >= 4 else "orange" if score == 3 else "red"


def print_judgments(result: dict, verbose: bool = False) -> None:
    if verbose:
        for number, case in enumerate(result["per_case"], start=1):
            print(paint(f"\n{'=' * 72}", "dim"))
            print(f"{number:>3}. " + paint(case["question"], "bold") + "  " + paint(f"[{case['outcome']}]", "dim"))
            if case.get("prompt"):
                print_block("Prompt", case["prompt"], "dim")
            print_block("Model response", case["answer"])
            judgment = case["judgment"]
            if judgment:
                scores = "  ".join(
                    paint(f"{name} {value}/5", score_style(value), "bold")
                    for name, value in (("faithfulness", judgment.faithfulness), ("relevance", judgment.relevance)))
                print_block("Judge", scores)
                print("       " + judgment.justification)
            else:
                print_block("Judge", paint(f"not judged ({case['outcome']})", "dim"))
        print()
    print(f"Cases that must be answered: {result['cases']} | answered: {result['answered']}")
    if result["faithfulness"] is not None:
        print(f"faithfulness: {result['faithfulness']:.2f}/5  relevance: {result['relevance']:.2f}/5")


def print_cost(result: dict, verbose: bool = False) -> None:
    def money(value):
        return f"${value:.5f}" if value is not None else "n/a"

    if verbose:
        for number, case in enumerate(result["per_case"], start=1):
            print(f"{number:>3}. {case['input_tokens']:>6} in {case['output_tokens']:>5} out "
                  f"{money(case['cost']):>10}  {case['question']}")
        print()
    print(f"Cases: {result['cases']} | tokens: {result['input_tokens']} in, {result['output_tokens']} out")
    if result["cost"] is None:
        print("cost: n/a (set pricing in the config)")
    else:
        print(f"cost: {money(result['cost'])} total, {money(result['cost'] / result['cases'])} per case")


def print_latency(result: dict, verbose: bool = False) -> None:
    if verbose:
        for number, case in enumerate(result["per_case"], start=1):
            print(f"{number:>3}. {case['latency']:>6.2f}s  {case['question']}")
        print()
    print(f"Cases: {result['cases']} | mean: {result['mean']:.2f}s  median: {result['median']:.2f}s  "
          f"max: {result['max']:.2f}s")


def print_stages(result: dict, verbose: bool = False) -> None:
    if verbose:
        for number, case in enumerate(result["per_case"], start=1):
            error = case["error"]
            status = f"{error['stage'].upper()}: {error['detail']}" if error else "ok"
            print(f"{number:>3}. {status}  {case['question']}")
        print()
    errors = result["errors"]
    print(f"Cases: {result['cases']} | " + ", ".join(f"{stage}={count}" for stage, count in errors.items())
          + f" | no error: {result['no_error']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    config = config_for(parser, argv)
    k = config["retrieval"]["k"]
    retrieval = config["retrieval"]
    dataset = config["eval_dataset"]

    common = argparse.ArgumentParser(add_help=False)
    add_config_argument(common)
    common.add_argument(
        "--persist-directory", type=Path, default=Path(config["persist_directory"]),
        help=f"ChromaDB persistence directory (default: {config['persist_directory']})",
    )
    common.add_argument("--collection", default=config["collection"], help=f"ChromaDB collection name (default: {config['collection']})")

    def dataset_options(default: str | None, source: str) -> argparse.ArgumentParser:
        options = argparse.ArgumentParser(add_help=False)
        options.add_argument(
            "--dataset", type=Path, default=PROJECT_ROOT / default if default else None,
            help=f"Evaluation dataset JSON (default: {default}, {source} in the config)",
        )
        options.add_argument("-v", "--verbose", action="store_true", help="Also list every case")
        return options

    with_dataset = dataset_options(dataset, "eval_dataset")
    # The judge adds an LLM call per answer, so it can default to a smaller dataset.
    judge_dataset = (dataset_options(config["judge_dataset"], "judge_dataset") if config["judge_dataset"]
                     else with_dataset)

    # Measurements that answer every answerable dataset question with rag() share these settings.
    answering = argparse.ArgumentParser(add_help=False)
    answering.add_argument("-k", type=int, default=k, help=f"Chunks to retrieve (default: {k})")
    answering.add_argument("--max-distance", type=float, default=retrieval["max_distance"],
                           help=f"Relevance threshold (default: {retrieval['max_distance']}, None keeps all)")
    answering.add_argument("--max-prompt-chars", type=int, default=retrieval["max_prompt_chars"],
                           help=f"Budget for the whole prompt (default: {retrieval['max_prompt_chars']})")

    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", parents=[common], help="Check that one query retrieves its expected chunk")
    check.add_argument("query", help="The question to retrieve for")
    check.add_argument("chunk_id", help="ID of the chunk expected in the results")
    check.add_argument("-k", type=int, default=k, help=f"Chunks to retrieve (default: {k})")

    metrics = commands.add_parser("metrics", parents=[common, with_dataset], help="Measure hit rate, recall, precision and MRR on the evaluation dataset")
    metrics.add_argument("-k", type=int, nargs="+", default=[k],
                         help=f"Top-k values to measure, e.g. -k 1 3 5 (default: {k})")
    metrics.add_argument("--max-distance", type=float,
                         help=f"Threshold to compare distances against (default: {retrieval['max_distance']}, "
                              "from the config)")
    metrics.add_argument("--only-failures", action="store_true",
                         help="List only the cases that missed a relevant chunk (recall below 1) and the "
                              "must-not-answer cases whose top chunk is within --max-distance")

    for name, summary in (
        ("cost", "Tokens and cost of answering the dataset (needs pricing in the config for USD)"),
        ("latency", "Time rag() takes to answer each dataset question"),
        ("stages", "Count the dataset questions that fail at retrieval, context or generation"),
        ("judge", "Score the answers' faithfulness and relevance with an LLM judge"),
        ("abstention", "Check that the questions that must not be answered are declined"),
    ):
        datasets = judge_dataset if name == "judge" else with_dataset
        commands.add_parser(name, parents=[common, datasets, answering], help=summary)
    args = parser.parse_args(argv)

    ks = args.k if args.command == "metrics" else [args.k]
    if any(value <= 0 for value in ks):
        parser.error("-k must be greater than zero.")
    if args.command in ANSWERING and (args.max_prompt_chars <= 0 or (args.max_distance is not None and args.max_distance < 0)):
        parser.error("--max-prompt-chars must be positive and --max-distance non-negative.")
    if args.command != "check" and args.dataset is None:
        parser.error("No evaluation dataset: pass --dataset or set eval_dataset in the config.")

    client = chromadb.PersistentClient(path=str(args.persist_directory))
    try:
        collection = client.get_collection(name=args.collection)
    except Exception as exc:
        parser.error(f"Could not open collection {args.collection!r}: {exc}")

    try:
        if args.command == "check":
            result = diagnose_retrieval(args.query, args.chunk_id, collection, k=args.k)
        elif args.command == "metrics":
            cases = load_eval_dataset(args.dataset)
            results = [retrieval_metrics(cases, collection, k=value) for value in args.k]
        else:
            llm = LLMClient()
            runs = run_cases(
                load_eval_dataset(args.dataset), collection, llm, k=args.k,
                max_distance=args.max_distance, max_prompt_chars=args.max_prompt_chars,
                answerable=args.command != "abstention",
            )
            if args.command == "cost":
                outcome, show = cost_report(runs, config["pricing"]), print_cost
            elif args.command == "latency":
                outcome, show = latency_report(runs), print_latency
            elif args.command == "stages":
                outcome, show = stage_report(runs), print_stages
            elif args.command == "abstention":
                outcome, show = abstention_report(runs), print_abstention
            else:
                outcome, show = judge_runs(runs, llm), print_judgments
    except (ValueError, ChromaError) as exc:
        parser.error(str(exc))

    if args.command == "check":
        print_result(args.query, args.chunk_id, result)
        sys.exit(0 if result["ok"] else 1)
    if args.command == "metrics":
        if args.max_distance is None:
            default = DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)
            threshold, source = retrieval["max_distance"], args.config or default
        else:
            threshold, source = args.max_distance, "--max-distance"
        print_metrics(results, verbose=args.verbose, only_failures=args.only_failures,
                      max_distance=threshold, threshold_source=source)
    else:
        show(outcome, verbose=args.verbose)


if __name__ == "__main__":
    main()
