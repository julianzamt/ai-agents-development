"""Diagnose a RAG pipeline: check one query, measure retrieval metrics, cost, latency, failing stage or judge scores."""

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.errors import ChromaError

from ..llm import LLMClient
from ..llm.config import PROJECT_ROOT
from .config import add_config_argument, config_for
from .diagnose import (
    cost_report, diagnose_retrieval, judge_runs, latency_report, load_eval_dataset, retrieval_metrics,
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


def print_cases(result: dict) -> None:
    print(f"\nk={result['k']}")
    for number, case in enumerate(result["per_case"], start=1):
        status = "HIT " if case["hit"] else "MISS"
        print(f"{number:>3}. {status} recall={case['recall']:.2f} precision={case['precision']:.2f} "
              f"rr={case['rr']:.2f}  {case['question']}")
        for rank, chunk_id in enumerate(case["retrieved"], start=1):
            marker = "*" if chunk_id in case["relevant"] else " "
            print(f"       {marker}{rank}. {chunk_id}")
        for chunk_id in case["relevant"]:
            if chunk_id not in case["retrieved"]:
                print(f"       missing: {chunk_id}")
    print()


def print_metrics(results: list[dict], verbose: bool = False) -> None:
    if verbose:
        for result in results:
            print_cases(result)
    print(f"Cases with relevant chunks: {results[0]['cases']}")
    print(f"{'k':>3} {'hit_rate':>9} {'recall':>8} {'precision':>10} {'mrr':>7}")
    for r in results:
        print(f"{r['k']:>3} {r['hit_rate']:>9.3f} {r['recall']:>8.3f} {r['precision']:>10.3f} {r['mrr']:>7.3f}")


# Subcommands that answer every answerable dataset question with rag().
ANSWERING = ("cost", "latency", "stages", "judge")


def print_judgments(result: dict, verbose: bool = False) -> None:
    if verbose:
        for number, case in enumerate(result["per_case"], start=1):
            judgment = case["judgment"]
            scores = (f"faithfulness={judgment.faithfulness} relevance={judgment.relevance}"
                      if judgment else f"not judged ({case['outcome']})")
            print(f"{number:>3}. {scores}  {case['question']}")
            if judgment:
                print(f"       {judgment.justification}")
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

    with_dataset = argparse.ArgumentParser(add_help=False)
    with_dataset.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / dataset if dataset else None,
        help=f"Evaluation dataset JSON (default: {dataset})",
    )
    with_dataset.add_argument("-v", "--verbose", action="store_true", help="Also list every case")

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

    for name, summary in (
        ("cost", "Tokens and cost of answering the dataset (needs pricing in the config for USD)"),
        ("latency", "Time rag() takes to answer each dataset question"),
        ("stages", "Count the dataset questions that fail at retrieval, context or generation"),
        ("judge", "Score the answers' faithfulness and relevance with an LLM judge"),
    ):
        commands.add_parser(name, parents=[common, with_dataset, answering], help=summary)
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
            )
            if args.command == "cost":
                outcome, show = cost_report(runs, config["pricing"]), print_cost
            elif args.command == "latency":
                outcome, show = latency_report(runs), print_latency
            elif args.command == "stages":
                outcome, show = stage_report(runs), print_stages
            else:
                outcome, show = judge_runs(runs, llm), print_judgments
    except (ValueError, ChromaError) as exc:
        parser.error(str(exc))

    if args.command == "check":
        print_result(args.query, args.chunk_id, result)
        sys.exit(0 if result["ok"] else 1)
    if args.command == "metrics":
        print_metrics(results, verbose=args.verbose)
    else:
        show(outcome, verbose=args.verbose)


if __name__ == "__main__":
    main()
