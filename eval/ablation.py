"""Ablation study — sweep one config variable at a time and log results.

Varies chunk_size, top_k, embedding model, etc. while holding other
variables at their default values. Logs every experiment run as structured
JSON following the schema from eval-skill.md.

Usage:
    python -m eval.ablation                          # Run all sweeps
    python -m eval.ablation --variable top_k         # Sweep only top_k
    python -m eval.ablation --limit 5                # Limit to 5 eval questions
    python -m eval.ablation --dry-run                # Show configs without running
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────

EVAL_DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = EVAL_DATA_DIR / "results"
ABLATION_DIR = RESULTS_DIR / "ablation"


# ── Default configuration ─────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 10,
    "embedding_model": "text-embedding-3-large",
    "use_summary_embedding": True,
    "llm_model": "gpt-4o",
}


# ── Sweep variables from eval-skill.md ─────────────────────────────────────

SWEEP_CONFIGS: dict[str, list[Any]] = {
    "chunk_size": [256, 512, 768, 1024],
    "chunk_overlap": [0, 25, 50, 100],
    "top_k": [5, 10, 15, 20],
    "embedding_model": ["text-embedding-3-small", "text-embedding-3-large"],
    "use_summary_embedding": [True, False],
    "llm_model": ["gpt-4o", "gpt-4o-mini"],
}


# ── Config application ────────────────────────────────────────────────────

def apply_config(config: dict[str, Any]) -> None:
    """Apply a config dict to environment variables so the agent picks them up.

    The LangGraph agent and retriever read from env vars, so we override
    them before each run.
    """
    os.environ["CHUNK_SIZE"] = str(config["chunk_size"])
    os.environ["CHUNK_OVERLAP"] = str(config["chunk_overlap"])
    os.environ["EMBEDDING_MODEL"] = str(config["embedding_model"])
    os.environ["LLM_MODEL"] = str(config["llm_model"])
    os.environ["USE_SUMMARY_EMBEDDING"] = str(config["use_summary_embedding"]).lower()

    # top_k is applied by patching the retriever's top_k dict
    _patch_retriever_top_k(config["top_k"])


def _patch_retriever_top_k(top_k: int) -> None:
    """Dynamically patch the retriever's top_k values for all query types."""
    try:
        from rag_agent.nodes import retriever as retriever_mod
        for key in retriever_mod._TOP_K:
            retriever_mod._TOP_K[key] = top_k
        logger.info("Patched retriever top_k to %d for all query types", top_k)
    except ImportError:
        logger.warning("Could not import retriever module for top_k patching")


def restore_defaults() -> None:
    """Restore environment variables to default config."""
    apply_config(DEFAULT_CONFIG)


# ── Single experiment runner ───────────────────────────────────────────────

def run_single_experiment(
    config: dict[str, Any],
    eval_set: list[dict],
    experiment_id: str,
) -> dict[str, Any]:
    """Run the full evaluation with a given config and return structured results.

    Uses the eval harness from eval.run to execute queries and compute metrics.
    """
    from eval.run import (
        classify_failure_mode,
        compute_ragas_metrics,
        extract_contexts,
        run_agent_query,
    )

    apply_config(config)

    questions: list[str] = []
    answers: list[str] = []
    all_contexts: list[list[str]] = []
    ground_truths: list[str] = []
    per_query_results: list[dict] = []
    total_latency = 0.0

    logger.info("Running %d queries with config: %s", len(eval_set), config)

    for i, item in enumerate(eval_set):
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info("  [%d/%d] %s", i + 1, len(eval_set), qid)

        try:
            agent_result = run_agent_query(question)
            answer = agent_result["final_answer"]
            contexts = extract_contexts(agent_result)
            latency = agent_result["latency_seconds"]
        except Exception as exc:
            logger.error("  Error on %s: %s", qid, exc)
            answer = f"ERROR: {exc}"
            contexts = []
            latency = 0.0

        questions.append(question)
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(ground_truth)
        total_latency += latency

        per_query_results.append({
            "id": qid,
            "query_type": item["query_type"],
            "difficulty": item["difficulty"],
            "latency_seconds": latency,
        })

    # Compute aggregate RAGAS metrics
    logger.info("Computing RAGAS metrics for experiment %s...", experiment_id)
    try:
        aggregate_scores = compute_ragas_metrics(
            questions, answers, all_contexts, ground_truths,
        )
    except Exception as exc:
        logger.error("RAGAS computation failed: %s", exc)
        aggregate_scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

    # Compute per-query RAGAS for error analysis
    from eval.run import compute_per_query_ragas

    for i, pq in enumerate(per_query_results):
        try:
            pq_scores = compute_per_query_ragas(
                questions[i], answers[i], all_contexts[i], ground_truths[i],
            )
            pq.update(pq_scores)
            failure_modes = classify_failure_mode(pq_scores)
            if failure_modes:
                pq["failure_modes"] = failure_modes
        except Exception as exc:
            logger.error("Per-query RAGAS failed for %s: %s", pq["id"], exc)
            pq.update({
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
            })

    mean_latency = round(total_latency / len(eval_set), 2) if eval_set else 0.0

    return {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "results": {
            **aggregate_scores,
            "mean_latency_seconds": mean_latency,
            "total_questions": len(eval_set),
        },
        "per_query_results": per_query_results,
    }


# ── Ablation sweep ────────────────────────────────────────────────────────

def run_ablation(
    variable: str,
    values: list[Any],
    base_config: dict[str, Any],
    eval_set: list[dict],
) -> list[dict]:
    """Sweep one variable while holding others at default.

    Returns a list of experiment result dicts.
    """
    results: list[dict] = []

    logger.info("=" * 60)
    logger.info("ABLATION: Sweeping %s over %s", variable, values)
    logger.info("Base config: %s", base_config)
    logger.info("=" * 60)

    for val in values:
        config = {**base_config, variable: val}
        experiment_id = f"exp_{variable}_{val}"

        logger.info("\n--- Experiment: %s ---", experiment_id)
        experiment = run_single_experiment(config, eval_set, experiment_id)
        results.append(experiment)

        # Log intermediate results
        r = experiment["results"]
        logger.info(
            "Results: faith=%.4f, relev=%.4f, prec=%.4f, recall=%.4f, lat=%.1fs",
            r.get("faithfulness") or 0,
            r.get("answer_relevancy") or 0,
            r.get("context_precision") or 0,
            r.get("context_recall") or 0,
            r.get("mean_latency_seconds") or 0,
        )

    # Restore defaults after sweep
    restore_defaults()

    return results


def run_all_ablations(
    eval_set: list[dict],
    variables: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Run ablation sweeps for all (or specified) variables.

    Returns a dict mapping variable_name → list of experiment results.
    """
    if variables is None:
        variables = list(SWEEP_CONFIGS.keys())

    all_results: dict[str, list[dict]] = {}

    for variable in variables:
        if variable not in SWEEP_CONFIGS:
            logger.warning("Unknown sweep variable: %s — skipping", variable)
            continue

        values = SWEEP_CONFIGS[variable]
        sweep_results = run_ablation(
            variable, values, DEFAULT_CONFIG.copy(), eval_set,
        )
        all_results[variable] = sweep_results

    return all_results


# ── Reporting ──────────────────────────────────────────────────────────────

def generate_ablation_table(all_results: dict[str, list[dict]]) -> str:
    """Generate a markdown summary table from ablation results."""
    lines: list[str] = [
        "| Config | Faithfulness | Relevancy | Precision | Recall | Latency (s) |",
        "|---|---|---|---|---|---|",
    ]

    for variable, experiments in all_results.items():
        for exp in experiments:
            c = exp["config"]
            r = exp["results"]

            # Build config label
            config_val = c[variable]
            if variable == "embedding_model":
                config_val = config_val.split("-")[-1]
            label = f"{variable}={config_val}"

            # Check if this is the default
            if c[variable] == DEFAULT_CONFIG[variable]:
                label += " (default)"

            lines.append(
                f"| {label} "
                f"| {r.get('faithfulness', 'N/A')} "
                f"| {r.get('answer_relevancy', 'N/A')} "
                f"| {r.get('context_precision', 'N/A')} "
                f"| {r.get('context_recall', 'N/A')} "
                f"| {r.get('mean_latency_seconds', 'N/A')} |"
            )
        lines.append("|---|---|---|---|---|---|")

    return "\n".join(lines)


def generate_findings(all_results: dict[str, list[dict]]) -> str:
    """Generate key findings bullet points from ablation results."""
    findings: list[str] = ["## Key Findings\n"]

    for variable, experiments in all_results.items():
        if not experiments:
            continue

        # Find best config for faithfulness
        best_faith = max(
            experiments,
            key=lambda e: e["results"].get("faithfulness") or 0,
        )
        best_val = best_faith["config"][variable]
        best_score = best_faith["results"].get("faithfulness", 0)

        findings.append(
            f"- **{variable}**: Best faithfulness at `{best_val}` "
            f"(score: {best_score:.4f})"
        )

        # Check for faithfulness-recall tradeoff
        if variable == "top_k":
            sorted_by_topk = sorted(experiments, key=lambda e: e["config"]["top_k"])
            faith_trend = [e["results"].get("faithfulness", 0) for e in sorted_by_topk]
            recall_trend = [e["results"].get("context_recall", 0) for e in sorted_by_topk]
            if faith_trend and recall_trend:
                findings.append(
                    f"  - Faithfulness trend as top_k increases: {faith_trend}"
                )
                findings.append(
                    f"  - Context recall trend as top_k increases: {recall_trend}"
                )

    return "\n".join(findings)


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation sweeps on the RAG agent")
    parser.add_argument(
        "--variable", type=str, default=None,
        help="Sweep only this variable (default: all variables)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N eval questions per experiment",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path for combined results",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configs without running experiments",
    )
    parser.add_argument(
        "--eval-set", type=str, default=None,
        help="Path to eval set JSON",
    )
    args = parser.parse_args()

    # Load eval set
    from eval.run import load_eval_set

    eval_path = Path(args.eval_set) if args.eval_set else None
    eval_set = load_eval_set(eval_path)

    if args.limit:
        eval_set = eval_set[:args.limit]
        logger.info("Limited to %d questions", args.limit)

    # Determine which variables to sweep
    variables = [args.variable] if args.variable else None

    if args.dry_run:
        target_vars = variables or list(SWEEP_CONFIGS.keys())
        print("\n🔬 Ablation Sweep Plan (dry run)")
        print("=" * 50)
        print(f"Eval set: {len(eval_set)} questions")
        print(f"Default config: {DEFAULT_CONFIG}")
        print()

        total_experiments = 0
        for var in target_vars:
            vals = SWEEP_CONFIGS.get(var, [])
            total_experiments += len(vals)
            print(f"  {var}: {vals}")

        print(f"\nTotal experiments: {total_experiments}")
        print(f"Estimated queries: {total_experiments * len(eval_set)}")
        return

    # Run ablations
    all_results = run_all_ablations(eval_set, variables)

    # Save combined results
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.output:
        output_path = Path(args.output)
    else:
        sweep_label = args.variable or "all"
        output_path = ABLATION_DIR / f"ablation_{sweep_label}_{timestamp}.json"

    # Flatten all experiments into a single list for the output
    all_experiments = []
    for variable, experiments in all_results.items():
        all_experiments.extend(experiments)

    output_data = {
        "ablation_id": f"ablation_{timestamp}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_config": DEFAULT_CONFIG,
        "variables_swept": list(all_results.keys()),
        "total_experiments": len(all_experiments),
        "experiments": all_experiments,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info("Ablation results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION SWEEP COMPLETE")
    print("=" * 60)
    print(f"\nTotal experiments: {len(all_experiments)}")
    print(f"Variables swept: {list(all_results.keys())}")
    print()

    # Summary table
    print(generate_ablation_table(all_results))
    print()
    print(generate_findings(all_results))
    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
