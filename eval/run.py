"""RAGAS evaluation harness for the Agentic Financial Compliance RAG system.

Loads the eval set, runs each question through the LangGraph agent,
collects answers and retrieved contexts, then computes all 4 RAGAS metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall

Usage:
    python -m eval.run                         # Run full eval
    python -m eval.run --limit 5               # Run first 5 questions only
    python -m eval.run --output results.json   # Custom output path
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA_DIR = Path(__file__).resolve().parent / "data"
EVAL_SET_PATH = EVAL_DATA_DIR / "eval_set.json"
RESULTS_DIR = EVAL_DATA_DIR / "results"


# ── Dataset loader ─────────────────────────────────────────────────────────

def load_eval_set(path: Path | None = None) -> list[dict]:
    """Load the evaluation question set from JSON."""
    p = path or EVAL_SET_PATH
    if not p.exists():
        raise FileNotFoundError(f"Eval set not found at {p}")
    with open(p) as f:
        data = json.load(f)
    logger.info("Loaded %d evaluation questions from %s", len(data), p)
    return data


# ── Agent runner ───────────────────────────────────────────────────────────

def run_agent_query(query: str) -> dict[str, Any]:
    """Run a single query through the LangGraph agent and return the result state.

    Returns dict with keys: final_answer, retrieved_chunks, text_chunks, etc.
    """
    from rag_agent.graph import run_query

    start = time.time()
    result = run_query(query)
    elapsed = time.time() - start

    return {
        "final_answer": result.get("final_answer") or result.get("draft_answer") or "",
        "retrieved_chunks": result.get("retrieved_chunks") or [],
        "text_chunks": result.get("text_chunks") or [],
        "table_chunks": result.get("table_chunks") or [],
        "is_grounded": result.get("is_grounded"),
        "confidence_score": result.get("confidence_score"),
        "contradiction_detected": result.get("contradiction_detected"),
        "cited_sources": result.get("cited_sources") or [],
        "latency_seconds": round(elapsed, 2),
    }


def extract_contexts(agent_result: dict) -> list[str]:
    """Extract retrieved context strings from agent results."""
    contexts: list[str] = []
    for chunk in agent_result.get("retrieved_chunks") or []:
        text = chunk.get("text", "")
        if text:
            contexts.append(text)
    return contexts


# ── RAGAS evaluation ───────────────────────────────────────────────────────

def _get_ragas_llm_and_embeddings():
    """Create RAGAS-compatible LLM and embeddings wrappers.

    Uses gpt-4o with max_tokens=8192 to avoid token-limit errors on
    faithfulness evaluation, and wraps langchain_openai objects so RAGAS
    can call embed_query correctly.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o", max_tokens=8192),
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-large"),
    )
    return llm, embeddings


def compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Compute RAGAS metrics using the v0.4 API.

    Returns a dict of metric_name → score.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Build SingleTurnSample list
    samples = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        samples.append(
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=ctx if ctx else ["No context retrieved."],
                reference=gt,
            )
        )

    dataset = EvaluationDataset(samples=samples)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    llm, embeddings = _get_ragas_llm_and_embeddings()

    logger.info("Running RAGAS evaluation on %d samples...", len(samples))
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    # EvaluationResult._repr_dict holds {metric_name: mean_score}
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        val = result._repr_dict.get(key)
        if val is not None:
            scores[key] = round(float(val), 4)
        else:
            scores[key] = None

    return scores


def compute_per_query_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float | None]:
    """Compute RAGAS metrics for a single question.

    Returns a dict of metric_name → score.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts if contexts else ["No context retrieved."],
        reference=ground_truth,
    )

    dataset = EvaluationDataset(samples=[sample])
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    llm, embeddings = _get_ragas_llm_and_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    # EvaluationResult._repr_dict holds {metric_name: mean_score}
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        val = result._repr_dict.get(key)
        if val is not None:
            scores[key] = round(float(val), 4)
        else:
            scores[key] = None

    return scores


# ── Error analysis ─────────────────────────────────────────────────────────

def classify_failure_mode(
    per_query_scores: dict[str, float | None],
) -> list[str]:
    """Classify failure modes for questions scoring below 0.6 on any metric."""
    failures: list[str] = []
    threshold = 0.6

    if (per_query_scores.get("context_recall") or 1.0) < threshold:
        failures.append("retrieval_miss")
    if (per_query_scores.get("context_precision") or 1.0) < threshold:
        failures.append("noise_retrieval")
    if (per_query_scores.get("faithfulness") or 1.0) < threshold:
        failures.append("hallucination")
    if (per_query_scores.get("answer_relevancy") or 1.0) < threshold:
        failures.append("irrelevant_answer")

    return failures


# ── Main evaluation loop ──────────────────────────────────────────────────

def run_evaluation(
    eval_set: list[dict],
    limit: int | None = None,
    compute_per_query: bool = True,
) -> dict[str, Any]:
    """Run the full evaluation pipeline.

    1. Run each question through the LangGraph agent
    2. Collect answers and contexts
    3. Compute aggregate RAGAS metrics
    4. Optionally compute per-query RAGAS metrics
    5. Perform error analysis on low-scoring questions

    Returns the full experiment results dict.
    """
    if limit:
        eval_set = eval_set[:limit]

    questions: list[str] = []
    answers: list[str] = []
    all_contexts: list[list[str]] = []
    ground_truths: list[str] = []
    per_query_results: list[dict] = []
    total_latency = 0.0

    # ── Step 1: Run agent on all questions ──
    logger.info("=" * 60)
    logger.info("STEP 1: Running %d queries through the RAG agent...", len(eval_set))
    logger.info("=" * 60)

    for i, item in enumerate(eval_set):
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info("[%d/%d] Running q=%s: %s", i + 1, len(eval_set), qid, question[:80])

        try:
            agent_result = run_agent_query(question)
            answer = agent_result["final_answer"]
            contexts = extract_contexts(agent_result)
            latency = agent_result["latency_seconds"]
        except Exception as exc:
            logger.error("Error running query %s: %s", qid, exc)
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
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "num_contexts_retrieved": len(contexts),
            "latency_seconds": latency,
        })

        logger.info(
            "  → answer_len=%d, contexts=%d, latency=%.1fs",
            len(answer), len(contexts), latency,
        )

    # ── Step 2: Compute aggregate RAGAS metrics ──
    logger.info("=" * 60)
    logger.info("STEP 2: Computing aggregate RAGAS metrics...")
    logger.info("=" * 60)

    aggregate_scores = compute_ragas_metrics(
        questions, answers, all_contexts, ground_truths,
    )

    logger.info("Aggregate scores: %s", aggregate_scores)

    # ── Step 3: Per-query RAGAS (optional) ──
    if compute_per_query:
        logger.info("=" * 60)
        logger.info("STEP 3: Computing per-query RAGAS metrics...")
        logger.info("=" * 60)

        for i, item in enumerate(per_query_results):
            logger.info("[%d/%d] RAGAS for %s...", i + 1, len(per_query_results), item["id"])
            try:
                pq_scores = compute_per_query_ragas(
                    questions[i], answers[i], all_contexts[i], ground_truths[i],
                )
            except Exception as exc:
                logger.error("Error computing per-query RAGAS for %s: %s", item["id"], exc)
                pq_scores = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "context_recall": None,
                }

            item.update(pq_scores)

            # Error analysis
            failure_modes = classify_failure_mode(pq_scores)
            if failure_modes:
                item["failure_modes"] = failure_modes
                logger.warning(
                    "  ⚠ %s scored below 0.6: %s", item["id"], failure_modes,
                )

    # ── Build final results ──
    mean_latency = round(total_latency / len(eval_set), 2) if eval_set else 0.0

    config = {
        "chunk_size": int(os.getenv("CHUNK_SIZE", "512")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
        "top_k": 10,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        "use_summary_embedding": os.getenv("USE_SUMMARY_EMBEDDING", "true").lower() == "true",
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o"),
    }

    experiment = {
        "experiment_id": f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "results": {
            **aggregate_scores,
            "mean_latency_seconds": mean_latency,
            "total_questions": len(eval_set),
        },
        "per_query_results": per_query_results,
    }

    return experiment


# ── Report generation ──────────────────────────────────────────────────────

def generate_summary_table(experiment: dict) -> str:
    """Generate a markdown summary table from experiment results."""
    r = experiment["results"]
    c = experiment["config"]

    config_label = (
        f"chunk={c['chunk_size']}/top_k={c['top_k']}"
        f"/embed={c['embedding_model'].split('-')[-1]}"
    )

    table = "| Config | Faithfulness | Relevancy | Precision | Recall | Latency (s) |\n"
    table += "|---|---|---|---|---|---|\n"
    table += (
        f"| {config_label} "
        f"| {r.get('faithfulness', 'N/A')} "
        f"| {r.get('answer_relevancy', 'N/A')} "
        f"| {r.get('context_precision', 'N/A')} "
        f"| {r.get('context_recall', 'N/A')} "
        f"| {r.get('mean_latency_seconds', 'N/A')} |\n"
    )

    return table


def generate_error_report(experiment: dict) -> str:
    """Generate error analysis for questions scoring below 0.6."""
    lines: list[str] = ["## Error Analysis\n"]
    threshold = 0.6
    error_count = 0

    for pq in experiment.get("per_query_results", []):
        failures = pq.get("failure_modes", [])
        if failures:
            error_count += 1
            lines.append(f"### {pq['id']} ({pq['query_type']}, {pq['difficulty']})")
            lines.append(f"**Question:** {pq['question']}")
            lines.append(f"**Failure modes:** {', '.join(failures)}")
            lines.append(f"**Scores:** faith={pq.get('faithfulness')}, "
                         f"relev={pq.get('answer_relevancy')}, "
                         f"prec={pq.get('context_precision')}, "
                         f"recall={pq.get('context_recall')}")
            lines.append("")

    if error_count == 0:
        lines.append("No questions scored below 0.6 on any metric.\n")

    return "\n".join(lines)


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG agent")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N questions (for quick testing)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: eval/data/results/<timestamp>.json)",
    )
    parser.add_argument(
        "--no-per-query", action="store_true",
        help="Skip per-query RAGAS computation (faster but less detailed)",
    )
    parser.add_argument(
        "--eval-set", type=str, default=None,
        help="Path to eval set JSON (default: eval/data/eval_set.json)",
    )
    args = parser.parse_args()

    # Load eval set
    eval_path = Path(args.eval_set) if args.eval_set else None
    eval_set = load_eval_set(eval_path)

    # Run evaluation
    experiment = run_evaluation(
        eval_set,
        limit=args.limit,
        compute_per_query=not args.no_per_query,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_DIR / f"{experiment['experiment_id']}.json"

    with open(output_path, "w") as f:
        json.dump(experiment, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nExperiment: {experiment['experiment_id']}")
    print(f"Timestamp:  {experiment['timestamp']}")
    print(f"Questions:  {experiment['results']['total_questions']}")
    print()

    # Summary table
    print(generate_summary_table(experiment))

    # Targets check
    targets = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.70,
    }
    print("\nTarget Check:")
    for metric, target in targets.items():
        score = experiment["results"].get(metric)
        if score is not None:
            status = "✅" if score >= target else "❌"
            print(f"  {status} {metric}: {score:.4f} (target: {target})")
        else:
            print(f"  ⚠️  {metric}: N/A")

    # Error analysis
    print()
    print(generate_error_report(experiment))

    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
