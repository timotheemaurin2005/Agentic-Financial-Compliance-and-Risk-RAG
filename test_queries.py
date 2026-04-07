"""Quick test of the 3 required queries after retrieval improvements."""

import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

from rag_agent.graph import run_query

QUERIES = [
    ("factual", "What was the target rate after the January 2025 meeting?"),
    ("contradiction", "How did inflation language change between September 2024 and March 2025?"),
    ("comparison", "Compare the risk assessment across all meetings in 2024"),
]

def main():
    for qtype, query in QUERIES:
        print("\n" + "=" * 70)
        print(f"QUERY TYPE: {qtype}")
        print(f"QUESTION: {query}")
        print("=" * 70)

        result = run_query(query)
        answer = result.get("final_answer") or result.get("draft_answer") or "NO ANSWER"
        chunks = result.get("retrieved_chunks") or []
        grounded = result.get("is_grounded")
        confidence = result.get("confidence_score")

        print(f"\nRETRIEVED CHUNKS: {len(chunks)}")
        print(f"GROUNDED: {grounded}")
        print(f"CONFIDENCE: {confidence}")
        print(f"\nANSWER:\n{answer[:1000]}")
        print("-" * 70)

if __name__ == "__main__":
    main()
