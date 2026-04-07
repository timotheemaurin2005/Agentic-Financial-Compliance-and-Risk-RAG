"""Orchestrate the full FOMC ingestion pipeline: download → parse → chunk → embed → upsert."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

from ingestion.chunker import chunk_minutes, chunk_statement
from ingestion.embedder import embed_chunks, generate_summaries
from ingestion.parser import download_all, parse_minutes_with_headers, parse_paragraphs
from ingestion.schemas import MEETING_DATES, Chunk
from ingestion.upserter import ensure_index, upsert_chunks, verify_upsert

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index clearing
# ---------------------------------------------------------------------------

def clear_all_namespaces(meeting_dates: list[str] | None = None) -> None:
    """Delete all vectors from every namespace in the Pinecone index.

    This must be run before re-ingestion when chunking parameters change.
    """
    load_dotenv()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]
    index = pc.Index(index_name)

    dates = meeting_dates or MEETING_DATES
    namespaces = [f"fomc_{d}" for d in dates]

    # Also check actual index stats for any namespaces we might have missed
    try:
        stats = index.describe_index_stats()
        existing_ns = list(stats.get("namespaces", {}).keys())
        for ns in existing_ns:
            if ns not in namespaces:
                namespaces.append(ns)
    except Exception as exc:
        logger.warning("Could not fetch index stats: %s", exc)

    for ns in namespaces:
        try:
            logger.info("Clearing namespace '%s' ...", ns)
            index.delete(delete_all=True, namespace=ns)
            logger.info("  ✅ Cleared '%s'", ns)
        except Exception as exc:
            logger.warning("  ⚠️  Failed to clear '%s': %s", ns, exc)

    logger.info("All namespaces cleared.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    meeting_dates: list[str] | None = None,
    clear_index: bool = False,
) -> None:
    """Run the full ingestion pipeline end-to-end.

    1. (Optional) Clear all Pinecone namespaces
    2. Download HTML from Fed website → save to data/raw/
    3. Parse HTML with BeautifulSoup → extract article content
    4. Detect sections → label each paragraph/block
    5. Chunk text → 400-600 tokens per chunk with section labels
    6. Generate summaries → one-line LLM summary per chunk
    7. Embed → dual vectors (raw + summary) per chunk
    8. Upsert → Pinecone with metadata filters
    9. Verify → test query to confirm retrieval works
    """
    load_dotenv()

    if meeting_dates is None:
        meeting_dates = MEETING_DATES

    # ── Step 0a: Clear index if requested ─────────────────────────────────
    if clear_index:
        logger.info("═══ Step 0a: Clearing Pinecone index ═══")
        clear_all_namespaces(meeting_dates)

    # ── Step 0: Ensure Pinecone index exists ──────────────────────────────
    logger.info("═══ Step 0: Ensuring Pinecone index exists ═══")
    ensure_index()

    # ── Step 1: Download ──────────────────────────────────────────────────
    logger.info("═══ Step 1: Downloading FOMC documents ═══")
    raw_docs = download_all(meeting_dates)

    # ── Step 2–4: Parse & Chunk each document ─────────────────────────────
    logger.info("═══ Steps 2-4: Parsing and chunking ═══")
    all_chunks: list[Chunk] = []

    for date in meeting_dates:
        docs = raw_docs.get(date, {})

        # Statement
        statement_html = docs.get("fomc_statement")
        if statement_html:
            paragraphs = parse_paragraphs(statement_html, doc_type="fomc_statement")
            logger.info("Statement %s: %d paragraphs extracted", date, len(paragraphs))
            stmt_chunks = chunk_statement(paragraphs, date)
            all_chunks.extend(stmt_chunks)
        else:
            logger.warning("No statement HTML for %s — skipping", date)

        # Minutes
        minutes_html = docs.get("fomc_minutes")
        if minutes_html:
            header_paras = parse_minutes_with_headers(minutes_html)
            logger.info("Minutes %s: %d header-paragraphs extracted", date, len(header_paras))
            min_chunks = chunk_minutes(header_paras, date)
            all_chunks.extend(min_chunks)
        else:
            logger.warning("No minutes HTML for %s — skipping", date)

    logger.info("Total chunks created: %d", len(all_chunks))

    if not all_chunks:
        logger.error("No chunks were created — aborting pipeline.")
        return

    # ── Step 5: Generate summaries ────────────────────────────────────────
    logger.info("═══ Step 5: Generating summaries ═══")
    generate_summaries(all_chunks)

    # ── Step 6: Embed ─────────────────────────────────────────────────────
    logger.info("═══ Step 6: Generating dual embeddings ═══")
    raw_embeddings, summary_embeddings = embed_chunks(all_chunks)
    logger.info(
        "Embeddings generated: %d raw, %d summary",
        len(raw_embeddings),
        len(summary_embeddings),
    )

    # ── Step 7: Upsert ────────────────────────────────────────────────────
    logger.info("═══ Step 7: Upserting to Pinecone ═══")
    total_upserted = upsert_chunks(all_chunks, raw_embeddings, summary_embeddings)
    logger.info("Upserted %d vectors total.", total_upserted)

    # ── Step 8: Verify ────────────────────────────────────────────────────
    logger.info("═══ Step 8: Verification ═══")
    # Wait a moment for Pinecone to register the vectors
    import time
    time.sleep(5)

    success = verify_upsert()
    if success:
        logger.info("🎉 Pipeline completed successfully!")
    else:
        logger.warning("⚠️  Pipeline completed but verification had issues.")

    # ── Summary ───────────────────────────────────────────────────────────
    _print_summary(all_chunks)


def _print_summary(chunks: list[Chunk]) -> None:
    """Print a human-readable summary of what was ingested."""
    from collections import Counter

    doc_types = Counter(c.metadata.doc_type for c in chunks)
    sections = Counter(c.metadata.section for c in chunks)
    tables = sum(1 for c in chunks if c.metadata.is_table)
    dates = sorted({c.metadata.meeting_date for c in chunks})

    logger.info("──── Pipeline Summary ────")
    logger.info("Meeting dates processed: %s", dates)
    logger.info("Document types: %s", dict(doc_types))
    logger.info("Sections: %s", dict(sections))
    logger.info("Table chunks: %d", tables)
    logger.info("Total chunks: %d", len(chunks))
    logger.info("Total vectors upserted: %d (2 per chunk)", len(chunks) * 2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ``python -m ingestion.pipeline``.

    Usage:
        python -m ingestion.pipeline                # Normal run
        python -m ingestion.pipeline --clear        # Clear index first, then re-ingest
    """
    import argparse

    parser = argparse.ArgumentParser(description="FOMC ingestion pipeline")
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear all Pinecone namespaces before re-ingesting",
    )
    parser.add_argument(
        "dates", nargs="*", default=None,
        help="Optional meeting dates to process (default: all)",
    )
    args = parser.parse_args()

    dates = args.dates if args.dates else None
    run_pipeline(meeting_dates=dates, clear_index=args.clear)


if __name__ == "__main__":
    main()
