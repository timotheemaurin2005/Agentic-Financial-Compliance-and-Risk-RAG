"""Retriever node — Pinecone filtered vector search with dual-query support."""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from rag_agent.state import RAGState

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────

_PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
_PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fin-compliance-rag")
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# top_k per query type
_TOP_K = {
    "factual": 10,
    "numerical": 10,
    "comparison": 15,
    "contradiction": 15,
}


# ── Singletons ─────────────────────────────────────────────────────────────

def _get_pinecone_index():
    """Return the Pinecone Index object (lazy singleton)."""
    pc = Pinecone(api_key=_PINECONE_API_KEY)
    return pc.Index(_PINECONE_INDEX_NAME)


def _embed_query(query: str) -> list[float]:
    """Embed the query using the same model as ingestion."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model=_EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_namespace(meeting_date: str, doc_type: str | None = None) -> str:
    """Build a Pinecone namespace string.

    Convention from CLAUDE.md: ``fomc_{meeting_date}`` (e.g. ``fomc_2025-01-29``).
    """
    return f"fomc_{meeting_date}"


def _parse_results(results: dict) -> list[dict]:
    """Flatten Pinecone query results into a list of chunk dicts."""
    chunks: list[dict] = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        chunks.append(
            {
                "id": match["id"],
                "score": match["score"],
                "text": meta.get("text", ""),
                "is_table": meta.get("is_table", False),
                "meeting_date": meta.get("meeting_date", ""),
                "doc_type": meta.get("doc_type", ""),
                "section": meta.get("section", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "source_url": meta.get("source_url", ""),
                "metadata": meta,
            }
        )
    return chunks


def _single_query(
    index: Any,
    query_embedding: list[float],
    top_k: int,
    metadata_filter: dict,
    namespace: str,
) -> list[dict]:
    """Run one Pinecone query and return parsed chunk dicts."""
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=metadata_filter if metadata_filter else None,
        include_metadata=True,
        namespace=namespace,
    )
    return _parse_results(results)


# ── Node function ──────────────────────────────────────────────────────────

def retriever_node(state: RAGState) -> dict:
    """Retrieve relevant chunks from Pinecone with metadata filtering.

    For ``comparison`` / ``contradiction`` queries, runs two separate namespace
    queries (one per meeting date) and merges the results.
    """
    query: str = state["query"]
    query_type: str = state.get("query_type") or "factual"
    metadata_filters: dict = state.get("metadata_filters") or {}
    classification: dict = state.get("_router_classification") or {}  # type: ignore[typeddict-item]

    top_k = _TOP_K.get(query_type, 10)

    logger.info(
        "Retriever — query_type=%s, top_k=%d, filters=%s",
        query_type, top_k, metadata_filters,
    )

    # Embed the query
    query_embedding = _embed_query(query)

    # Get Pinecone index
    index = _get_pinecone_index()

    meeting_dates: list[str] = classification.get("meeting_dates", [])
    doc_types: list[str] = classification.get("doc_types", [])

    all_chunks: list[dict] = []

    # ── Dual-query for comparison / contradiction ──────────────────────
    if query_type in ("comparison", "contradiction") and len(meeting_dates) >= 2:
        for date in meeting_dates:
            # Build a per-date filter (override meeting_date in the filter)
            per_date_filter: dict = {}
            if doc_types:
                per_date_filter["doc_type"] = (
                    {"$eq": doc_types[0]} if len(doc_types) == 1
                    else {"$in": doc_types}
                )
            sections = classification.get("sections", [])
            if sections:
                per_date_filter["section"] = (
                    {"$eq": sections[0]} if len(sections) == 1
                    else {"$in": sections}
                )

            namespace = _build_namespace(date)
            logger.info("Retriever — querying namespace=%s", namespace)

            chunks = _single_query(
                index, query_embedding, top_k, per_date_filter, namespace,
            )
            all_chunks.extend(chunks)
    else:
        # ── Single-query path ──────────────────────────────────────────
        # If we have exactly one meeting date, use it as the namespace
        if len(meeting_dates) == 1:
            namespace = _build_namespace(meeting_dates[0])
            # Remove meeting_date from the filter (namespace handles it)
            single_filter = {k: v for k, v in metadata_filters.items() if k != "meeting_date"}
            # Unwrap $and if it contains meeting_date
            if "$and" in metadata_filters:
                single_filter = {
                    "$and": [
                        c for c in metadata_filters["$and"]
                        if "meeting_date" not in c
                    ]
                }
                if len(single_filter["$and"]) == 1:
                    single_filter = single_filter["$and"][0]
                elif len(single_filter["$and"]) == 0:
                    single_filter = {}
        else:
            namespace = ""
            single_filter = metadata_filters

        logger.info("Retriever — querying namespace=%s", namespace or "(default)")
        all_chunks = _single_query(
            index, query_embedding, top_k, single_filter, namespace,
        )

    # ── Sort by relevance and split into table / text ──────────────────
    all_chunks.sort(key=lambda c: c["score"], reverse=True)

    table_chunks = [c for c in all_chunks if c.get("is_table")]
    text_chunks = [c for c in all_chunks if not c.get("is_table")]

    logger.info(
        "Retriever — total=%d, tables=%d, text=%d",
        len(all_chunks), len(table_chunks), len(text_chunks),
    )

    return {
        "retrieved_chunks": all_chunks,
        "table_chunks": table_chunks,
        "text_chunks": text_chunks,
    }
