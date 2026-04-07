"""Retriever node — Pinecone filtered vector search with dual-query support.

Improvements over baseline:
- Cross-namespace retrieval for comparison/contradiction queries
- Query expansion via LLM rewrites (3 rewrites + original = 4 embeddings)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from ingestion.schemas import MEETING_DATES
from rag_agent.state import RAGState

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────

_PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
_PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fin-compliance-rag")
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# top_k per query type
_TOP_K = {
    "factual": 15,
    "numerical": 15,
    "comparison": 25,
    "contradiction": 25,
}

# All known namespaces (one per meeting date)
_ALL_NAMESPACES = [f"fomc_{d}" for d in MEETING_DATES]

# Query expansion prompt
_QUERY_EXPANSION_PROMPT = (
    "Rewrite the following question about FOMC monetary policy "
    "in 3 different ways, keeping the same meaning but using "
    "different terminology. Return only the 3 rewrites, one per line.\n\n"
    "Question: {query}"
)


# ── Singletons ─────────────────────────────────────────────────────────────

def _get_pinecone_index():
    """Return the Pinecone Index object (lazy singleton)."""
    pc = Pinecone(api_key=_PINECONE_API_KEY)
    return pc.Index(_PINECONE_INDEX_NAME)


def _get_openai_client() -> OpenAI:
    """Return OpenAI client."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _embed_query(query: str) -> list[float]:
    """Embed the query using the same model as ingestion."""
    client = _get_openai_client()
    response = client.embeddings.create(model=_EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


def _embed_queries(queries: list[str]) -> list[list[float]]:
    """Embed multiple queries in a single API call."""
    client = _get_openai_client()
    response = client.embeddings.create(model=_EMBEDDING_MODEL, input=queries)
    # Sort by index to guarantee order
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [d.embedding for d in sorted_data]


# ── Query expansion ───────────────────────────────────────────────────────

def _expand_query(query: str) -> list[str]:
    """Generate 3 query rewrites using the LLM.

    Returns a list of 4 strings: [original, rewrite1, rewrite2, rewrite3].
    """
    client = _get_openai_client()
    try:
        response = client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite monetary policy questions using different terminology. Return exactly 3 rewrites, one per line. No numbering, no bullets, no extra text.",
                },
                {"role": "user", "content": _QUERY_EXPANSION_PROMPT.format(query=query)},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
        rewrites = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        # Take up to 3 rewrites
        rewrites = rewrites[:3]
        logger.info("Query expansion — %d rewrites generated", len(rewrites))
    except Exception as exc:
        logger.warning("Query expansion failed: %s — using original only", exc)
        rewrites = []

    return [query] + rewrites


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


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate chunks by ID, keeping the highest score for each."""
    seen: dict[str, dict] = {}
    for chunk in chunks:
        cid = chunk["id"]
        if cid not in seen or chunk["score"] > seen[cid]["score"]:
            seen[cid] = chunk
    return list(seen.values())


# ── Node function ──────────────────────────────────────────────────────────

def _strip_section_from_filter(filt: dict) -> dict:
    """Remove the 'section' constraint from a Pinecone filter dict."""
    if not filt:
        return {}
    if "section" in filt:
        out = {k: v for k, v in filt.items() if k != "section"}
        return out
    if "$and" in filt:
        clauses = [c for c in filt["$and"] if "section" not in c]
        if len(clauses) == 0:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
    return filt


def _strip_doctype_from_filter(filt: dict) -> dict:
    """Remove the 'doc_type' constraint from a Pinecone filter dict."""
    if not filt:
        return {}
    if "doc_type" in filt:
        out = {k: v for k, v in filt.items() if k != "doc_type"}
        return out
    if "$and" in filt:
        clauses = [c for c in filt["$and"] if "doc_type" not in c]
        if len(clauses) == 0:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
    return filt


def retriever_node(state: RAGState) -> dict:
    """Retrieve relevant chunks from Pinecone with metadata filtering.

    Improvements over baseline:
    - Query expansion: 3 LLM rewrites + original → 4 embeddings
    - Cross-namespace: secondary sweep for comparison/contradiction queries
    - Fallback broadening: if filtered search returns 0, progressively relax filters
    - Multi-meeting numerical: queries with multiple dates get multi-namespace search
    - Deduplication: merge all results, keep highest score per chunk ID
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

    # ── Step 1: Query expansion ────────────────────────────────────────
    expanded_queries = _expand_query(query)
    logger.info("Retriever — expanded to %d queries", len(expanded_queries))

    # Embed all expanded queries in one batch
    all_embeddings = _embed_queries(expanded_queries)

    # Get Pinecone index
    index = _get_pinecone_index()

    meeting_dates: list[str] = classification.get("meeting_dates", [])
    doc_types: list[str] = classification.get("doc_types", [])
    sections: list[str] = classification.get("sections", [])

    all_chunks: list[dict] = []

    # ── Helper: build per-date filter ──────────────────────────────────
    def _per_date_filter(include_sections: bool = True) -> dict:
        """Build a filter for per-namespace queries (no meeting_date since
        namespaces handle that)."""
        f: dict = {}
        if doc_types:
            f["doc_type"] = (
                {"$eq": doc_types[0]} if len(doc_types) == 1
                else {"$in": doc_types}
            )
        if include_sections and sections:
            f["section"] = (
                {"$eq": sections[0]} if len(sections) == 1
                else {"$in": sections}
            )
        return f

    # ── Determine if this is a multi-meeting query ─────────────────────
    is_multi = (
        query_type in ("comparison", "contradiction")
        or (query_type in ("factual", "numerical") and len(meeting_dates) >= 2)
    )

    if is_multi and len(meeting_dates) >= 2:
        # ── Multi-meeting path ─────────────────────────────────────────
        # Query each meeting namespace with all expanded embeddings
        for date in meeting_dates:
            namespace = _build_namespace(date)
            pf = _per_date_filter(include_sections=True)
            logger.info("Retriever — querying namespace=%s filter=%s", namespace, pf)

            for emb in all_embeddings:
                chunks = _single_query(index, emb, top_k, pf, namespace)
                all_chunks.extend(chunks)

            # If section-filtered query returned nothing for this date, retry without section
            date_chunks = [c for c in all_chunks if c.get("meeting_date") == date]
            if not date_chunks and sections:
                broad_f = _per_date_filter(include_sections=False)
                logger.info("Retriever — fallback: dropping section filter for %s", date)
                for emb in all_embeddings[:2]:  # original + 1 rewrite
                    chunks = _single_query(index, emb, top_k, broad_f, namespace)
                    all_chunks.extend(chunks)

        # ── Cross-namespace retrieval (comparison/contradiction only) ──
        if query_type in ("comparison", "contradiction"):
            queried_namespaces = {_build_namespace(d) for d in meeting_dates}
            cross_top_k = max(top_k // 3, 5)

            # Only sweep namespaces NOT already queried, with original embedding only
            other_namespaces = [ns for ns in _ALL_NAMESPACES if ns not in queried_namespaces]
            logger.info(
                "Retriever — cross-namespace sweep across %d additional namespaces (top_k=%d)",
                len(other_namespaces), cross_top_k,
            )
            for ns in other_namespaces:
                cross_chunks = _single_query(index, all_embeddings[0], cross_top_k, {}, ns)
                all_chunks.extend(cross_chunks)

    elif is_multi and len(meeting_dates) < 2:
        # Comparison/contradiction but router didn't extract enough dates
        # → sweep ALL namespaces with original + 1 rewrite
        logger.info("Retriever — multi-meeting with <2 dates, sweeping all namespaces")
        for ns in _ALL_NAMESPACES:
            for emb in all_embeddings[:2]:
                chunks = _single_query(index, emb, top_k, {}, ns)
                all_chunks.extend(chunks)

    else:
        # ── Single-meeting path (factual / numerical with 0-1 dates) ──
        if len(meeting_dates) == 1:
            namespace = _build_namespace(meeting_dates[0])
            # Build filter without meeting_date (namespace handles it)
            single_filter = {k: v for k, v in metadata_filters.items() if k != "meeting_date"}
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

        logger.info("Retriever — querying namespace=%s filter=%s", namespace or "(default)", single_filter)

        for emb in all_embeddings:
            chunks = _single_query(index, emb, top_k, single_filter, namespace)
            all_chunks.extend(chunks)

        # ── Fallback broadening: progressively relax filters ───────────
        deduped_so_far = _deduplicate_chunks(all_chunks)
        if len(deduped_so_far) == 0 and single_filter:
            # Try 1: drop section filter
            broader = _strip_section_from_filter(single_filter)
            if broader != single_filter:
                logger.info("Retriever — fallback 1: dropping section filter → %s", broader)
                for emb in all_embeddings[:2]:
                    chunks = _single_query(index, emb, top_k, broader, namespace)
                    all_chunks.extend(chunks)

            deduped_so_far = _deduplicate_chunks(all_chunks)
            if len(deduped_so_far) == 0:
                # Try 2: drop doc_type filter too (just namespace)
                logger.info("Retriever — fallback 2: dropping all filters, namespace only")
                for emb in all_embeddings[:2]:
                    chunks = _single_query(index, emb, top_k, {}, namespace)
                    all_chunks.extend(chunks)

            deduped_so_far = _deduplicate_chunks(all_chunks)
            if len(deduped_so_far) == 0 and namespace:
                # Try 3: sweep all namespaces
                logger.info("Retriever — fallback 3: sweeping all namespaces")
                for ns in _ALL_NAMESPACES:
                    chunks = _single_query(index, all_embeddings[0], top_k, {}, ns)
                    all_chunks.extend(chunks)

    # ── Step 4: Deduplicate and sort ───────────────────────────────────
    all_chunks = _deduplicate_chunks(all_chunks)
    all_chunks.sort(key=lambda c: c["score"], reverse=True)

    # Cap total retrieved_chunks to avoid RAGAS processing huge contexts
    _MAX_RETRIEVED = 50
    _MAX_TEXT_CHUNKS = 25
    _MAX_TABLE_CHUNKS = 10

    capped_chunks = all_chunks[:_MAX_RETRIEVED]
    table_chunks = [c for c in capped_chunks if c.get("is_table")][:_MAX_TABLE_CHUNKS]
    text_chunks = [c for c in capped_chunks if not c.get("is_table")][:_MAX_TEXT_CHUNKS]

    logger.info(
        "Retriever — total=%d (deduped), returned=%d, tables=%d, text=%d",
        len(all_chunks), len(capped_chunks), len(table_chunks), len(text_chunks),
    )

    return {
        "retrieved_chunks": capped_chunks,
        "table_chunks": table_chunks,
        "text_chunks": text_chunks,
    }

