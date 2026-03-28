"""Router node — classifies the incoming query and extracts metadata filters."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from rag_agent.prompts.router_prompt import ROUTER_PROMPT
from rag_agent.state import RAGState

load_dotenv()
logger = logging.getLogger(__name__)

# ── LLM client setup ──────────────────────────────────────────────────────

_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")


def _get_client() -> OpenAI:
    """Return an OpenAI-compatible client.

    If ``LLM_MODEL`` points to an Anthropic model the caller is expected to
    have an OpenAI-compatible proxy or to swap this helper.  For now we keep
    the dependency surface minimal and use the OpenAI SDK everywhere.
    """
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_metadata_filters(classification: dict[str, Any]) -> dict:
    """Convert the router's extracted entities into Pinecone metadata filters.

    Returns a Pinecone-compatible ``filter`` dict that can be passed straight
    to ``index.query(filter=...)``.
    """
    conditions: list[dict] = []

    # Meeting dates → filter on meeting_date field
    meeting_dates: list[str] = classification.get("meeting_dates", [])
    if meeting_dates:
        if len(meeting_dates) == 1:
            conditions.append({"meeting_date": {"$eq": meeting_dates[0]}})
        else:
            conditions.append({"meeting_date": {"$in": meeting_dates}})

    # Doc types
    doc_types: list[str] = classification.get("doc_types", [])
    if doc_types:
        if len(doc_types) == 1:
            conditions.append({"doc_type": {"$eq": doc_types[0]}})
        else:
            conditions.append({"doc_type": {"$in": doc_types}})

    # Sections
    sections: list[str] = classification.get("sections", [])
    if sections:
        if len(sections) == 1:
            conditions.append({"section": {"$eq": sections[0]}})
        else:
            conditions.append({"section": {"$in": sections}})

    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Node function ──────────────────────────────────────────────────────────

def router_node(state: RAGState) -> dict:
    """Classify the user query and generate metadata filters.

    Returns a partial state update with ``query_type``, ``metadata_filters``,
    and the raw classification dict stashed in ``_router_classification`` (for
    downstream use by the retriever).
    """
    query: str = state["query"]
    logger.info("Router — classifying query: %s", query[:120])

    client = _get_client()
    response = client.chat.completions.create(
        model=_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise JSON-only classifier. Respond ONLY with valid JSON.",
            },
            {"role": "user", "content": ROUTER_PROMPT.format(query=query)},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content or "{}"
    try:
        classification = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("Router — failed to parse LLM response: %s", raw_text)
        classification = {
            "query_type": "factual",
            "meeting_dates": [],
            "doc_types": [],
            "sections": [],
        }

    query_type = classification.get("query_type", "factual")
    metadata_filters = _build_metadata_filters(classification)

    logger.info("Router — query_type=%s, filters=%s", query_type, metadata_filters)

    return {
        "query_type": query_type,
        "metadata_filters": metadata_filters,
        # Stash full classification for the retriever to access meeting_dates
        "_router_classification": classification,
    }
