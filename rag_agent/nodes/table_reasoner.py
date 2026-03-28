"""Table Reasoner node — structured table analysis (conditional)."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from rag_agent.prompts.table_prompt import TABLE_REASONING_PROMPT
from rag_agent.state import RAGState

load_dotenv()
logger = logging.getLogger(__name__)

_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")


def _format_tables(table_chunks: list[dict]) -> str:
    """Format table chunks into labelled markdown blocks for the prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(table_chunks, 1):
        date = chunk.get("meeting_date", "unknown")
        doc = chunk.get("doc_type", "unknown")
        section = chunk.get("section", "unknown")
        text = chunk.get("text", "")
        parts.append(
            f"--- Table {i} (Meeting: {date}, Document: {doc}, Section: {section}) ---\n{text}"
        )
    return "\n\n".join(parts)


def table_reasoner_node(state: RAGState) -> dict:
    """Analyse table chunks and append structured analysis to text_chunks.

    This node ONLY fires when ``table_chunks`` is non-empty.  The conditional
    edge in ``graph.py`` enforces this.
    """
    table_chunks: list[dict] = state.get("table_chunks") or []
    text_chunks: list[dict] = list(state.get("text_chunks") or [])
    query: str = state["query"]

    if not table_chunks:
        logger.info("TableReasoner — no table chunks, skipping")
        return {}

    logger.info("TableReasoner — analysing %d table(s)", len(table_chunks))

    formatted_tables = _format_tables(table_chunks)
    prompt = TABLE_REASONING_PROMPT.format(
        formatted_tables=formatted_tables,
        query=query,
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a monetary policy data analyst. Provide precise, structured analysis.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    analysis = response.choices[0].message.content or ""
    logger.info("TableReasoner — analysis length: %d chars", len(analysis))

    # Append the table analysis as a synthetic text chunk
    text_chunks.append(
        {
            "id": "table_analysis",
            "score": 1.0,
            "text": f"[TABLE ANALYSIS]\n{analysis}",
            "is_table": False,
            "meeting_date": "multiple",
            "doc_type": "analysis",
            "section": "table_analysis",
            "chunk_index": -1,
            "source_url": "",
            "metadata": {},
        }
    )

    return {"text_chunks": text_chunks}
