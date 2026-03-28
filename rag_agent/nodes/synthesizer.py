"""Synthesizer node — answer generation with passage citations."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from rag_agent.prompts.synthesis_prompt import FED_LANGUAGE_BLOCK, get_synthesis_prompt
from rag_agent.state import RAGState

load_dotenv()
logger = logging.getLogger(__name__)

_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")


def _format_numbered_passages(text_chunks: list[dict]) -> str:
    """Build a numbered passage block from text chunks."""
    parts: list[str] = []
    for i, chunk in enumerate(text_chunks, 1):
        date = chunk.get("meeting_date", "")
        doc = chunk.get("doc_type", "")
        section = chunk.get("section", "")
        header = f"[{i}] (Meeting: {date}, Document: {doc}, Section: {section})"
        parts.append(f"{header}\n{chunk.get('text', '')}")
    return "\n\n".join(parts)


def _extract_cited_sources(text_chunks: list[dict], answer: str) -> list[dict]:
    """Parse bracket citations like [1], [3] from the answer and collect sources."""
    import re

    cited_indices: set[int] = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        cited_indices.add(int(m.group(1)))

    sources: list[dict] = []
    for idx in sorted(cited_indices):
        if 1 <= idx <= len(text_chunks):
            chunk = text_chunks[idx - 1]
            sources.append(
                {
                    "passage_index": idx,
                    "meeting_date": chunk.get("meeting_date", ""),
                    "doc_type": chunk.get("doc_type", ""),
                    "section": chunk.get("section", ""),
                    "source_url": chunk.get("source_url", ""),
                    "score": chunk.get("score", 0.0),
                }
            )
    return sources


def _detect_contradiction(answer: str) -> bool:
    """Heuristic: detect contradiction language in the answer."""
    contradiction_signals = [
        "disagree",
        "contradict",
        "conflict",
        "inconsistent",
        "changed",
        "shifted",
        "differs",
        "no longer",
        "in contrast",
        "unlike",
        "reversal",
    ]
    answer_lower = answer.lower()
    return any(signal in answer_lower for signal in contradiction_signals)


def synthesizer_node(state: RAGState) -> dict:
    """Generate a draft answer with passage citations.

    Uses query-type-specific prompts and the Fed language sensitivity block.
    Sets ``contradiction_detected`` when change-language is detected.
    """
    text_chunks: list[dict] = state.get("text_chunks") or []
    query: str = state["query"]
    query_type: str = state.get("query_type") or "factual"

    logger.info(
        "Synthesizer — query_type=%s, passages=%d", query_type, len(text_chunks),
    )

    if not text_chunks:
        return {
            "draft_answer": "I could not find relevant information to answer this question.",
            "cited_sources": [],
            "contradiction_detected": False,
        }

    # Build the prompt
    numbered_passages = _format_numbered_passages(text_chunks)
    prompt_template = get_synthesis_prompt(query_type)
    prompt = prompt_template.format(
        fed_language_block=FED_LANGUAGE_BLOCK,
        numbered_passages=numbered_passages,
        query=query,
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior monetary policy analyst. "
                    "Provide precise, well-cited answers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    answer = response.choices[0].message.content or ""
    cited_sources = _extract_cited_sources(text_chunks, answer)
    contradiction_detected = _detect_contradiction(answer)

    logger.info(
        "Synthesizer — answer_len=%d, cited=%d, contradiction=%s",
        len(answer), len(cited_sources), contradiction_detected,
    )

    return {
        "draft_answer": answer,
        "cited_sources": cited_sources,
        "contradiction_detected": contradiction_detected,
    }
