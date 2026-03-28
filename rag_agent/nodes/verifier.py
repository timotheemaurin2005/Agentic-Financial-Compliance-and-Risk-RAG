"""Verifier node — grounding self-check with retry loop."""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from rag_agent.prompts.verification_prompt import VERIFICATION_PROMPT
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


def verifier_node(state: RAGState) -> dict:
    """Self-check whether the draft answer is grounded in the context.

    Returns:
    - If grounded (``is_grounded=True`` and ``confidence >= 0.7``): sets
      ``final_answer`` to the draft.
    - If not grounded and ``retry_count < 2``: increments ``retry_count``
      (the conditional edge will route back to the retriever).
    - If retries exhausted: sets ``final_answer`` with a low-confidence
      disclaimer.
    """
    draft_answer: str = state.get("draft_answer") or ""
    text_chunks: list[dict] = state.get("text_chunks") or []
    retry_count: int = state.get("retry_count") or 0

    logger.info("Verifier — retry_count=%d", retry_count)

    if not draft_answer:
        return {
            "is_grounded": False,
            "confidence_score": 0.0,
            "final_answer": "Unable to generate an answer.",
            "retry_count": retry_count,
        }

    # Build the verification prompt
    numbered_passages = _format_numbered_passages(text_chunks)
    prompt = VERIFICATION_PROMPT.format(
        numbered_passages=numbered_passages,
        draft_answer=draft_answer,
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a rigorous fact-checker. Respond ONLY with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content or "{}"
    try:
        verdict = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("Verifier — failed to parse verdict: %s", raw_text)
        verdict = {"is_grounded": False, "confidence": 0.0, "unsupported_claims": []}

    is_grounded: bool = verdict.get("is_grounded", False)
    confidence: float = float(verdict.get("confidence", 0.0))
    unsupported: list[str] = verdict.get("unsupported_claims", [])

    logger.info(
        "Verifier — grounded=%s, confidence=%.2f, unsupported=%s",
        is_grounded, confidence, unsupported,
    )

    # ── Decision logic ─────────────────────────────────────────────────
    if is_grounded and confidence >= 0.7:
        return {
            "is_grounded": True,
            "confidence_score": confidence,
            "final_answer": draft_answer,
            "retry_count": retry_count,
        }

    # Not grounded — try again or give up
    if retry_count < 2:
        logger.info("Verifier — not grounded, retrying (retry_count will be %d)", retry_count + 1)
        return {
            "is_grounded": False,
            "confidence_score": confidence,
            "retry_count": retry_count + 1,
        }

    # Retries exhausted — return with disclaimer
    disclaimer = (
        f"\n\n⚠️ **Low-confidence answer** (confidence: {confidence:.0%}). "
        "Some claims may not be fully supported by the source documents. "
        "Please verify against the original FOMC publications."
    )
    if unsupported:
        disclaimer += "\n\nPotentially unsupported claims:\n"
        for claim in unsupported:
            disclaimer += f"- {claim}\n"

    return {
        "is_grounded": False,
        "confidence_score": confidence,
        "final_answer": draft_answer + disclaimer,
        "retry_count": retry_count,
    }
