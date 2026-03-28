"""Dual embedding: raw text + LLM-generated summary via OpenAI API."""

from __future__ import annotations

import logging
import os
import time

from openai import OpenAI, RateLimitError

from ingestion.schemas import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_BATCH_SIZE = 100          # Max chunks per embedding API call
MAX_RETRIES = 5                     # Exponential backoff retries
INITIAL_BACKOFF_SECS = 2.0

SUMMARY_PROMPT_TEMPLATE = (
    "Summarise the following FOMC document excerpt in one sentence.\n"
    "Focus on: the policy topic discussed, the meeting date, and any specific "
    "language about rates, inflation, or employment.\n"
    "Pay special attention to qualifier words (some, most, several, a few "
    "participants) as these signal the degree of consensus.\n"
    "\n"
    "Excerpt:\n"
    "{chunk_text}\n"
    "\n"
    "One-sentence summary:"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _call_with_backoff(fn, *args, **kwargs):
    """Call *fn* with exponential backoff on rate-limit errors."""
    backoff = INITIAL_BACKOFF_SECS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as exc:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                "Rate-limited (attempt %d/%d) — retrying in %.1fs: %s",
                attempt,
                MAX_RETRIES,
                backoff,
                exc,
            )
            time.sleep(backoff)
            backoff *= 2
    return None  # unreachable, but keeps mypy happy


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summaries(
    chunks: list[Chunk],
    *,
    client: OpenAI | None = None,
    llm_model: str | None = None,
) -> list[Chunk]:
    """Fill ``chunk.summary`` using the LLM for every chunk that lacks one.

    Modifies chunks in-place and returns the same list.
    """
    if client is None:
        client = _get_client()
    if llm_model is None:
        llm_model = os.getenv("LLM_MODEL", "gpt-4o")

    logger.info("Generating summaries for %d chunks using %s …", len(chunks), llm_model)

    for i, chunk in enumerate(chunks):
        if chunk.summary:
            continue
        prompt = SUMMARY_PROMPT_TEMPLATE.format(chunk_text=chunk.text)

        resp = _call_with_backoff(
            client.chat.completions.create,
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.3,
        )
        chunk.summary = resp.choices[0].message.content.strip()

        if (i + 1) % 20 == 0:
            logger.info("  summaries: %d / %d", i + 1, len(chunks))

    logger.info("Summary generation complete.")
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    *,
    client: OpenAI | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Embed a list of texts, batching at EMBEDDING_BATCH_SIZE.

    Returns a flat list of embedding vectors (one per input text).
    """
    if client is None:
        client = _get_client()
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        logger.info(
            "Embedding batch %d–%d of %d",
            start,
            start + len(batch),
            len(texts),
        )
        resp = _call_with_backoff(
            client.embeddings.create,
            input=batch,
            model=model,
        )
        # Sort by index to guarantee order
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        all_embeddings.extend([d.embedding for d in sorted_data])

    return all_embeddings


def embed_chunks(
    chunks: list[Chunk],
    *,
    client: OpenAI | None = None,
    model: str | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Generate dual embeddings (raw + summary) for a list of chunks.

    Returns
    -------
    raw_embeddings : list[list[float]]
        One embedding per chunk, from the raw chunk text.
    summary_embeddings : list[list[float]]
        One embedding per chunk, from the chunk summary.
    """
    raw_texts = [c.text for c in chunks]
    summary_texts = [c.summary for c in chunks]

    logger.info("Embedding %d raw texts …", len(raw_texts))
    raw_embs = embed_texts(raw_texts, client=client, model=model)

    logger.info("Embedding %d summaries …", len(summary_texts))
    summary_embs = embed_texts(summary_texts, client=client, model=model)

    return raw_embs, summary_embs
