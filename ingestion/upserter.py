"""Pinecone upsert with auto-index creation and metadata filters."""

from __future__ import annotations

import logging
import os
import time

from pinecone import Pinecone, ServerlessSpec

from ingestion.schemas import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPSERT_BATCH_SIZE = 100
DIMENSION = 3072                  # text-embedding-3-large
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"              # Pinecone free-tier region


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _get_client() -> Pinecone:
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def ensure_index(
    *,
    pc: Pinecone | None = None,
    index_name: str | None = None,
) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    if pc is None:
        pc = _get_client()
    if index_name is None:
        index_name = os.environ["PINECONE_INDEX_NAME"]

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        logger.info("Pinecone index '%s' already exists.", index_name)
        return

    logger.info(
        "Creating Pinecone index '%s' (dim=%d, metric=%s) …",
        index_name,
        DIMENSION,
        METRIC,
    )
    pc.create_index(
        name=index_name,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )

    # Wait for the index to be ready
    while not pc.describe_index(index_name).status.get("ready", False):
        logger.info("Waiting for index to become ready …")
        time.sleep(2)

    logger.info("Index '%s' is ready.", index_name)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    chunks: list[Chunk],
    raw_embeddings: list[list[float]],
    summary_embeddings: list[list[float]],
    *,
    pc: Pinecone | None = None,
    index_name: str | None = None,
) -> int:
    """Upsert dual vectors (raw + summary) for each chunk into Pinecone.

    Returns the total number of vectors upserted.
    """
    if pc is None:
        pc = _get_client()
    if index_name is None:
        index_name = os.environ["PINECONE_INDEX_NAME"]

    index = pc.Index(index_name)
    total_upserted = 0

    # Group by namespace (one per meeting date)
    ns_map: dict[str, list[tuple[Chunk, list[float], list[float]]]] = {}
    for chunk, raw_emb, sum_emb in zip(chunks, raw_embeddings, summary_embeddings):
        ns = f"fomc_{chunk.metadata.meeting_date}"
        ns_map.setdefault(ns, []).append((chunk, raw_emb, sum_emb))

    for namespace, items in ns_map.items():
        vectors: list[dict] = []
        for chunk, raw_emb, sum_emb in items:
            meta = chunk.metadata.to_pinecone_dict()
            # Also store the chunk text in metadata for retrieval debugging
            meta["text"] = chunk.text[:1000]   # Pinecone metadata limit ≈40 KB
            meta["summary"] = chunk.summary[:500]
            meta["embedding_type"] = "raw"

            vectors.append({
                "id": f"{chunk.id}_raw",
                "values": raw_emb,
                "metadata": {**meta, "embedding_type": "raw"},
            })
            vectors.append({
                "id": f"{chunk.id}_summary",
                "values": sum_emb,
                "metadata": {**meta, "embedding_type": "summary"},
            })

        # Batch upsert
        for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
            batch = vectors[start : start + UPSERT_BATCH_SIZE]
            try:
                index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
                logger.info(
                    "Upserted %d vectors to namespace '%s' (batch %d–%d)",
                    len(batch),
                    namespace,
                    start,
                    start + len(batch),
                )
            except Exception as exc:
                logger.warning(
                    "Upsert failed for batch %d–%d in namespace '%s': %s — retrying once",
                    start,
                    start + len(batch),
                    namespace,
                    exc,
                )
                try:
                    time.sleep(2)
                    index.upsert(vectors=batch, namespace=namespace)
                    total_upserted += len(batch)
                except Exception as exc2:
                    logger.error(
                        "Upsert retry failed: %s — skipping batch", exc2
                    )

    logger.info("Total vectors upserted: %d", total_upserted)
    return total_upserted


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_upsert(
    *,
    pc: Pinecone | None = None,
    index_name: str | None = None,
    namespace: str = "fomc_2025-01-29",
    doc_type: str = "fomc_statement",
    meeting_date: str = "2025-01-29",
) -> bool:
    """Run a test query to verify vectors were upserted correctly."""
    if pc is None:
        pc = _get_client()
    if index_name is None:
        index_name = os.environ["PINECONE_INDEX_NAME"]

    index = pc.Index(index_name)

    # Describe namespace stats
    stats = index.describe_index_stats()
    ns_stats = stats.get("namespaces", {})
    logger.info("Index stats — namespaces: %s", ns_stats)

    if namespace not in ns_stats:
        logger.error("Namespace '%s' not found in index!", namespace)
        return False

    vec_count = ns_stats[namespace].get("vector_count", 0)
    logger.info("Namespace '%s' has %d vectors.", namespace, vec_count)

    if vec_count == 0:
        logger.error("No vectors in namespace '%s'!", namespace)
        return False

    # Query with a dummy vector (zeros) just to confirm metadata filter works
    dummy_vec = [0.0] * DIMENSION
    results = index.query(
        vector=dummy_vec,
        top_k=1,
        namespace=namespace,
        filter={"doc_type": doc_type, "meeting_date": meeting_date},
        include_metadata=True,
    )

    if results.matches:
        match = results.matches[0]
        logger.info(
            "✅ Verification passed — top-1 result: id=%s, metadata=%s",
            match.id,
            match.metadata,
        )
        return True
    else:
        logger.warning("⚠️  Verification query returned no matches.")
        return False
