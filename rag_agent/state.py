"""RAGState — shared state schema for the LangGraph agentic RAG pipeline."""

from typing import Any, Literal, Optional, TypedDict


class RAGState(TypedDict):
    """Shared state passed between all nodes in the LangGraph state machine."""

    # ── Input ──────────────────────────────────────────────────────────
    query: str
    query_type: Optional[Literal[
        "factual",          # "What was the rate decision in January 2025?"
        "numerical",        # "By how many basis points did they cut in September 2024?"
        "comparison",       # "Compare the risk assessment in the Sep and Dec minutes"
        "contradiction",    # "Did forward guidance language change between meetings?"
    ]]

    # ── Retrieval ──────────────────────────────────────────────────────
    metadata_filters: Optional[dict]
    _router_classification: Optional[dict[str, Any]]  # full LLM classification for retriever
    retrieved_chunks: list[dict]
    table_chunks: list[dict]
    text_chunks: list[dict]

    # ── Generation ─────────────────────────────────────────────────────
    draft_answer: Optional[str]
    cited_sources: list[dict]
    contradiction_detected: Optional[bool]

    # ── Verification ───────────────────────────────────────────────────
    is_grounded: Optional[bool]
    confidence_score: Optional[float]
    retry_count: int

    # ── Output ─────────────────────────────────────────────────────────
    final_answer: Optional[str]
    error: Optional[str]
