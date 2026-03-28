"""LangGraph state machine — full agentic RAG graph assembly."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from rag_agent.nodes.retriever import retriever_node
from rag_agent.nodes.router import router_node
from rag_agent.nodes.synthesizer import synthesizer_node
from rag_agent.nodes.table_reasoner import table_reasoner_node
from rag_agent.nodes.verifier import verifier_node
from rag_agent.state import RAGState

logger = logging.getLogger(__name__)


# ── Conditional edge functions ─────────────────────────────────────────────

def route_after_retrieval(state: RAGState) -> str:
    """Route to table_reasoner if table chunks exist, else synthesizer."""
    if len(state.get("table_chunks") or []) > 0:
        return "table_reasoner"
    return "synthesizer"


def route_after_verification(state: RAGState) -> str:
    """Route based on grounding verdict.

    - Grounded + high confidence → END
    - Not grounded + retries remaining → retry via retriever
    - Retries exhausted → END (with disclaimer)
    """
    if state.get("is_grounded") and (state.get("confidence_score") or 0) >= 0.7:
        return END
    if (state.get("retry_count") or 0) < 2:
        return "retriever"
    return END


# ── Graph assembly ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and return the (uncompiled) RAG StateGraph."""

    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("table_reasoner", table_reasoner_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("verifier", verifier_node)

    # Entry point
    graph.set_entry_point("router")

    # Edges
    graph.add_edge("router", "retriever")
    graph.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {"table_reasoner": "table_reasoner", "synthesizer": "synthesizer"},
    )
    graph.add_edge("table_reasoner", "synthesizer")
    graph.add_edge("synthesizer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verification,
        {END: END, "retriever": "retriever"},
    )

    return graph


def compile_graph():
    """Build and compile the RAG graph, ready for invocation."""
    graph = build_graph()
    return graph.compile()


# Pre-compiled app instance for direct import
app = compile_graph()


# ── Convenience runner ─────────────────────────────────────────────────────

def run_query(query: str) -> dict:
    """Run a query through the full RAG pipeline.

    Returns the final state dict.
    """
    initial_state: RAGState = {
        "query": query,
        "query_type": None,
        "metadata_filters": None,
        "retrieved_chunks": [],
        "table_chunks": [],
        "text_chunks": [],
        "draft_answer": None,
        "cited_sources": [],
        "contradiction_detected": None,
        "is_grounded": None,
        "confidence_score": None,
        "retry_count": 0,
        "final_answer": None,
        "error": None,
    }

    logger.info("Running query: %s", query[:120])
    result = app.invoke(initial_state)
    logger.info("Query complete — grounded=%s, confidence=%s",
                result.get("is_grounded"), result.get("confidence_score"))
    return result
