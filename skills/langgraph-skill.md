# LangGraph Skill — Agentic RAG State Machine

## Purpose

This skill defines the LangGraph state machine that powers the agentic RAG system. Every agent working on `rag_agent/` must follow this spec exactly.

## State Schema

The shared state passed between all nodes:

```python
from typing import TypedDict, Literal, Optional

class RAGState(TypedDict):
    # Input
    query: str                                          # User's original question
    query_type: Optional[Literal[
        "factual",          # Simple lookup: "What was Apple's revenue in 2024?"
        "numerical",        # Calculation/trend: "How much did revenue change YoY?"
        "comparison",       # Cross-doc: "Compare risk factors 2023 vs 2024"
        "contradiction",    # Explicit conflict detection across sources
    ]]

    # Retrieval
    metadata_filters: Optional[dict]                    # Pinecone metadata filters
    retrieved_chunks: list[dict]                        # Raw retrieved chunks with metadata
    table_chunks: list[dict]                            # Subset: chunks where is_table=True
    text_chunks: list[dict]                             # Subset: chunks where is_table=False

    # Generation
    draft_answer: Optional[str]                         # Synthesiser output before verification
    cited_sources: list[dict]                           # Sources used in the answer
    contradiction_detected: Optional[bool]              # True if sources disagree

    # Verification
    is_grounded: Optional[bool]                         # Verifier verdict
    confidence_score: Optional[float]                   # 0.0–1.0
    retry_count: int                                    # Number of retrieval retries (max 2)

    # Output
    final_answer: Optional[str]                         # Verified answer returned to user
    error: Optional[str]                                # Error message if pipeline fails
```

## Graph Topology

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  ROUTER  │
                    └────┬─────┘
                         │
                    ┌────▼──────┐
                    │ RETRIEVER │◀──────────────┐
                    └────┬──────┘               │
                         │                      │
                    ┌────▼──────────┐           │
                    │ TABLE_REASONER│           │
                    │ (conditional) │           │
                    └────┬──────────┘           │
                         │                      │
                    ┌────▼────────┐             │
                    │ SYNTHESIZER │             │
                    └────┬────────┘             │
                         │                      │
                    ┌────▼───────┐    NO + retries < 2
                    │  VERIFIER  │──────────────┘
                    └────┬───────┘
                         │ YES (grounded)
                    ┌────▼─────┐
                    │   END    │
                    └──────────┘
```

## Node Specifications

### 1. Router Node (`nodes/router.py`)

**Input:** `query`
**Output:** `query_type`, `metadata_filters`

**Logic:**
- Send the query to the LLM with a classification prompt
- The LLM must return a JSON object with `query_type` and extracted entities:
  ```json
  {
    "query_type": "comparison",
    "companies": ["AAPL"],
    "fiscal_years": [2023, 2024],
    "sections": ["risk_factors"],
    "doc_types": ["10-K"]
  }
  ```
- Build `metadata_filters` from the extracted entities
- For `contradiction` and `comparison` queries: MUST extract at least two fiscal years or two doc types

**Router Prompt (in `prompts/router_prompt.py`):**
```
You are a financial query classifier. Given a user question about SEC filings and market reports, classify it and extract structured entities.

Query types:
- "factual": Simple lookup of a specific fact or figure
- "numerical": Requires calculation, trend analysis, or numerical comparison
- "comparison": Asks to compare information across different documents or time periods
- "contradiction": Asks whether sources agree or disagree, or what changed

Extract:
- companies: List of ticker symbols mentioned or implied
- fiscal_years: List of years mentioned or implied
- sections: Relevant filing sections (risk_factors, mda, financial_statements, business_overview)
- doc_types: Document types (10-K, 10-Q, earnings_transcript, market_report)

Respond ONLY with valid JSON matching the schema above.

User query: {query}
```

### 2. Retriever Node (`nodes/retriever.py`)

**Input:** `query`, `query_type`, `metadata_filters`
**Output:** `retrieved_chunks`, `table_chunks`, `text_chunks`

**Logic:**
- Embed the query using the same embedding model as ingestion
- Query Pinecone with:
  - `top_k=10` for factual/numerical queries
  - `top_k=15` for comparison/contradiction queries (need more coverage)
  - ALWAYS apply `metadata_filters` — never do unfiltered search
- For comparison/contradiction: run TWO separate queries (one per fiscal year) and merge results
- Split retrieved chunks into `table_chunks` and `text_chunks` based on `is_table` metadata
- Sort by relevance score descending

**Pinecone Query Pattern:**
```python
results = index.query(
    vector=query_embedding,
    top_k=top_k,
    filter=metadata_filters,    # ALWAYS filtered
    include_metadata=True,
    namespace=namespace,         # e.g., "AAPL_2024"
)
```

### 3. Table Reasoner Node (`nodes/table_reasoner.py`)

**Input:** `table_chunks`, `text_chunks`, `query`, `query_type`
**Output:** Updated `text_chunks` (table analysis appended as text)

**Conditional:** Only runs if `len(table_chunks) > 0`. If no tables retrieved, skip to Synthesizer.

**Logic:**
- Format all table chunks as clean markdown tables
- For comparison/contradiction queries: present tables side-by-side with clear year labels
- Send to LLM with a structured table analysis prompt
- The LLM outputs a textual analysis of the tables (trends, changes, anomalies)
- Append this analysis to `text_chunks` so the Synthesizer has it

**Table Prompt Pattern:**
```
You are a financial analyst. Analyse the following tables extracted from SEC filings.

{formatted_tables}

Question: {query}

Provide:
1. Key figures and their values
2. Year-over-year changes (if multiple periods)
3. Any notable anomalies or significant shifts
4. Whether the tables support or contradict each other (if from different periods)

Be precise with numbers. Cite which table/year each figure comes from.
```

### 4. Synthesizer Node (`nodes/synthesizer.py`)

**Input:** `text_chunks`, `query`, `query_type`
**Output:** `draft_answer`, `cited_sources`, `contradiction_detected`

**Logic:**
- Combine all text chunks (including table analysis) as numbered context passages
- Use a query-type-specific prompt:
  - **Factual/numerical:** Direct answer with source citations
  - **Comparison:** Structured comparison with explicit "Source A says X, Source B says Y" format
  - **Contradiction:** Must explicitly state whether sources agree or disagree, what changed, and cite both sides
- Extract source citations as a list of `{company, fiscal_year, doc_type, section, chunk_id}`
- Set `contradiction_detected` if the LLM identifies conflicting information

**Synthesis Prompt (contradiction mode):**
```
You are a financial compliance analyst. Answer the question using ONLY the provided context passages. Your task is to identify whether these sources agree or disagree.

Context passages:
{numbered_passages}

Question: {query}

Instructions:
- If sources AGREE: State the consistent finding and cite all supporting passages
- If sources DISAGREE: Clearly state what each source says, what specifically changed, and cite each passage
- Always cite passage numbers in square brackets, e.g., [1], [3]
- If the evidence is insufficient to answer, say so explicitly
- Never invent information not present in the passages
```

### 5. Verifier Node (`nodes/verifier.py`)

**Input:** `draft_answer`, `text_chunks`, `query`, `retry_count`
**Output:** `is_grounded`, `confidence_score`, `final_answer` OR incremented `retry_count`

**Logic:**
- Send the draft answer + original context to the LLM as a grounding check
- The LLM evaluates: "Is every claim in the answer supported by the context passages?"
- Returns a JSON verdict:
  ```json
  {
    "is_grounded": true,
    "confidence": 0.87,
    "unsupported_claims": []
  }
  ```
- If `is_grounded == true` and `confidence >= 0.7`: set `final_answer = draft_answer`, proceed to END
- If `is_grounded == false` or `confidence < 0.7`:
  - If `retry_count < 2`: increment `retry_count`, loop back to RETRIEVER with a refined query
  - If `retry_count >= 2`: set `final_answer` with a low-confidence disclaimer

**Verification Prompt:**
```
You are a fact-checking assistant. Evaluate whether the following answer is fully supported by the provided context.

Context passages:
{numbered_passages}

Answer to verify:
{draft_answer}

For each claim in the answer, check if it is directly supported by a context passage. Respond ONLY with valid JSON:
{
  "is_grounded": bool,
  "confidence": float (0.0 to 1.0),
  "unsupported_claims": ["list of claims not supported by context"]
}
```

## Conditional Edges

```python
def route_after_retrieval(state: RAGState) -> str:
    """Route to table_reasoner if tables exist, otherwise skip to synthesizer."""
    if len(state["table_chunks"]) > 0:
        return "table_reasoner"
    return "synthesizer"

def route_after_verification(state: RAGState) -> str:
    """Loop back to retriever if not grounded, otherwise end."""
    if state["is_grounded"] and state["confidence_score"] >= 0.7:
        return "end"
    if state["retry_count"] < 2:
        return "retriever"
    return "end"  # Give up after 2 retries, return with disclaimer
```

## Graph Assembly (`graph.py`)

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(RAGState)

graph.add_node("router", router_node)
graph.add_node("retriever", retriever_node)
graph.add_node("table_reasoner", table_reasoner_node)
graph.add_node("synthesizer", synthesizer_node)
graph.add_node("verifier", verifier_node)

graph.set_entry_point("router")
graph.add_edge("router", "retriever")
graph.add_conditional_edges("retriever", route_after_retrieval)
graph.add_edge("table_reasoner", "synthesizer")
graph.add_edge("synthesizer", "verifier")
graph.add_conditional_edges("verifier", route_after_verification)

app = graph.compile()
```

## Testing Checklist

- [ ] Router correctly classifies all 4 query types with sample questions
- [ ] Retriever returns filtered results (verify metadata matches filters)
- [ ] Table reasoner only fires when table chunks are present
- [ ] Synthesizer produces cited answers with passage references
- [ ] Verifier catches a deliberately wrong answer (inject a hallucinated claim)
- [ ] Retry loop works: verifier failure → retriever → synthesizer → verifier (max 2 loops)
- [ ] Full graph executes end-to-end on a sample query without errors