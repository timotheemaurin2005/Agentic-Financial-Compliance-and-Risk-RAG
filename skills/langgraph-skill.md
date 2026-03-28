# LangGraph Skill — Agentic RAG State Machine

## Purpose

This skill defines the LangGraph state machine that powers the agentic RAG system for FOMC document analysis. Every agent working on `rag_agent/` must follow this spec exactly.

## State Schema

The shared state passed between all nodes:

```python
from typing import TypedDict, Literal, Optional

class RAGState(TypedDict):
    # Input
    query: str
    query_type: Optional[Literal[
        "factual",          # "What was the rate decision in January 2025?"
        "numerical",        # "By how many basis points did they cut in September 2024?"
        "comparison",       # "Compare the risk assessment in the Sep and Dec minutes"
        "contradiction",    # "Did forward guidance language change between meetings?"
    ]]

    # Retrieval
    metadata_filters: Optional[dict]
    retrieved_chunks: list[dict]
    table_chunks: list[dict]
    text_chunks: list[dict]

    # Generation
    draft_answer: Optional[str]
    cited_sources: list[dict]
    contradiction_detected: Optional[bool]

    # Verification
    is_grounded: Optional[bool]
    confidence_score: Optional[float]
    retry_count: int

    # Output
    final_answer: Optional[str]
    error: Optional[str]
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
    "query_type": "contradiction",
    "meeting_dates": ["2024-09-18", "2025-01-29"],
    "doc_types": ["fomc_statement"],
    "sections": ["forward_guidance"]
  }
  ```
- Build `metadata_filters` from the extracted entities
- For `contradiction` and `comparison` queries: MUST extract at least two meeting dates or both doc_types

**Router Prompt (in `prompts/router_prompt.py`):**
```
You are a monetary policy query classifier. Given a user question about FOMC statements and meeting minutes, classify it and extract structured entities.

Query types:
- "factual": Simple lookup of a specific fact, rate decision, or piece of language
- "numerical": Requires calculation, basis point comparison, or vote counting
- "comparison": Asks to compare information across different meetings or document types
- "contradiction": Asks whether policy language changed, whether statement and minutes conflict, or what shifted between meetings

Available meeting dates: 2024-09-18, 2024-11-07, 2024-12-18, 2025-01-29, 2025-03-19
Available doc_types: fomc_statement, fomc_minutes
Available sections:
  Statements: rate_decision, economic_assessment, forward_guidance, vote_tally
  Minutes: staff_outlook, participants_views_economy, participants_views_policy, risk_assessment, committee_action, dissenting_views

Extract:
- meeting_dates: List of meeting dates mentioned or implied
- doc_types: Document types relevant to the query
- sections: Relevant sections to search

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
- For comparison/contradiction: run TWO separate queries (one per meeting date) and merge results
- Split retrieved chunks into `table_chunks` and `text_chunks` based on `is_table` metadata
- Sort by relevance score descending

**Pinecone Query Pattern:**
```python
results = index.query(
    vector=query_embedding,
    top_k=top_k,
    filter=metadata_filters,
    include_metadata=True,
    namespace=namespace,         # e.g., "fomc_2025-01-29"
)
```

### 3. Table Reasoner Node (`nodes/table_reasoner.py`)

**Input:** `table_chunks`, `text_chunks`, `query`, `query_type`
**Output:** Updated `text_chunks` (table analysis appended as text)

**Conditional:** Only runs if `len(table_chunks) > 0`. If no tables retrieved, skip to Synthesizer.

**Logic:**
- Format all table chunks as clean markdown tables
- For vote tallies: compare across meetings to spot dissent changes
- Send to LLM with a structured table analysis prompt
- Append analysis to `text_chunks`

**Table Prompt Pattern:**
```
You are a monetary policy analyst. Analyse the following tables extracted from FOMC documents.

{formatted_tables}

Question: {query}

Provide:
1. Key data points and their values
2. Changes between meetings (if multiple periods present)
3. Any notable shifts (e.g., new dissenting votes, changed rate targets)
4. Whether the tables support or contradict each other

Be precise. Cite which meeting date each figure comes from.
```

### 4. Synthesizer Node (`nodes/synthesizer.py`)

**Input:** `text_chunks`, `query`, `query_type`
**Output:** `draft_answer`, `cited_sources`, `contradiction_detected`

**Logic:**
- Combine all text chunks as numbered context passages
- Use query-type-specific prompts
- Set `contradiction_detected` if the LLM identifies conflicting information

**Synthesis Prompt (contradiction mode):**
```
You are a monetary policy analyst specialising in Fed communications. Answer the question using ONLY the provided context passages. Your task is to identify whether these sources agree or disagree.

CRITICAL: Pay close attention to Fed language signals:
- Qualifier shifts: "some participants" vs "most participants" vs "all participants"
- Certainty shifts: "noted" vs "judged" vs "agreed"
- Direction shifts: "further tightening" vs "maintaining" vs "prepared to adjust"
- Risk balance: "upside risks" vs "roughly in balance" vs "downside risks"

These are NOT casual word choices — they are deliberate policy signals.

Context passages:
{numbered_passages}

Question: {query}

Instructions:
- If sources AGREE: State the consistent finding and cite all supporting passages
- If sources DISAGREE: Quote the exact language from each source, state what changed, and explain the policy implications
- Always cite passage numbers in square brackets, e.g., [1], [3]
- If the evidence is insufficient to answer, say so explicitly
- Never invent information not present in the passages
```

### 5. Verifier Node (`nodes/verifier.py`)

**Input:** `draft_answer`, `text_chunks`, `query`, `retry_count`
**Output:** `is_grounded`, `confidence_score`, `final_answer` OR incremented `retry_count`

**Logic:**
- Send the draft answer + original context to the LLM as a grounding check
- Returns a JSON verdict:
  ```json
  {
    "is_grounded": true,
    "confidence": 0.87,
    "unsupported_claims": []
  }
  ```
- If `is_grounded == true` and `confidence >= 0.7`: set `final_answer = draft_answer`, proceed to END
- If not grounded or low confidence:
  - If `retry_count < 2`: increment, loop back to RETRIEVER with refined query
  - If `retry_count >= 2`: set `final_answer` with a low-confidence disclaimer

**Verification Prompt:**
```
You are a fact-checking assistant for monetary policy analysis. Evaluate whether the following answer is fully supported by the provided context.

Context passages:
{numbered_passages}

Answer to verify:
{draft_answer}

Check each claim. Pay special attention to:
- Are meeting dates correctly attributed?
- Are quoted Fed phrases actually present in the context?
- Are comparison claims supported by passages from BOTH meetings?

Respond ONLY with valid JSON:
{
  "is_grounded": bool,
  "confidence": float (0.0 to 1.0),
  "unsupported_claims": ["list of claims not supported by context"]
}
```

## Conditional Edges

```python
def route_after_retrieval(state: RAGState) -> str:
    if len(state["table_chunks"]) > 0:
        return "table_reasoner"
    return "synthesizer"

def route_after_verification(state: RAGState) -> str:
    if state["is_grounded"] and state["confidence_score"] >= 0.7:
        return "end"
    if state["retry_count"] < 2:
        return "retriever"
    return "end"
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

- [ ] Router correctly classifies: "What rate was set in Jan 2025?" → factual
- [ ] Router correctly classifies: "How did inflation language change Sep to Jan?" → contradiction
- [ ] Router extracts correct meeting dates and doc types from query
- [ ] Retriever returns filtered results (verify metadata matches filters)
- [ ] Table reasoner only fires when vote tally or projection tables are present
- [ ] Synthesizer quotes exact Fed language when detecting contradictions
- [ ] Verifier catches a deliberately wrong meeting date attribution
- [ ] Retry loop works: verifier failure → retriever → synthesizer → verifier (max 2 loops)
- [ ] Full graph executes end-to-end on a sample query without errors