# Agentic Financial Compliance & Risk RAG

## Project Overview

An agentic Retrieval-Augmented Generation system for monetary policy analysis. The system ingests FOMC Meeting Minutes and FOMC Statements from the Federal Reserve, then uses a LangGraph state machine to retrieve, compare, and detect contradictions across documents — e.g., comparing how rate guidance language shifted between the January and March 2025 FOMC statements, or identifying where the Minutes reveal internal disagreement that the Statement's consensus language obscures.

This is NOT a simple retrieve-then-generate loop. The agent classifies queries, applies metadata-filtered retrieval, handles tabular data as structured objects, detects contradictions across meetings, and self-verifies answers for grounding.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  ingestion/  │────▶│   Pinecone   │◀────│  rag_agent/  │
│  parse/chunk │     │  Vector DB   │     │  LangGraph   │
│  embed/upsert│     └──────────────┘     │  state machine│
└─────────────┘                           └──────┬───────┘
                                                 │
                                          ┌──────▼───────┐
                                          │    api/       │
                                          │   FastAPI     │
                                          │   + frontend  │
                                          └──────────────┘
```

## Directory Structure

```
project-root/
├── CLAUDE.md                  # This file — persistent context for Claude Code
├── skills/
│   ├── ingestion-skill.md
│   ├── langgraph-skill.md
│   └── eval-skill.md
├── ingestion/
│   ├── __init__.py
│   ├── parser.py              # PDF/HTML parsing with table extraction
│   ├── chunker.py             # Section-aware chunking with metadata
│   ├── embedder.py            # Dual embedding (raw + summary)
│   ├── upserter.py            # Pinecone upsert with metadata filters
│   ├── pipeline.py            # End-to-end ingestion orchestrator
│   └── schemas.py             # Pydantic models for chunks and metadata
├── rag_agent/
│   ├── __init__.py
│   ├── graph.py               # LangGraph state machine definition
│   ├── state.py               # TypedDict state schema
│   ├── nodes/
│   │   ├── router.py          # Query classification node
│   │   ├── retriever.py       # Filtered vector search node
│   │   ├── table_reasoner.py  # Structured table comparison node
│   │   ├── synthesizer.py     # Answer generation with citations
│   │   └── verifier.py        # Grounding self-check loop
│   └── prompts/
│       ├── router_prompt.py
│       ├── synthesis_prompt.py
│       ├── table_prompt.py
│       └── verification_prompt.py
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI app
│   ├── routes.py              # POST /query, GET /documents
│   └── streaming.py           # SSE streaming support
├── eval/
│   ├── __init__.py
│   ├── dataset.py             # Eval dataset loader
│   ├── run_ragas.py           # RAGAS evaluation harness
│   ├── ablation.py            # Hyperparameter sweep (chunk size, top-k, etc.)
│   └── data/
│       └── eval_set.json      # 30 gold-standard Q&A pairs
├── data/
│   ├── raw/                   # Original FOMC PDFs/HTML (minutes + statements)
│   └── processed/             # Chunked JSON with metadata
├── .env                       # API keys (NEVER commit)
├── .gitignore
├── pyproject.toml
├── Makefile
└── README.md
```

## Document Sources

### FOMC Statements
- Short (1–2 pages), official post-meeting press releases
- Contain: rate decision, vote tally, forward guidance language, economic assessment
- Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- Format: HTML (scrape from Fed website) or PDF
- Published: Same day as meeting (8 meetings per year)

### FOMC Minutes
- Long (15–20 pages), detailed record of meeting discussion
- Contain: staff economic outlook, participants' views on inflation/employment/rates, risk assessment, policy debate, dissenting views
- Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- Format: HTML or PDF
- Published: 3 weeks after each meeting

### Key Contradiction Patterns to Detect
1. **Statement vs Statement:** Rate guidance language shifts between meetings (e.g., "at or near its peak" → "prepared to adjust")
2. **Statement vs Minutes:** Consensus language in statement vs internal disagreement revealed in minutes (e.g., statement says "unanimous" but minutes show "several participants" preferred a different path)
3. **Minutes vs Minutes:** Evolving risk assessments (e.g., "upside risks to inflation" in one meeting → "risks roughly in balance" in the next)
4. **Dot plot implications:** Forward guidance in statements vs the range of rate projections discussed in minutes

## Tech Stack

| Layer         | Technology                    | Notes                                          |
|---------------|-------------------------------|-------------------------------------------------|
| Ingestion     | Python (pymupdf, beautifulsoup4) | PDF/HTML parsing for Fed documents           |
| Chunking      | Custom (ingestion/chunker.py) | Section-aware, ~500 tokens text, full tables    |
| Embedding     | text-embedding-3-large        | Dual: raw chunk + LLM-generated summary         |
| Vector DB     | Pinecone (free tier)          | Metadata filtering on meeting_date/doc_type     |
| Orchestration | LangGraph                     | 5-node agentic state machine                    |
| LLM           | GPT-4o (synthesis/routing)    | Or Claude — switchable via env var              |
| Serving       | FastAPI + SSE                 | Streaming responses                             |
| Evaluation    | RAGAS                         | Faithfulness, relevancy, precision, recall       |
| Frontend      | React (optional)              | Chat UI with source citation panel              |

## Metadata Schema (MANDATORY for every chunk)

Every chunk upserted to Pinecone MUST carry this metadata:

```python
{
    "doc_type": str,         # One of: "fomc_statement", "fomc_minutes"
    "meeting_date": str,     # ISO format: "2025-01-29", "2025-03-19", etc.
    "year": int,             # e.g., 2025
    "section": str,          # For statements: "rate_decision", "economic_assessment",
                             #   "forward_guidance", "vote_tally"
                             # For minutes: "staff_outlook", "participants_views_economy",
                             #   "participants_views_policy", "risk_assessment",
                             #   "committee_action", "dissenting_views"
    "is_table": bool,        # True if chunk is a structured table (e.g., vote tally)
    "source_url": str,       # Federal Reserve URL
    "chunk_index": int,      # Position within the source document
}
```

## Environment Variables

```
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=fin-compliance-rag
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...         # Optional — if using Claude for synthesis
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4o             # Switchable: gpt-4o | claude-sonnet-4-6
```

## Key Design Decisions

1. **LangGraph over LangChain RetrievalQA** — We need conditional routing, self-verification loops, and multi-node orchestration. Do NOT use simple chain-based retrieval.
2. **Tables as structured objects** — Vote tallies and economic projection tables are chunked as markdown tables, NOT flattened into prose.
3. **Dual embedding** — Each chunk gets two vectors: one from the raw text, one from an LLM-generated summary. This improves retrieval for conceptual queries.
4. **Metadata-filtered retrieval** — The retriever node ALWAYS applies metadata filters (meeting_date + doc_type at minimum) before similarity search. Never do unfiltered search.
5. **Contradiction detection** — For comparison queries, the retriever explicitly pulls chunks from BOTH meetings. The synthesizer prompt explicitly asks for agreement/disagreement analysis with exact language comparison.
6. **Fed language sensitivity** — The system must detect subtle shifts in Fed language: "some participants" vs "most participants", "at or near" vs "somewhat above", "further tightening" vs "prepared to adjust". These are deliberate signals, not casual word choices.

## Target Meetings (Initial Dataset)

Download these meetings to start (gives 5 consecutive meetings for trend detection):

| Meeting Date | Statement | Minutes | Key Context                    |
|-------------|-----------|---------|--------------------------------|
| 2024-09-18  | Yes       | Yes     | First rate cut in cycle        |
| 2024-11-07  | Yes       | Yes     | Post-election meeting          |
| 2024-12-18  | Yes       | Yes     | December projections           |
| 2025-01-29  | Yes       | Yes     | First meeting of 2025          |
| 2025-03-19  | Yes       | Yes     | Latest available               |

This gives 10 documents (5 statements + 5 minutes) — enough for meaningful comparison without overwhelming the free Pinecone tier.

## Conventions

- Python 3.11+
- snake_case everywhere
- Type hints on all function signatures
- Pydantic models for all data structures (schemas.py)
- No hardcoded API keys — always env vars
- All prompts live in rag_agent/prompts/ as string templates
- Pinecone index name: `fin-compliance-rag`
- Pinecone namespace: `fomc_{meeting_date}` (e.g., `fomc_2025-01-29`)

## What NOT To Do

- Do NOT use LangChain's RetrievalQA or ConversationalRetrievalChain
- Do NOT flatten tables into paragraph text
- Do NOT hardcode API keys or model names
- Do NOT do unfiltered vector search (always apply metadata filters)
- Do NOT skip the verification node — every answer must be grounding-checked
- Do NOT embed tables and text with the same strategy — tables get summary-only embedding
- Do NOT treat Fed language as casual — subtle word changes between meetings are intentional policy signals