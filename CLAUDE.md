# Agentic Financial Compliance & Risk RAG

## Project Overview

An agentic Retrieval-Augmented Generation system for financial compliance and risk analysis. The system ingests SEC filings (10-K, 10-Q), earnings transcripts, and market reports, then uses a LangGraph state machine to retrieve, compare, and detect contradictions across documents — e.g., comparing a company's 2023 10-K risk disclosures against a 2024 market outlook report.

This is NOT a simple retrieve-then-generate loop. The agent classifies queries, applies metadata-filtered retrieval, handles tabular data as structured objects, detects contradictions across time periods, and self-verifies answers for grounding.

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
├── .antigravity/
│   └── skills/                # Shared skill playbooks (Antigravity + Claude Code)
│       ├── ingestion-skill.md
│       ├── langgraph-skill.md
│       └── eval-skill.md
├── ingestion/
│   ├── __init__.py
│   ├── parser.py              # PDF/HTML parsing with table extraction
│   ├── chunker.py             # Section-aware chunking with metadata
│   ├── embedder.py            # Dual embedding (raw + summary)
│   ├── upserter.py            # Pinecone upsert with metadata filters
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
│   ├── raw/                   # Original SEC filings, reports (PDF/HTML)
│   └── processed/             # Chunked JSON with metadata
├── pyproject.toml
├── Makefile
└── README.md
```

## Tech Stack

| Layer         | Technology                    | Notes                                          |
|---------------|-------------------------------|-------------------------------------------------|
| Ingestion     | Python (pymupdf, tabula-py)   | PDF parsing + structured table extraction       |
| Chunking      | Custom (ingestion/chunker.py) | Section-aware, ~500 tokens text, full tables    |
| Embedding     | text-embedding-3-large        | Dual: raw chunk + LLM-generated summary         |
| Vector DB     | Pinecone (free tier)          | Metadata filtering on company/year/section      |
| Orchestration | LangGraph                     | 5-node agentic state machine                    |
| LLM           | GPT-4o (synthesis/routing)    | Or Claude — switchable via env var              |
| Serving       | FastAPI + SSE                 | Streaming responses                             |
| Evaluation    | RAGAS                         | Faithfulness, relevancy, precision, recall       |
| Frontend      | React (optional)              | Chat UI with source citation panel              |

## Metadata Schema (MANDATORY for every chunk)

Every chunk upserted to Pinecone MUST carry this metadata:

```python
{
    "company": str,          # e.g., "AAPL", "MSFT"
    "fiscal_year": int,      # e.g., 2023, 2024
    "doc_type": str,         # One of: "10-K", "10-Q", "earnings_transcript", "market_report"
    "section": str,          # One of: "risk_factors", "mda", "financial_statements",
                             #         "business_overview", "executive_summary", "other"
    "is_table": bool,        # True if chunk is a structured table
    "source_url": str,       # EDGAR URL or report source
    "chunk_index": int,      # Position within the source document
    "page_number": int,      # Original page number in PDF
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
2. **Tables as structured objects** — Tables are chunked as markdown tables, NOT flattened into prose. The table_reasoner node receives them in structured format.
3. **Dual embedding** — Each chunk gets two vectors: one from the raw text, one from an LLM-generated summary. This improves retrieval for conceptual queries.
4. **Metadata-filtered retrieval** — The retriever node ALWAYS applies metadata filters (company + year at minimum) before similarity search. Never do unfiltered search.
5. **Contradiction detection** — For comparison queries, the retriever explicitly pulls chunks from BOTH time periods. The synthesizer prompt explicitly asks for agreement/disagreement analysis.

## Conventions

- Python 3.11+
- snake_case everywhere
- Type hints on all function signatures
- Pydantic models for all data structures (schemas.py)
- No hardcoded API keys — always env vars
- All prompts live in rag_agent/prompts/ as string templates
- Pinecone index name: `fin-compliance-rag`
- Pinecone namespace: `{company}_{fiscal_year}` (e.g., `AAPL_2024`)

## What NOT To Do

- Do NOT use LangChain's RetrievalQA or ConversationalRetrievalChain
- Do NOT flatten tables into paragraph text
- Do NOT hardcode API keys or model names
- Do NOT do unfiltered vector search (always apply metadata filters)
- Do NOT skip the verification node — every answer must be grounding-checked
- Do NOT embed tables and text with the same strategy — tables get summary-only embedding