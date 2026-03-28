# Agentic Financial Compliance & Risk RAG

An agentic Retrieval-Augmented Generation system that analyses FOMC (Federal Open Market Committee) documents to detect policy shifts, contradictions, and evolving monetary policy signals across meetings.

Unlike simple RAG pipelines, this system uses a **LangGraph state machine** with 5 specialised nodes: routing, retrieval, table reasoning, synthesis, and self-verification — to handle nuanced questions about Federal Reserve communications.

## What It Does

**Input:** Natural language questions about Fed monetary policy

**Output:** Grounded, cited answers with contradiction detection

Example queries:
- *"How did the FOMC's forward guidance language change between September 2024 and January 2025?"*
- *"Did the January 2025 minutes reveal any disagreement that the statement didn't mention?"*
- *"Track the evolution of inflation risk language across the last 5 meetings"*

The system detects subtle but deliberate shifts in Fed language — "some participants" vs "most participants", "at or near its peak" vs "somewhat above" — that signal real policy changes.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
              ┌────────────────┐
              │     Router     │  Classifies: factual / numerical /
              │                │  comparison / contradiction
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │   Retriever    │  Filtered Pinecone search with
              │                │  metadata (meeting date, doc type)
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ Table Reasoner │  Structured analysis of vote
              │  (conditional) │  tallies & economic projections
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │  Synthesizer   │  Generates cited answer with
              │                │  contradiction detection
              └───────┬────────┘
                      ▼
              ┌────────────────┐     ✗ (retry up to 2x)
              │   Verifier     │────────────────┐
              │                │                │
              └───────┬────────┘                │
                      │ ✓ (grounded)            │
                      ▼                         │
              ┌────────────────┐        ┌───────┴──────┐
              │  Final Answer  │        │  Re-retrieve  │
              │  with citations│        │  & re-generate│
              └────────────────┘        └──────────────┘
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | **LangGraph** | 5-node agentic state machine with conditional routing and retry loops |
| Vector DB | **Pinecone** | Metadata-filtered similarity search across FOMC documents |
| Embeddings | **text-embedding-3-large** | Dual embedding strategy (raw text + LLM-generated summaries) |
| LLM | **GPT-4o** | Query routing, summarisation, synthesis, and verification |
| Backend | **FastAPI** | REST API with SSE streaming |
| Evaluation | **RAGAS** | Faithfulness, relevancy, context precision, and context recall |
| Ingestion | **BeautifulSoup + tiktoken** | Section-aware parsing and chunking of Fed documents |

## Document Corpus

The system ingests FOMC Statements and Meeting Minutes from 5 consecutive meetings:

| Meeting | Key Context |
|---------|------------|
| September 2024 | First rate cut in the easing cycle |
| November 2024 | Post-election meeting |
| December 2024 | Updated economic projections |
| January 2025 | First meeting of 2025 |
| March 2025 | Latest available |

This provides 10 documents (5 statements + 5 minutes) with rich contradiction-detection opportunities as the Fed's rate path evolved.

## Key Features

### Contradiction Detection
The system explicitly identifies when sources disagree — whether that's forward guidance shifting between meetings, or a statement's consensus language masking the disagreement revealed in the minutes.

### Fed Language Sensitivity
Trained to recognise deliberate policy signals in Fed communications:
- **Qualifier shifts:** "a few participants" → "some" → "most" → "all"
- **Certainty shifts:** "noted" → "judged" → "agreed"
- **Direction shifts:** "further tightening" → "maintaining" → "prepared to adjust"

### Dual Embedding Strategy
Each chunk is embedded twice — once from raw text, once from an LLM-generated summary — improving retrieval for conceptual queries that use different language than the source documents.

### Self-Verification Loop
Every answer passes through a grounding check. If the verifier finds unsupported claims, the system automatically re-retrieves with a refined query (up to 2 retries) before delivering the answer.

## Evaluation Results

Evaluated across 30 gold-standard questions using RAGAS, covering factual lookups, numerical comparisons, cross-document analysis, and contradiction detection.

| Metric | Score | Target |
|--------|-------|--------|
| Faithfulness | 0.62 | 0.85 |
| Answer Relevancy | 0.71 | 0.80 |
| Context Precision | 0.49 | 0.75 |
| Context Recall | 0.34 | 0.70 |

**Analysis:** The primary bottleneck is retrieval recall — the system retrieves approximately 34% of relevant context, which cascades into lower faithfulness and precision scores. Factual single-document queries (rate decisions, vote tallies) perform strongly, while cross-meeting contradiction queries suffer most due to namespace-separated storage requiring multiple retrieval passes across meeting dates.

**Failure Mode Breakdown** (8 of 30 questions scored below 0.6 on at least one metric):

| Failure Mode | Count | Root Cause |
|-------------|-------|------------|
| Hallucination | 5 | Synthesizer generates claims beyond retrieved context |
| Retrieval miss | 2 | Relevant chunks not retrieved due to namespace isolation |
| Noise retrieval | 2 | Irrelevant chunks diluting context window |

**Identified Improvements:**
- Increase `top_k` from 10 to 20 for comparison and contradiction queries to improve recall
- Implement cross-namespace retrieval to avoid missing one side of a multi-meeting comparison
- Tune chunk overlap from 50 to 100 tokens to reduce information loss at chunk boundaries
- Add query expansion in the retriever node to catch paraphrased concepts

## Evaluation Framework

The system is evaluated using **RAGAS** across 30 gold-standard questions spanning 4 query types:

| Query Type | Count | What It Tests |
|-----------|-------|--------------|
| Factual | 8 | Single-document lookups (rate decisions, vote tallies) |
| Numerical | 8 | Calculations and basis point comparisons |
| Comparison | 7 | Cross-document analysis (statement vs minutes) |
| Contradiction | 7 | Detecting language shifts across meetings |

An ablation study framework (`eval/ablation.py`) is included to sweep chunk size, top-k retrieval, embedding model, and LLM choice to identify optimal configurations and quantify tradeoffs.

## Project Structure

```
├── ingestion/          # Document download, parsing, chunking, embedding, Pinecone upsert
├── rag_agent/          # LangGraph state machine, nodes, and prompts
│   ├── nodes/          # Router, Retriever, Table Reasoner, Synthesizer, Verifier
│   └── prompts/        # Prompt templates for each node
├── api/                # FastAPI serving layer with streaming
├── eval/               # RAGAS evaluation harness and ablation runner
├── skills/             # Agent skill playbooks (ingestion, langgraph, eval)
├── data/
│   ├── raw/            # Original FOMC HTML documents
│   └── processed/      # Chunked JSON with metadata
├── CLAUDE.md           # Project spec and conventions
└── pyproject.toml      # Dependencies and build config
```

## Quick Start

```bash
# Clone
git clone https://github.com/timotheemaurin2005/Agentic-Financial-Compliance-and-Risk-RAG.git
cd Agentic-Financial-Compliance-and-Risk-RAG

# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your Pinecone, OpenAI keys

# Run ingestion pipeline
python -m ingestion.pipeline

# Start the API
uvicorn api.main:app --reload

# Run evaluation
python -m eval.run
```

## API Usage

**Query the system:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How did inflation language change between September 2024 and January 2025?"}'
```

**Response:**
```json
{
  "answer": "Cited answer with [1], [2] passage references...",
  "sources": [
    {
      "meeting_date": "2024-09-18",
      "doc_type": "fomc_statement",
      "section": "forward_guidance",
      "source_url": "https://www.federalreserve.gov/...",
      "score": 0.82
    }
  ],
  "contradiction_detected": true,
  "confidence": 0.91
}
```

**Stream responses (SSE):**
```bash
curl -N -X POST http://localhost:8000/stream/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Did the January 2025 minutes reveal disagreement?"}'
```

**List ingested documents:**
```bash
curl http://localhost:8000/documents
```

## Author

**Timothée Maurin** — BSc Data Science, University College London

Built as a portfolio project demonstrating production-grade RAG system design with agentic orchestration, structured evaluation, and domain-specific document understanding.

## Licence

MIT
