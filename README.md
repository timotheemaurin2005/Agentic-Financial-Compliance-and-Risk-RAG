# Financial Compliance & Risk RAG

A Python monorepo for ingesting, indexing, and querying financial compliance documents using a RAG (Retrieval-Augmented Generation) pipeline.

## Project Layout

```
.
├── ingestion/        # Document parsing & vector ingestion
├── rag_agent/        # LangGraph-based RAG agent
├── api/              # FastAPI service
├── eval/             # RAGAS-based evaluation suite
├── data/
│   ├── raw/          # Source documents (PDFs, Excel, …)
│   └── processed/    # Chunked / embedded artefacts
├── pyproject.toml
└── Makefile
```

## Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install all dependencies
make install

# 3. Run the ingestion pipeline
make ingest

# 4. Start the API
make serve

# 5. Run the eval suite
make eval
```

## Requirements

- Python ≥ 3.11
- A Pinecone API key (`PINECONE_API_KEY`)
- An OpenAI API key (`OPENAI_API_KEY`)
