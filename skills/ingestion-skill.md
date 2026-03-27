# Ingestion Skill — Financial Document Parsing & Chunking

## Purpose

This skill governs how raw SEC filings, earnings transcripts, and market reports are parsed, chunked, embedded, and upserted into Pinecone. Follow these rules exactly.

## Document Sources

| Source           | Format    | How to Obtain                                      |
|------------------|-----------|-----------------------------------------------------|
| SEC 10-K / 10-Q  | HTML/PDF  | EDGAR full-text search API or direct filing URL     |
| Earnings calls   | PDF/TXT   | Company IR pages, Seeking Alpha, Motley Fool        |
| Market reports    | PDF       | Analyst reports, industry outlook PDFs              |

## Parsing Rules

### PDF Parsing
- Use `pymupdf` (fitz) as the primary PDF parser
- Extract text page-by-page, preserving page numbers in metadata
- For scanned PDFs, fall back to `pytesseract` OCR

### Table Extraction
- Use `tabula-py` for PDF tables (Java-backed, most accurate for financial tables)
- If tabula fails, fall back to `camelot` (lattice mode first, then stream mode)
- Output tables as **markdown format**, preserving column headers and alignment
- Example output:
  ```markdown
  | Metric          | FY2023   | FY2024   | Change   |
  |-----------------|----------|----------|----------|
  | Revenue ($M)    | 383,285  | 391,035  | +2.0%    |
  | Net Income ($M) | 96,995   | 93,736   | -3.4%    |
  ```
- NEVER flatten a table into a sentence like "Revenue was 383,285 in FY2023 and 391,035 in FY2024"

### Section Detection
- SEC filings have standard sections. Detect them using heading patterns:
  - "Item 1A" or "Risk Factors" → `risk_factors`
  - "Item 7" or "Management's Discussion and Analysis" → `mda`
  - "Item 8" or "Financial Statements" → `financial_statements`
  - "Item 1" or "Business" → `business_overview`
  - For non-SEC docs, use LLM classification to assign the closest section label
- Tag every chunk with its detected section

## Chunking Strategy

### Text Chunks
- Target size: **400–600 tokens** (use tiktoken `cl100k_base` to count)
- Split on section boundaries first, then on paragraph boundaries
- Never split mid-sentence
- Overlap: 50 tokens between consecutive chunks from the same section
- Each chunk inherits the section label of its parent section

### Table Chunks
- One table = one chunk (do NOT split tables across chunks)
- If a table exceeds 800 tokens, split by logical row groups (e.g., by year or category)
- Prepend a one-line context header: "Table: [section] from [company] [doc_type] [fiscal_year]"
- Set `is_table: true` in metadata

### Chunk Schema (Pydantic)
```python
from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    company: str              # Ticker symbol, e.g., "AAPL"
    fiscal_year: int          # e.g., 2024
    doc_type: str             # "10-K" | "10-Q" | "earnings_transcript" | "market_report"
    section: str              # "risk_factors" | "mda" | "financial_statements" | "business_overview" | "executive_summary" | "other"
    is_table: bool            # True if this chunk is a structured table
    source_url: str           # EDGAR URL or report source
    chunk_index: int          # Sequential position in the document
    page_number: int          # Original PDF page

class Chunk(BaseModel):
    id: str                   # Format: "{company}_{fiscal_year}_{doc_type}_{chunk_index}"
    text: str                 # The chunk content (text or markdown table)
    summary: str              # LLM-generated one-line summary
    metadata: ChunkMetadata
```

## Embedding Rules

### Dual Embedding
Every chunk gets TWO embeddings:
1. **Raw embedding** — embed `chunk.text` directly
2. **Summary embedding** — generate a one-line summary via LLM, then embed that

### Summary Generation Prompt
```
Summarise the following financial document excerpt in one sentence.
Focus on: what metric/topic is discussed, the company, the time period, and any numerical trend.

Excerpt:
{chunk_text}

One-sentence summary:
```

### Embedding Model
- Default: `text-embedding-3-large` (3072 dimensions)
- Batch size: 100 chunks per API call
- Normalise vectors (Pinecone handles this, but verify)

## Pinecone Upsert Rules

- Index name: `fin-compliance-rag`
- Namespace: `{company}_{fiscal_year}` (e.g., `AAPL_2024`)
- Each vector ID: `{chunk.id}_{raw|summary}` (two vectors per chunk)
- Metadata: flatten the ChunkMetadata dict — Pinecone requires flat key-value pairs
- Upsert in batches of 100 vectors
- After upsert, verify with a test query: retrieve top-1 for a known chunk and confirm metadata matches

## Error Handling

- If a PDF fails to parse: log the filename and skip, don't crash the pipeline
- If table extraction returns empty: log a warning, treat the page as text-only
- If embedding API rate-limits: implement exponential backoff (max 5 retries)
- If Pinecone upsert fails: retry once, then log and continue

## Testing Checklist

- [ ] Parse a 10-K PDF and verify section labels are correct
- [ ] Extract at least one table and confirm markdown format is valid
- [ ] Verify chunk token counts are in the 400–600 range
- [ ] Confirm dual embeddings produce two vectors per chunk in Pinecone
- [ ] Run a test query with metadata filter and verify correct results return