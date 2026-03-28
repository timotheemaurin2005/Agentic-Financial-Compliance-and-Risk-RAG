# Ingestion Skill — FOMC Document Parsing & Chunking

## Purpose

This skill governs how FOMC Statements and Meeting Minutes are downloaded, parsed, chunked, embedded, and upserted into Pinecone. Follow these rules exactly.

## Document Sources

| Document         | Format     | URL Pattern                                                              |
|------------------|------------|--------------------------------------------------------------------------|
| FOMC Statement   | HTML       | `https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm` |
| FOMC Minutes     | HTML       | `https://www.federalreserve.gov/monetarypolicy/fomcminutes{YYYYMMDD}.htm`          |

### Target Meetings
| Meeting Date | Statement URL Date | Notes                    |
|--------------|--------------------|--------------------------|
| 2024-09-18   | 20240918           | First rate cut in cycle  |
| 2024-11-07   | 20241107           | Post-election meeting    |
| 2024-12-18   | 20241218           | December projections     |
| 2025-01-29   | 20250129           | First meeting of 2025    |
| 2025-03-19   | 20250319           | Latest available         |

### Download Rules
- Use `requests` + `BeautifulSoup` to scrape HTML content from the Fed website
- Save raw HTML to `data/raw/{doc_type}_{meeting_date}.html` (e.g., `fomc_statement_2025-01-29.html`)
- Respect rate limits: add a 2-second delay between requests
- Set a proper User-Agent header
- These are public US government documents — no copyright restrictions

## Parsing Rules

### HTML Parsing
- Use `BeautifulSoup` (not pymupdf — these are HTML pages, not PDFs)
- Extract the main article content (typically inside a `<div class="col-xs-12 col-sm-8 col-md-8">` or similar container)
- Strip navigation, headers, footers, and sidebar content
- Preserve paragraph structure

### Section Detection — FOMC Statements
Statements are short and follow a consistent structure. Detect sections by content patterns:

| Section              | Detection Pattern                                                    |
|----------------------|----------------------------------------------------------------------|
| `economic_assessment`| First 1–2 paragraphs: discusses jobs, inflation, economic activity   |
| `rate_decision`      | Paragraph containing "federal funds rate" and a target range         |
| `forward_guidance`   | Paragraph containing language about future policy direction          |
| `vote_tally`         | Final paragraph listing "Voting for/against" with member names       |

### Section Detection — FOMC Minutes
Minutes have explicit section headers. Map them:

| Minutes Header (approximate)                        | Section Label                    |
|-----------------------------------------------------|----------------------------------|
| "Staff Review of the Economic Situation"             | `staff_outlook`                  |
| "Staff Review of the Financial Situation"            | `staff_outlook`                  |
| "Staff Economic Outlook"                             | `staff_outlook`                  |
| "Participants' Views on Current Conditions"          | `participants_views_economy`     |
| "Participants' Views on the Outlook"                 | `participants_views_economy`     |
| "Participants' Discussion of Policy"                 | `participants_views_policy`      |
| "Committee Policy Action"                            | `committee_action`               |
| Dissent paragraphs or "voted against"                | `dissenting_views`               |

If a section header doesn't match any pattern, label it `other`.

### Table Extraction
- Vote tallies in statements: parse the "Voting for" / "Voting against" text into a markdown table:
  ```markdown
  | Vote     | Members                                                |
  |----------|--------------------------------------------------------|
  | For      | Powell, Williams, Barr, Bowman, Cook, Goolsbee, ...    |
  | Against  | None                                                   |
  ```
- Economic projection tables in minutes (if present): preserve as markdown tables
- Set `is_table: true` in metadata for these chunks

## Chunking Strategy

### Text Chunks
- Target size: **400–600 tokens** (use tiktoken `cl100k_base` to count)
- Split on section boundaries first, then on paragraph boundaries
- Never split mid-sentence
- Overlap: 50 tokens between consecutive chunks from the same section
- Each chunk inherits the section label of its parent section

### FOMC Statement Special Rule
- Statements are short (typically 4–6 paragraphs). Each paragraph should be its own chunk with the appropriate section label.
- Do NOT merge paragraphs from different sections into one chunk.

### Table Chunks
- One table = one chunk (do NOT split tables across chunks)
- Prepend a one-line context header: "Table: [section] from FOMC [doc_type] [meeting_date]"
- Set `is_table: true` in metadata

### Chunk Schema (Pydantic)
```python
from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    doc_type: str             # "fomc_statement" | "fomc_minutes"
    meeting_date: str         # ISO format: "2025-01-29"
    year: int                 # e.g., 2025
    section: str              # See section detection tables above
    is_table: bool            # True if this chunk is a structured table
    source_url: str           # Federal Reserve URL
    chunk_index: int          # Sequential position in the document

class Chunk(BaseModel):
    id: str                   # Format: "{doc_type}_{meeting_date}_{chunk_index}"
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
Summarise the following FOMC document excerpt in one sentence.
Focus on: the policy topic discussed, the meeting date, and any specific language about rates, inflation, or employment.
Pay special attention to qualifier words (some, most, several, a few participants) as these signal the degree of consensus.

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
- Namespace: `fomc_{meeting_date}` (e.g., `fomc_2025-01-29`)
- Each vector ID: `{chunk.id}_{raw|summary}` (two vectors per chunk)
- Metadata: flatten the ChunkMetadata dict — Pinecone requires flat key-value pairs
- Upsert in batches of 100 vectors
- After upsert, verify with a test query: retrieve top-1 for a known chunk and confirm metadata matches

## Error Handling

- If a Fed URL returns 404: log the URL and skip, don't crash the pipeline
- If section detection fails: label the chunk as `other`, log a warning
- If embedding API rate-limits: implement exponential backoff (max 5 retries)
- If Pinecone upsert fails: retry once, then log and continue

## Pipeline Execution Order

```
1. Download HTML from Fed website → save to data/raw/
2. Parse HTML with BeautifulSoup → extract article content
3. Detect sections → label each paragraph/block
4. Chunk text → 400-600 tokens per chunk with section labels
5. Extract tables → separate table chunks with is_table=true
6. Generate summaries → one-line LLM summary per chunk
7. Embed → dual vectors (raw + summary) per chunk
8. Upsert → Pinecone with metadata filters
9. Verify → test query to confirm retrieval works
```

## Testing Checklist

- [ ] All 10 documents download successfully (5 statements + 5 minutes)
- [ ] Section labels are correctly assigned for at least one statement and one minutes document
- [ ] Vote tally is parsed as a markdown table
- [ ] Chunk token counts are in the 400–600 range
- [ ] Dual embeddings produce two vectors per chunk in Pinecone
- [ ] Test query with metadata filter `{"doc_type": "fomc_statement", "meeting_date": "2025-01-29"}` returns correct results
- [ ] Namespaces are correctly separated per meeting date