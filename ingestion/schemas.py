"""Pydantic models for ingestion pipeline data structures."""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Target FOMC meetings (from CLAUDE.md / ingestion-skill.md)
# ---------------------------------------------------------------------------
MEETING_DATES: list[str] = [
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    "2025-01-29",
    "2025-03-19",
]


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _date_to_compact(meeting_date: str) -> str:
    """Convert ISO date '2025-01-29' → '20250129'."""
    return meeting_date.replace("-", "")


def statement_url(meeting_date: str) -> str:
    """Return the Federal Reserve URL for an FOMC statement."""
    compact = _date_to_compact(meeting_date)
    return f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{compact}a.htm"


def minutes_url(meeting_date: str) -> str:
    """Return the Federal Reserve URL for FOMC minutes."""
    compact = _date_to_compact(meeting_date)
    return f"https://www.federalreserve.gov/monetarypolicy/fomcminutes{compact}.htm"


# ---------------------------------------------------------------------------
# Metadata schema  (matches ingestion-skill.md exactly)
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk upserted to Pinecone."""

    doc_type: str          # "fomc_statement" | "fomc_minutes"
    meeting_date: str      # ISO format: "2025-01-29"
    year: int              # e.g., 2025
    section: str           # See section-detection tables in ingestion-skill.md
    is_table: bool         # True if this chunk is a structured table
    source_url: str        # Federal Reserve URL
    chunk_index: int       # Sequential position within the document

    def to_pinecone_dict(self) -> dict:
        """Flatten to a dict suitable for Pinecone metadata (flat key-value)."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A single chunk ready for embedding and upsert."""

    id: str                # Format: "{doc_type}_{meeting_date}_{chunk_index}"
    text: str              # The chunk content (text or markdown table)
    summary: str = ""      # LLM-generated one-line summary (filled by embedder)
    metadata: ChunkMetadata
