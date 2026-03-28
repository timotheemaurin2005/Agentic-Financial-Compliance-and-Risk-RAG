"""Section-aware chunking for FOMC Statements and Minutes."""

from __future__ import annotations

import logging
import re
from typing import Literal

import tiktoken

from ingestion.schemas import Chunk, ChunkMetadata, minutes_url, statement_url

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENC = tiktoken.get_encoding("cl100k_base")

TARGET_MIN_TOKENS = 400
TARGET_MAX_TOKENS = 600
OVERLAP_TOKENS = 50


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Section detection — FOMC Statements
# ---------------------------------------------------------------------------

_RATE_DECISION_PATTERNS = [
    re.compile(r"federal funds rate", re.IGNORECASE),
    re.compile(r"target range", re.IGNORECASE),
]
_VOTE_PATTERN = re.compile(r"voting\s+for", re.IGNORECASE)
_FORWARD_GUIDANCE_PATTERNS = [
    re.compile(r"in determining the (timing|extent)", re.IGNORECASE),
    re.compile(r"future (adjustments|path)", re.IGNORECASE),
    re.compile(r"additional (information|evidence)", re.IGNORECASE),
    re.compile(r"appropriate stance", re.IGNORECASE),
    re.compile(r"risks to .{0,40}goals", re.IGNORECASE),
    re.compile(r"(prepared|would be prepared) to adjust", re.IGNORECASE),
    re.compile(r"assess (incoming|additional)", re.IGNORECASE),
    re.compile(r"carefully assess", re.IGNORECASE),
    re.compile(r"extent and timing", re.IGNORECASE),
]


def _classify_statement_paragraph(text: str, idx: int, total: int) -> str:
    """Assign a section label to a single paragraph of an FOMC statement."""

    # Vote tally — usually the very last paragraph(s)
    if _VOTE_PATTERN.search(text):
        return "vote_tally"

    # Rate decision — mentions federal funds rate + target range
    if any(p.search(text) for p in _RATE_DECISION_PATTERNS):
        if "target range" in text.lower() or "basis point" in text.lower():
            return "rate_decision"

    # Forward guidance
    if any(p.search(text) for p in _FORWARD_GUIDANCE_PATTERNS):
        return "forward_guidance"

    # First ~40% of paragraphs → economic assessment (heuristic)
    if idx < max(2, int(total * 0.4)):
        return "economic_assessment"

    return "other"


# ---------------------------------------------------------------------------
# Section detection — FOMC Minutes
# ---------------------------------------------------------------------------

_MINUTES_SECTION_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"staff review of the economic situation", re.I), "staff_outlook"),
    (re.compile(r"staff review of the financial situation", re.I), "staff_outlook"),
    (re.compile(r"staff economic outlook", re.I), "staff_outlook"),
    (re.compile(r"participants.{0,10}views on current conditions", re.I), "participants_views_economy"),
    (re.compile(r"participants.{0,10}views on the outlook", re.I), "participants_views_economy"),
    (re.compile(r"participants.{0,20}discussion of policy", re.I), "participants_views_policy"),
    (re.compile(r"committee policy action", re.I), "committee_action"),
    (re.compile(r"voting for this action", re.I), "committee_action"),
    (re.compile(r"voting against this action", re.I), "dissenting_views"),
    (re.compile(r"voted against", re.I), "dissenting_views"),
]


def _classify_minutes_header(header_text: str) -> str:
    """Map a minutes section header to the canonical section label."""
    for pattern, label in _MINUTES_SECTION_MAP:
        if pattern.search(header_text):
            return label
    return "other"


# ---------------------------------------------------------------------------
# Vote tally → markdown table
# ---------------------------------------------------------------------------

def _parse_vote_tally(text: str) -> str | None:
    """Try to extract a markdown vote-tally table from a paragraph.

    Returns the markdown table string, or None if parsing fails.
    """
    for_match = re.search(
        r"[Vv]oting\s+for\s+(?:the\s+)?(?:this\s+)?(?:monetary\s+policy\s+)?action[:\s]*(.+?)(?:\.\s*[Vv]oting\s+against|$)",
        text,
        re.DOTALL,
    )
    against_match = re.search(
        r"[Vv]oting\s+against\s+(?:the\s+)?(?:this\s+)?(?:monetary\s+policy\s+)?action[:\s]*(.+?)(?:\.|$)",
        text,
        re.DOTALL,
    )

    for_members = for_match.group(1).strip().rstrip(".") if for_match else "Unknown"
    against_members = against_match.group(1).strip().rstrip(".") if against_match else "None"

    if for_members == "Unknown" and against_members == "None":
        return None

    table = (
        "| Vote | Members |\n"
        "|------|---------|\n"
        f"| For | {for_members} |\n"
        f"| Against | {against_members} |"
    )
    return table


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _split_text_to_target_tokens(
    text: str,
    *,
    min_tokens: int = TARGET_MIN_TOKENS,
    max_tokens: int = TARGET_MAX_TOKENS,
) -> list[str]:
    """Split *text* into chunks aiming for min–max tokens.

    Splits on sentence boundaries to avoid mid-sentence breaks.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_count = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        if current_count + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Overlap: keep trailing sentences that fit within OVERLAP_TOKENS
            overlap_chunk: list[str] = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_tok = _count_tokens(s)
                if overlap_count + s_tok > OVERLAP_TOKENS:
                    break
                overlap_chunk.insert(0, s)
                overlap_count += s_tok
            current_chunk = overlap_chunk + [sent]
            current_count = overlap_count + sent_tokens
        else:
            current_chunk.append(sent)
            current_count += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# Public API — chunk a statement
# ---------------------------------------------------------------------------

def chunk_statement(
    paragraphs: list[str],
    meeting_date: str,
) -> list[Chunk]:
    """Chunk an FOMC statement (one chunk per paragraph + vote table chunk).

    Each paragraph becomes its own chunk with the appropriate section label.
    """
    doc_type = "fomc_statement"
    src_url = statement_url(meeting_date)
    year = int(meeting_date[:4])
    chunks: list[Chunk] = []
    idx = 0

    for p_idx, para in enumerate(paragraphs):
        section = _classify_statement_paragraph(para, p_idx, len(paragraphs))

        # If this is the vote tally, also create a table chunk
        if section == "vote_tally":
            table_md = _parse_vote_tally(para)
            if table_md:
                table_chunk = Chunk(
                    id=f"{doc_type}_{meeting_date}_{idx}",
                    text=f"Table: vote_tally from FOMC statement {meeting_date}\n\n{table_md}",
                    metadata=ChunkMetadata(
                        doc_type=doc_type,
                        meeting_date=meeting_date,
                        year=year,
                        section="vote_tally",
                        is_table=True,
                        source_url=src_url,
                        chunk_index=idx,
                    ),
                )
                chunks.append(table_chunk)
                idx += 1

        # Text chunk for the paragraph
        chunk = Chunk(
            id=f"{doc_type}_{meeting_date}_{idx}",
            text=para,
            metadata=ChunkMetadata(
                doc_type=doc_type,
                meeting_date=meeting_date,
                year=year,
                section=section,
                is_table=False,
                source_url=src_url,
                chunk_index=idx,
            ),
        )
        chunks.append(chunk)
        idx += 1

    logger.info(
        "Chunked statement %s → %d chunks",
        meeting_date,
        len(chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Public API — chunk minutes
# ---------------------------------------------------------------------------

def chunk_minutes(
    header_paragraphs: list[tuple[str | None, str]],
    meeting_date: str,
) -> list[Chunk]:
    """Chunk FOMC minutes using section-header detection.

    Parameters
    ----------
    header_paragraphs : list of (header_or_none, paragraph_text) tuples
        Output from ``parser.parse_minutes_with_headers``.
    meeting_date : str
        ISO-format meeting date.

    Long sections are split at paragraph/sentence boundaries to stay within
    the 400–600 token target range.
    """
    doc_type = "fomc_minutes"
    src_url = minutes_url(meeting_date)
    year = int(meeting_date[:4])
    chunks: list[Chunk] = []
    idx = 0

    # ----- assign section labels to each paragraph -----
    current_section = "other"
    labelled: list[tuple[str, str]] = []  # (section, text)

    for header, text in header_paragraphs:
        # If there's an explicit header, try to classify it
        if header:
            detected = _classify_minutes_header(header)
            if detected != "other":
                current_section = detected
            # If there's body text after the header, include it
            if text:
                labelled.append((current_section, text))
            continue

        # Plain paragraph without header
        if text:
            labelled.append((current_section, text))

    # ----- group contiguous paragraphs of the same section -----
    section_groups: list[tuple[str, list[str]]] = []
    for section, text in labelled:
        if section_groups and section_groups[-1][0] == section:
            section_groups[-1][1].append(text)
        else:
            section_groups.append((section, [text]))

    # ----- chunk each group -----
    for section, texts in section_groups:
        combined = "\n\n".join(texts)
        token_count = _count_tokens(combined)

        if token_count <= TARGET_MAX_TOKENS:
            # Whole group fits in one chunk — only create if non-trivial
            if token_count > 0:
                chunks.append(
                    Chunk(
                        id=f"{doc_type}_{meeting_date}_{idx}",
                        text=combined,
                        metadata=ChunkMetadata(
                            doc_type=doc_type,
                            meeting_date=meeting_date,
                            year=year,
                            section=section,
                            is_table=False,
                            source_url=src_url,
                            chunk_index=idx,
                        ),
                    )
                )
                idx += 1
        else:
            # Need to split
            sub_chunks = _split_text_to_target_tokens(combined)
            for sub in sub_chunks:
                chunks.append(
                    Chunk(
                        id=f"{doc_type}_{meeting_date}_{idx}",
                        text=sub,
                        metadata=ChunkMetadata(
                            doc_type=doc_type,
                            meeting_date=meeting_date,
                            year=year,
                            section=section,
                            is_table=False,
                            source_url=src_url,
                            chunk_index=idx,
                        ),
                    )
                )
                idx += 1

    logger.info(
        "Chunked minutes %s → %d chunks (sections: %s)",
        meeting_date,
        len(chunks),
        {s for s, _ in section_groups},
    )
    return chunks

