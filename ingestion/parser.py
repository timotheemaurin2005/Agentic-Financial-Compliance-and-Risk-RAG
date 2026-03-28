"""Download and parse FOMC Statements & Minutes from the Federal Reserve website."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

from ingestion.schemas import (
    MEETING_DATES,
    minutes_url,
    statement_url,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_RAW_DIR = Path("data/raw")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36 "
    "FinComplianceRAG/1.0 (academic research)"
)
REQUEST_DELAY_SECS = 2.0


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_html(url: str) -> str | None:
    """Fetch HTML from *url*, returning the response text or None on failure."""
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            logger.warning("404 Not Found — skipping: %s", url)
            return None
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        logger.error("Failed to download %s: %s", url, exc)
        return None


def _save_html(html: str, filename: str) -> Path:
    """Persist raw HTML to data/raw/."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW_DIR / filename
    path.write_text(html, encoding="utf-8")
    logger.info("Saved %s (%d bytes)", path, len(html))
    return path


def _extract_article_body_statement(html: str) -> Tag | None:
    """Return the main article container for a **statement** page.

    Statement pages on the Fed site have *two* ``div.col-xs-12.col-sm-8.col-md-8``
    containers — the first holds date/title/share links, and the **second** holds
    the actual statement body with ``<p>`` tags containing the policy text.
    """
    soup = BeautifulSoup(html, "lxml")

    containers = soup.select("div.col-xs-12.col-sm-8.col-md-8")
    if len(containers) >= 2:
        return containers[1]  # second container has the body text
    if containers:
        return containers[0]

    # Fallback
    article = soup.find("article") or soup.select_one("#article")
    if article:
        return article

    logger.warning("Could not locate statement container — falling back to <body>")
    return soup.body


def _extract_article_body_minutes(html: str) -> Tag | None:
    """Return the main article container for a **minutes** page.

    Minutes pages use ``div.col-xs-12.col-sm-8`` (no ``col-md-8``).
    """
    soup = BeautifulSoup(html, "lxml")

    # Minutes pages use a slightly different layout
    container = soup.select_one("#article")
    if container:
        return container

    container = soup.select_one("div.col-xs-12.col-sm-8")
    if container:
        return container

    container = soup.select_one("div.col-xs-12.col-sm-8.col-md-8")
    if container:
        return container

    logger.warning("Could not locate minutes container — falling back to <body>")
    return soup.body


def parse_paragraphs(html: str, doc_type: str = "fomc_statement") -> list[str]:
    """Extract cleaned paragraph texts from a Fed HTML page.

    Returns a list of non-empty paragraph strings with whitespace normalised.
    For minutes, paragraphs that start with a bold section header will retain
    the header text at the beginning of the paragraph.
    """
    if doc_type == "fomc_statement":
        body = _extract_article_body_statement(html)
    else:
        body = _extract_article_body_minutes(html)

    if body is None:
        return []

    paragraphs: list[str] = []
    for p_tag in body.find_all("p"):
        text = p_tag.get_text(separator=" ", strip=True)
        if not text:
            continue

        # Filter out boilerplate
        if text.startswith("For media inquiries"):
            continue
        if text.startswith("Implementation Note issued"):
            continue
        if text.startswith("Last Update:"):
            continue

        paragraphs.append(text)
    return paragraphs


def parse_minutes_with_headers(html: str) -> list[tuple[str | None, str]]:
    """Parse minutes HTML, returning ``(header_or_none, paragraph_text)`` tuples.

    Section headers in FOMC minutes are typically ``<strong>`` tags at the start
    of a ``<p>`` element.  When we detect one, we split it out so the chunker
    can use it for section classification.
    """
    body = _extract_article_body_minutes(html)
    if body is None:
        return []

    results: list[tuple[str | None, str]] = []

    for p_tag in body.find_all("p"):
        text = p_tag.get_text(separator=" ", strip=True)
        if not text:
            continue

        # Skip boilerplate
        if text.startswith("Last Update:"):
            continue

        # Check if the paragraph starts with a <strong> tag (section header)
        first_child = next(
            (c for c in p_tag.children if hasattr(c, "name") and c.name == "strong"),
            None,
        )
        if first_child and first_child == p_tag.find("strong"):
            header_text = first_child.get_text(strip=True)
            # The remainder is the paragraph body
            remainder = text[len(header_text):].strip()
            if remainder:
                results.append((header_text, remainder))
            else:
                # Header-only paragraph
                results.append((header_text, ""))
        else:
            results.append((None, text))

    return results


# ---------------------------------------------------------------------------
# Public API — download all meetings
# ---------------------------------------------------------------------------

def download_all(
    meeting_dates: list[str] | None = None,
) -> dict[str, dict[str, str | None]]:
    """Download FOMC statements & minutes for every target meeting.

    Returns a nested dict keyed by meeting_date → doc_type → raw HTML string.
    ``None`` values indicate download failures (404 or network error).

    Side-effect: raw HTML files are saved under ``data/raw/``.
    """
    if meeting_dates is None:
        meeting_dates = MEETING_DATES

    results: dict[str, dict[str, str | None]] = {}

    for i, date in enumerate(meeting_dates):
        logger.info("━━━ Meeting %s ━━━", date)
        results[date] = {}

        # --- Statement ---
        s_url = statement_url(date)
        logger.info("Downloading statement: %s", s_url)
        s_html = _download_html(s_url)
        if s_html:
            _save_html(s_html, f"fomc_statement_{date}.html")
        results[date]["fomc_statement"] = s_html

        time.sleep(REQUEST_DELAY_SECS)

        # --- Minutes ---
        m_url = minutes_url(date)
        logger.info("Downloading minutes: %s", m_url)
        m_html = _download_html(m_url)
        if m_html:
            _save_html(m_html, f"fomc_minutes_{date}.html")
        results[date]["fomc_minutes"] = m_html

        # Delay between meetings (skip after the last one)
        if i < len(meeting_dates) - 1:
            time.sleep(REQUEST_DELAY_SECS)

    return results
