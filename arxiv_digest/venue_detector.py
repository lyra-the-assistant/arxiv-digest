"""
Detect publication venue and project page URL from arxiv metadata.

Uses the ``journal_ref`` and ``comment`` fields (populated by the enricher)
to identify whether a paper has been accepted at a conference or journal,
and to extract project page URLs.
"""

import logging
import re

from . import config
from .rss_fetcher import ArxivPaper

logger = logging.getLogger(__name__)

# Pre-compiled patterns for venue detection from comments
_ACCEPTED_PATTERN = re.compile(
    r"(?:accepted|published|appear(?:ing|s)?)\s+"
    r"(?:at|by|in|to|for)\s+"
    r"(?:the\s+)?"
    r"(.{3,80}?)(?:\.|,|;|\s*$)",
    re.IGNORECASE,
)

# Pattern: "VenueName Year" or "VenueName'Year" (e.g., "ICRA 2025", "NeurIPS'24")
_VENUE_YEAR_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?)\s*['\u2019]?\s*(\d{2,4})\b"
)

# URL extraction from comment text
_URL_PATTERN = re.compile(r"https?://[^\s<>\"',;)]+")

# Venues whose abbreviations are too short / ambiguous for standalone matching
_CONTEXT_REQUIRED = config.VENUE_CONTEXT_REQUIRED


def _match_venue_in_text(text: str) -> tuple[str | None, str | None]:
    """
    Try to find a known venue in a text string.

    Returns (venue_display_name, venue_type) or (None, None).
    """
    if not text:
        return None, None

    # 1. Check long venue names first (less ambiguous)
    for long_name, (abbrev, vtype) in config.VENUE_LONG_NAMES.items():
        if long_name.lower() in text.lower():
            return f"{abbrev} ({long_name})", vtype

    # 2. Check abbreviation patterns
    for abbrev, (full_name, vtype) in config.VENUE_PATTERNS.items():
        if abbrev in _CONTEXT_REQUIRED:
            # Require "accepted/published" context or "VenueYear" pattern
            pattern = re.compile(
                r"(?:accepted|published|appear)\s+.*?\b" + re.escape(abbrev) + r"\b",
                re.IGNORECASE,
            )
            if pattern.search(text):
                return f"{abbrev} ({full_name})", vtype
            # Also check "VENUE YEAR" pattern
            pattern2 = re.compile(
                r"\b" + re.escape(abbrev) + r"\s*['\u2019]?\s*\d{2,4}\b"
            )
            if pattern2.search(text):
                return f"{abbrev} ({full_name})", vtype
        else:
            # For unambiguous abbreviations, a word-boundary match suffices
            if re.search(r"\b" + re.escape(abbrev) + r"\b", text):
                return f"{abbrev} ({full_name})", vtype

    return None, None


def detect_venue(paper: ArxivPaper) -> None:
    """
    Detect the publication venue for a paper (in-place).

    Checks ``journal_ref``, then ``comment`` for venue information.
    Sets ``paper.venue`` and ``paper.venue_type``.
    """
    # 1. Check journal_ref (most authoritative)
    if paper.journal_ref:
        venue, vtype = _match_venue_in_text(paper.journal_ref)
        if venue:
            paper.venue = venue
            paper.venue_type = vtype
            return
        # If journal_ref exists but doesn't match known venues,
        # use the raw journal_ref text
        paper.venue = paper.journal_ref
        paper.venue_type = "journal"  # journal_ref usually means journal
        return

    # 2. Check comment field
    if paper.comment:
        # First try the "Accepted at/in ..." pattern
        m = _ACCEPTED_PATTERN.search(paper.comment)
        if m:
            accepted_text = m.group(1).strip()
            venue, vtype = _match_venue_in_text(accepted_text)
            if venue:
                paper.venue = venue
                paper.venue_type = vtype
                return
            # Use the raw accepted text
            paper.venue = accepted_text
            # Guess type based on keywords
            if any(
                kw in accepted_text.lower()
                for kw in ("journal", "transaction", "letter")
            ):
                paper.venue_type = "journal"
            else:
                paper.venue_type = "conference"
            return

        # Try matching venue abbreviations directly in comment
        venue, vtype = _match_venue_in_text(paper.comment)
        if venue:
            paper.venue = venue
            paper.venue_type = vtype
            return

    # 3. Default: no venue detected
    paper.venue = None
    paper.venue_type = None


def detect_project_url(paper: ArxivPaper) -> None:
    """
    Extract project page URL from the paper's comment field (in-place).

    Looks for URLs in the comment that are NOT arxiv.org links.
    Prioritises URLs that look like project pages (github.io, project-specific).
    """
    if not paper.comment:
        paper.project_url = None
        return

    urls = _URL_PATTERN.findall(paper.comment)
    if not urls:
        paper.project_url = None
        return

    # Filter out arxiv links and common non-project URLs
    candidates = []
    for url in urls:
        # Clean trailing punctuation
        url = url.rstrip(".,;:)")
        lower = url.lower()
        # Skip arxiv and common license URLs
        if "arxiv.org" in lower:
            continue
        if "creativecommons.org" in lower:
            continue
        if "license" in lower:
            continue
        candidates.append(url)

    if not candidates:
        paper.project_url = None
        return

    # Prioritize: github.io pages, then github repos, then others
    for url in candidates:
        if "github.io" in url.lower():
            paper.project_url = url
            return

    # Check for explicit "project page" or "website" context
    comment_lower = paper.comment.lower()
    for url in candidates:
        # Check if there's "project" or "website" or "page" near this URL
        idx = paper.comment.find(url)
        if idx >= 0:
            context = comment_lower[max(0, idx - 50) : idx + len(url) + 20]
            if any(kw in context for kw in ("project", "website", "page", "homepage")):
                paper.project_url = url
                return

    # Fallback: first non-arxiv URL
    paper.project_url = candidates[0]


def detect_all(papers: list[ArxivPaper]) -> None:
    """
    Detect venue and project URL for all papers (in-place).
    """
    venues_found = 0
    projects_found = 0
    for paper in papers:
        detect_venue(paper)
        detect_project_url(paper)
        if paper.venue:
            venues_found += 1
        if paper.project_url:
            projects_found += 1

    logger.info(
        "Venue detection: %d / %d papers have venue info",
        venues_found,
        len(papers),
    )
    logger.info(
        "Project page detection: %d / %d papers have project URLs",
        projects_found,
        len(papers),
    )
