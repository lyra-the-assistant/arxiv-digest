"""
Fetch and parse the daily arxiv RSS feed.

Parses each RSS item into an ArxivPaper dataclass with fields extracted
from the RSS XML: title, abstract, announce type, categories, authors, etc.
"""

import logging
import re
from dataclasses import dataclass, field

import feedparser

from . import config

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Represents a single arxiv paper parsed from the RSS feed."""

    arxiv_id: str  # e.g., "2603.03380"
    title: str
    abstract: str
    announce_type: str  # new, replace, cross, replace-cross
    categories: list[str]  # e.g., ["cs.RO", "cs.AI"]
    authors: list[str]
    arxiv_url: str  # https://arxiv.org/abs/XXXX.XXXXX
    pdf_url: str  # https://arxiv.org/pdf/XXXX.XXXXX
    pub_date: str = ""  # RFC-822 date string from RSS

    # Enriched later by arxiv_enricher
    comment: str | None = None
    journal_ref: str | None = None
    doi: str | None = None

    # Populated by venue_detector / project page detection
    venue: str | None = None
    venue_type: str | None = None  # "conference", "journal", or None
    project_url: str | None = None

    # Populated by relevance scorer
    relevance_score: float = 0.0
    relevance_reason: str = ""
    matched_interests: list[str] = field(default_factory=list)


# Regex to parse the RSS description field.
# Format: "arXiv:2603.03380v1 Announce Type: new \nAbstract: ..."
_DESC_PATTERN = re.compile(
    r"arXiv:[\d.]+v\d+\s+Announce Type:\s*\S+\s*\n?"
    r"Abstract:\s*(.*)",
    re.DOTALL,
)

# Regex to extract arxiv ID from the GUID field.
# Format: "oai:arXiv.org:2603.03380v1" → "2603.03380"
_GUID_PATTERN = re.compile(r"oai:arXiv\.org:([\d.]+)v?\d*")

# Also handle direct IDs in link URLs
_LINK_ID_PATTERN = re.compile(r"arxiv\.org/abs/([\d.]+)")


def _parse_arxiv_id(entry: dict) -> str | None:
    """Extract the arxiv ID from a feedparser entry."""
    # Try GUID first
    guid = entry.get("id", "")
    m = _GUID_PATTERN.search(guid)
    if m:
        return m.group(1)
    # Fallback: parse from link
    link = entry.get("link", "")
    m = _LINK_ID_PATTERN.search(link)
    if m:
        return m.group(1)
    return None


def _parse_abstract(description: str) -> str:
    """Extract the abstract from the RSS description field."""
    m = _DESC_PATTERN.search(description)
    if m:
        return m.group(1).strip()
    # Fallback: return everything after "Abstract:" if present
    idx = description.find("Abstract:")
    if idx >= 0:
        return description[idx + 9:].strip()
    return description.strip()


def _parse_authors(entry: dict) -> list[str]:
    """Extract author names from a feedparser entry."""
    authors = []
    # feedparser may put authors in 'authors' list or 'author' string
    if "authors" in entry:
        for a in entry["authors"]:
            name = a.get("name", "").strip()
            if name:
                authors.append(name)
    elif "author" in entry:
        # Single author or comma-separated
        raw = entry["author"]
        for name in raw.split(","):
            name = name.strip()
            if name:
                authors.append(name)
    # Also check dc:creator via feedparser's author_detail or raw tags
    if not authors:
        # feedparser stores dc:creator in various ways
        raw_tags = entry.get("tags", [])
        # Actually dc:creator comes through as 'author' in feedparser
        pass
    return authors


def _parse_categories(entry: dict) -> list[str]:
    """Extract category tags from a feedparser entry."""
    categories = []
    for tag in entry.get("tags", []):
        term = tag.get("term", "").strip()
        if term:
            categories.append(term)
    return categories


def _parse_announce_type(entry: dict) -> str:
    """Extract the announce type from a feedparser entry."""
    # feedparser stores custom namespaced elements in various ways
    # Try the arxiv namespace first
    atype = entry.get("arxiv_announce_type", "")
    if atype:
        return atype.strip()
    # Fallback: parse from description
    desc = entry.get("summary", entry.get("description", ""))
    m = re.search(r"Announce Type:\s*(\S+)", desc)
    if m:
        return m.group(1).strip()
    return "new"


def fetch_papers() -> list[ArxivPaper]:
    """
    Fetch today's arxiv RSS feed and return a list of ArxivPaper objects.

    Returns all papers from the configured RSS channels (cs.RO, cs.CV, cs.AI, cs.LG).
    """
    logger.info("Fetching RSS feed from %s", config.RSS_FEED_URL)
    feed = feedparser.parse(config.RSS_FEED_URL)

    if feed.bozo and not feed.entries:
        logger.error("RSS feed parse error: %s", feed.bozo_exception)
        return []

    logger.info(
        "Feed fetched: %d items, last build: %s",
        len(feed.entries),
        feed.feed.get("updated", feed.feed.get("published", "unknown")),
    )

    papers = []
    seen_ids = set()

    for entry in feed.entries:
        arxiv_id = _parse_arxiv_id(entry)
        if not arxiv_id:
            logger.warning("Could not parse arxiv ID from entry: %s", entry.get("title", "?"))
            continue

        # Deduplicate within the feed (same paper may appear in multiple categories)
        if arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)

        title = entry.get("title", "").strip()
        description = entry.get("summary", entry.get("description", ""))
        abstract = _parse_abstract(description)
        announce_type = _parse_announce_type(entry)
        categories = _parse_categories(entry)
        authors = _parse_authors(entry)
        link = entry.get("link", f"https://arxiv.org/abs/{arxiv_id}")
        pub_date = entry.get("published", "")

        paper = ArxivPaper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            announce_type=announce_type,
            categories=categories,
            authors=authors,
            arxiv_url=link,
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            pub_date=pub_date,
        )
        papers.append(paper)

    logger.info("Parsed %d unique papers from RSS feed", len(papers))
    return papers
