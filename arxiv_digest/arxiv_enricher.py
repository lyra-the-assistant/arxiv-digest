"""
Enrich papers with metadata from the arxiv API.

For each matched paper, queries the arxiv API to retrieve the
``comment``, ``journal_ref``, and ``doi`` fields which are not
available in the RSS feed.  These fields are used downstream for
venue detection and project-page extraction.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests

from . import config
from .rss_fetcher import ArxivPaper

logger = logging.getLogger(__name__)

# Namespaces used in the arxiv Atom API response
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _batch_query(arxiv_ids: list[str]) -> dict[str, dict]:
    """
    Query the arxiv API for a batch of IDs and return a mapping of
    arxiv_id → {comment, journal_ref, doi}.
    """
    id_list = ",".join(arxiv_ids)
    params = {
        "id_list": id_list,
        "max_results": str(len(arxiv_ids)),
    }
    url = config.ARXIV_API_URL
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("arxiv API request failed: %s", e)
        return {}

    results: dict[str, dict] = {}
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        logger.error("Failed to parse arxiv API XML: %s", e)
        return {}

    for entry in root.findall("atom:entry", _NS):
        # Extract arxiv ID from <id> element
        entry_id_el = entry.find("atom:id", _NS)
        if entry_id_el is None or entry_id_el.text is None:
            continue
        # Format: http://arxiv.org/abs/2603.03380v1
        m = re.search(r"(\d{4}\.\d{4,5})", entry_id_el.text)
        if not m:
            continue
        aid = m.group(1)

        comment_el = entry.find("arxiv:comment", _NS)
        journal_el = entry.find("arxiv:journal_ref", _NS)
        doi_el = entry.find("arxiv:doi", _NS)

        results[aid] = {
            "comment": comment_el.text.strip() if comment_el is not None and comment_el.text else None,
            "journal_ref": journal_el.text.strip() if journal_el is not None and journal_el.text else None,
            "doi": doi_el.text.strip() if doi_el is not None and doi_el.text else None,
        }

    return results


def enrich_papers(papers: list[ArxivPaper]) -> None:
    """
    Enrich a list of papers in-place with metadata from the arxiv API.

    Populates each paper's ``comment``, ``journal_ref``, and ``doi`` fields.
    Queries are batched (up to ARXIV_API_BATCH_SIZE IDs per request) with
    a delay between requests to respect rate limits.
    """
    if not papers:
        return

    ids = [p.arxiv_id for p in papers]
    batch_size = config.ARXIV_API_BATCH_SIZE

    all_results: dict[str, dict] = {}

    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        logger.info(
            "Querying arxiv API for batch %d–%d of %d",
            i + 1,
            min(i + batch_size, len(ids)),
            len(ids),
        )
        results = _batch_query(batch)
        all_results.update(results)

        # Rate limit: sleep between batches (skip after last batch)
        if i + batch_size < len(ids):
            time.sleep(config.ARXIV_API_DELAY)

    # Merge results back into papers
    enriched_count = 0
    for paper in papers:
        if paper.arxiv_id in all_results:
            data = all_results[paper.arxiv_id]
            paper.comment = data.get("comment")
            paper.journal_ref = data.get("journal_ref")
            paper.doi = data.get("doi")
            enriched_count += 1

    logger.info(
        "Enriched %d / %d papers with arxiv API metadata",
        enriched_count,
        len(papers),
    )
