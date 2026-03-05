"""
LLM-based relevance scoring for arxiv papers.

Two-stage pipeline:
  1. **Keyword pre-filter** — cheap pass to reduce ~700 papers to ~150
     candidates, saving LLM tokens.
  2. **LLM evaluation** — each candidate is judged by an LLM against the
     user's research interest descriptions.  The LLM returns a relevance
     score (1-5) and a short reason.

When no LLM API is configured (LLM_API_KEY is empty), the module falls
back to keyword-only scoring with a warning.

In production the upstream agent may replace the built-in LLM call with
its own sub-agent orchestration; the interface is the same: each paper
gets ``relevance_score``, ``matched_interests``, and ``relevance_reason``.
"""

import json
import logging
import re
import time
from typing import Callable

from .config import (
    CATEGORY_BONUS,
    INTEREST_PROFILES,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_BATCH_SIZE,
    LLM_MODEL,
    MAX_RESULTS,
    PREFILTER_THRESHOLD,
    RELEVANCE_THRESHOLD,
    TITLE_MULTIPLIER,
)
from .rss_fetcher import ArxivPaper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 1 — keyword pre-filter (cheap, local)
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _count_matches(text: str, keyword: str) -> int:
    if len(keyword) <= 4:
        return len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
    return text.count(keyword)


def _keyword_score(paper: ArxivPaper) -> float:
    """Return a raw keyword score used for the pre-filter stage."""
    title_norm = _normalize(paper.title)
    abstract_norm = _normalize(paper.abstract)
    total = 0.0
    for profile in INTEREST_PROFILES:
        for keyword, weight in profile["keywords"]:
            kw = keyword.lower()
            tc = _count_matches(title_norm, kw)
            ac = _count_matches(abstract_norm, kw)
            if tc or ac:
                total += weight * (tc * TITLE_MULTIPLIER + ac)
    for cat in paper.categories:
        total += CATEGORY_BONUS.get(cat, 0.0)
    return total


def prefilter(papers: list[ArxivPaper]) -> list[ArxivPaper]:
    """
    Cheap keyword pass to narrow candidates before the LLM stage.

    Uses a *lower* threshold than the old keyword-only scorer so that
    borderline papers still get a chance with the LLM.
    """
    candidates = []
    for p in papers:
        s = _keyword_score(p)
        p.relevance_score = s  # stash for logging
        if s >= PREFILTER_THRESHOLD:
            candidates.append(p)
    candidates.sort(key=lambda p: p.relevance_score, reverse=True)
    logger.info(
        "Keyword pre-filter: %d / %d papers passed (threshold=%.1f)",
        len(candidates),
        len(papers),
        PREFILTER_THRESHOLD,
    )
    return candidates


# ---------------------------------------------------------------------------
# Stage 2 — LLM evaluation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a research paper relevance evaluator. You will be given a list of \
arxiv papers (title + abstract) and a set of research interest descriptions. \
For each paper, decide whether it is relevant to ANY of the interests.

Research interests:
{interests}

Evaluation criteria — a paper is relevant if:
- It directly addresses one or more of the research interests, OR
- It proposes methods, benchmarks, or systems clearly useful for the interests.

A paper is NOT relevant if it merely shares a vague keyword but addresses a \
fundamentally different problem domain.

Target balanced precision and recall: do not be too strict or too loose."""

_USER_PROMPT = """\
Evaluate the following {n} papers. For each paper return a JSON object with:
- "id": the arxiv ID (string)
- "score": integer 1-5 (1=irrelevant, 3=borderline, 5=highly relevant)
- "interests": list of matched interest names (empty if irrelevant)
- "reason": one-sentence explanation of why it is or isn't relevant

Return ONLY a JSON array of these objects, no other text.

Papers:
{papers}"""


def _format_interests() -> str:
    parts = []
    for p in INTEREST_PROFILES:
        parts.append(f'- **{p["name"]}**: {p["description"]}')
    return "\n".join(parts)


def _format_paper_block(paper: ArxivPaper) -> str:
    abstract = paper.abstract[:1500]  # cap to save tokens
    return (
        f"[{paper.arxiv_id}] {paper.title}\n"
        f"Abstract: {abstract}"
    )


def _call_llm(system: str, user: str) -> str | None:
    """Call the LLM API (OpenAI-compatible). Returns the response text."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed — run: pip install openai")
        return None

    client_kwargs: dict = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        client_kwargs["base_url"] = LLM_BASE_URL

    try:
        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error("LLM API call failed: %s", e)
        return None


def _parse_llm_response(text: str) -> list[dict]:
    """Extract the JSON array from the LLM response (tolerant of markdown fences)."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the text
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse LLM response as JSON array")
    return []


def llm_evaluate(papers: list[ArxivPaper]) -> list[ArxivPaper]:
    """
    Use an LLM to score each paper's relevance.

    Papers are sent in batches. Results are merged back into each paper's
    ``relevance_score``, ``matched_interests``, and ``relevance_reason``.

    Returns only papers with LLM score ≥ RELEVANCE_THRESHOLD.
    """
    if not papers:
        return []

    system = _SYSTEM_PROMPT.format(interests=_format_interests())
    relevant: list[ArxivPaper] = []
    paper_map = {p.arxiv_id: p for p in papers}
    batch_size = LLM_BATCH_SIZE

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        paper_blocks = "\n\n".join(_format_paper_block(p) for p in batch)
        user = _USER_PROMPT.format(n=len(batch), papers=paper_blocks)

        logger.info(
            "LLM evaluation batch %d–%d of %d...",
            i + 1, min(i + batch_size, len(papers)), len(papers),
        )

        response_text = _call_llm(system, user)
        if not response_text:
            logger.warning("LLM returned no response for batch %d; skipping", i)
            continue

        results = _parse_llm_response(response_text)
        for item in results:
            aid = str(item.get("id", "")).strip()
            score = int(item.get("score", 0))
            interests = item.get("interests", [])
            reason = str(item.get("reason", ""))

            paper = paper_map.get(aid)
            if not paper:
                # Try fuzzy match (LLM might return "2603.03380v1" instead of "2603.03380")
                clean = re.sub(r"v\d+$", "", aid)
                paper = paper_map.get(clean)
            if not paper:
                continue

            paper.relevance_score = float(score)
            paper.matched_interests = interests if isinstance(interests, list) else []
            paper.relevance_reason = reason

            if score >= RELEVANCE_THRESHOLD:
                relevant.append(paper)

        # Rate limit between batches
        if i + batch_size < len(papers):
            time.sleep(1)

    logger.info(
        "LLM evaluation: %d / %d papers are relevant (threshold=%d)",
        len(relevant), len(papers), RELEVANCE_THRESHOLD,
    )
    return relevant


# ---------------------------------------------------------------------------
# Keyword-only fallback (no LLM)
# ---------------------------------------------------------------------------


def _keyword_evaluate(papers: list[ArxivPaper]) -> list[ArxivPaper]:
    """
    Fallback: use keyword scoring when no LLM is available.

    Applies a higher threshold than the pre-filter stage and generates
    a keyword-based relevance reason.
    """
    KEYWORD_FINAL_THRESHOLD = 4.0
    relevant = []

    for paper in papers:
        title_norm = _normalize(paper.title)
        abstract_norm = _normalize(paper.abstract)
        total_score = 0.0
        matched_interests: list[str] = []
        reason_parts: list[str] = []

        for profile in INTEREST_PROFILES:
            profile_score = 0.0
            hits: list[str] = []
            for keyword, weight in profile["keywords"]:
                kw = keyword.lower()
                tc = _count_matches(title_norm, kw)
                ac = _count_matches(abstract_norm, kw)
                if tc or ac:
                    profile_score += weight * (tc * TITLE_MULTIPLIER + ac)
                    loc = []
                    if tc:
                        loc.append("title")
                    if ac:
                        loc.append("abstract")
                    hits.append(f'"{keyword}" ({"+".join(loc)})')
            if hits:
                matched_interests.append(profile["name"])
                shown = ", ".join(hits[:5])
                extra = f" (+{len(hits)-5} more)" if len(hits) > 5 else ""
                reason_parts.append(f'**{profile["name"]}**: matched {shown}{extra}')
            total_score += profile_score

        for cat in paper.categories:
            total_score += CATEGORY_BONUS.get(cat, 0.0)

        paper.relevance_score = total_score
        paper.matched_interests = matched_interests
        paper.relevance_reason = "; ".join(reason_parts)

        if total_score >= KEYWORD_FINAL_THRESHOLD:
            relevant.append(paper)

    relevant.sort(key=lambda p: p.relevance_score, reverse=True)
    logger.info(
        "Keyword fallback: %d / %d papers matched (threshold=%.1f)",
        len(relevant), len(papers), KEYWORD_FINAL_THRESHOLD,
    )
    return relevant


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_relevant(papers: list[ArxivPaper]) -> list[ArxivPaper]:
    """
    Main entry point: filter papers by relevance.

    1. Keyword pre-filter (cheap)
    2. LLM evaluation (if configured) or keyword fallback
    3. Cap at MAX_RESULTS
    """
    # Stage 1: pre-filter
    candidates = prefilter(papers)

    # Stage 2: LLM or fallback
    if LLM_API_KEY:
        logger.info("Using LLM (%s) for relevance evaluation...", LLM_MODEL)
        relevant = llm_evaluate(candidates)
    else:
        logger.warning(
            "LLM_API_KEY not set — falling back to keyword-only scoring. "
            "Set LLM_API_KEY, LLM_MODEL (and optionally LLM_BASE_URL) for LLM evaluation."
        )
        relevant = _keyword_evaluate(candidates)

    # Sort and cap
    relevant.sort(key=lambda p: p.relevance_score, reverse=True)
    if len(relevant) > MAX_RESULTS:
        logger.warning("Capping from %d to %d", len(relevant), MAX_RESULTS)
        relevant = relevant[:MAX_RESULTS]

    logger.info("Final: %d relevant papers out of %d total", len(relevant), len(papers))
    return relevant
