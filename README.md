# arxiv-digest

Daily arXiv paper digest for robotics research. Fetches new papers from `cs.RO`, filters them by research relevance via an LLM agent, generates a structured markdown report, and syncs matched papers to Zotero with PDFs attached.

## How it works

This project is designed as a **skill** invoked by an upstream LLM agent (e.g. OpenClaw, Cursor). The pipeline has three steps:

1. **Fetch** — pull today's arXiv RSS feed for `cs.RO`, enrich with metadata from the arXiv Search API, filter to `new` and `cross` announcements.
2. **Judge** — the invoking agent reads the fetched papers and judges each one for relevance against two research themes using LLM reasoning (no keyword heuristics).
3. **Process** — enrich relevant papers with venue info and project page URLs, write a markdown digest, and add them to Zotero (deduplicated, with arXiv PDFs).

See [`SKILL.md`](SKILL.md) for the full invocation protocol.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/web/simple
```

Place Zotero credentials in `.secret/zotero.env`:

```
ZOTERO_API_KEY=<your key>
ZOTERO_USER_ID=<your user id>
```

## Usage

```bash
source .venv/bin/activate

# Step 1: fetch papers
python src/main.py fetch

# Step 2: agent writes data/relevance.json (see SKILL.md)

# Step 3: process, generate digest, sync to Zotero
python src/main.py process --relevance data/relevance.json

# Dry run (skip Zotero sync)
python src/main.py process --relevance data/relevance.json --dry-run
```

## Project structure

```
├── SKILL.md                # Skill descriptor (agent invocation protocol)
├── requirements.txt        # Python dependencies
├── .secret/
│   └── zotero.env          # Zotero API credentials (not committed)
├── src/
│   ├── main.py             # CLI entry point (fetch / process)
│   ├── config.py           # Categories, paths, env loading
│   ├── arxiv_fetcher.py    # RSS feed + Search API enrichment
│   ├── venue_detector.py   # Conference / journal detection
│   ├── project_page_finder.py  # URL extraction from abstracts
│   ├── zotero_client.py    # Zotero API (dedup, create, PDF upload)
│   └── digest_writer.py    # Markdown report generation
├── data/                   # Runtime intermediates (papers.json, relevance.json)
└── digests/                # Daily markdown reports
```

## Research interests

The relevance filter targets two themes:

- **Theme A** — Algorithms that take atomic capabilities and natural-language commands, then schedule, plan, and execute actions (LLM planners, hierarchical skills, TAMP, etc.)
- **Theme B** — Fast inference of robot learning policies on edge platforms (quantisation, distillation, efficient VLA/VLM, latency-aware design, etc.)

## Attribution

Thank you to arXiv for use of its open access interoperability.
