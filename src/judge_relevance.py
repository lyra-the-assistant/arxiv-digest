#!/usr/bin/env python3
"""Judge paper relevance using LLM."""

import os
import json
import logging
from anthropic import Anthropic

logger = logging.getLogger(__name__)

THEMES = """
Theme A: NL → Atomic Capability Planning / Execution
Theme B: Edge-Efficient Robot Learning Inference
"""

SYSTEM_PROMPT = f"""You are a research paper relevance judge. Evaluate if papers match these themes:

{THEMES}

Respond with JSON only:
{{"is_relevant": true/false, "theme": "A/B/Both/None", "reason": "brief explanation"}}"""


def judge_papers_sequential(papers: list) -> list:
    """Judge papers sequentially using Claude."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    client = Anthropic(api_key=api_key)
    results = []
    
    for i, paper in enumerate(papers, 1):
        logger.info(f"Judging {i}/{len(papers)}: {paper['arxiv_id']}")
        
        prompt = f"""Title: {paper['title']}
Abstract: {paper['abstract']}

Is this relevant?"""
        
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text.strip()
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            
            result = json.loads(text)
            result["arxiv_id"] = paper["arxiv_id"]
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to judge {paper['arxiv_id']}: {e}")
            results.append({
                "arxiv_id": paper["arxiv_id"],
                "is_relevant": False,
                "theme": "None",
                "reason": f"Error: {str(e)}"
            })
    
    return results
