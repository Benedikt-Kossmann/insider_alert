"""Sentiment analysis for SEC filings (8-K, risk factors).

Splits filing text into FinBERT-digestible chunks and aggregates
sentiment scores per section.
"""
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

_SECTION_PATTERNS = {
    "material_event": re.compile(
        r"(?:Item\s+(?:2\.0[12]|5\.02|8\.01))(.+?)(?=Item\s+\d|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    "risk_factors": re.compile(
        r"(?:Risk\s+Factors)(.+?)(?=Item\s+\d|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
}


def _chunk_text(text: str, max_chars: int = 450) -> list[str]:
    """Split text into FinBERT-digestible chunks (~512 tokens ≈ 450 chars)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current += " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def analyze_filing(filing_text: str) -> dict:
    """Analyse SEC filing text with FinBERT.

    Parameters
    ----------
    filing_text : str
        Raw text of the filing (e.g. 8-K body).

    Returns
    -------
    dict
        ``{"filing_sentiment": float[-1,1], "risk_factor_change": float[-1,1]}``
    """
    if not filing_text:
        return {"filing_sentiment": 0.0, "risk_factor_change": 0.0}

    from insider_alert.nlp.finbert_sentiment import analyze_batch

    section_scores: dict[str, float] = {}
    for section_name, pattern in _SECTION_PATTERNS.items():
        match = pattern.search(filing_text)
        if not match:
            continue
        section_text = match.group(1)[:5000]   # cap to avoid extremely long sections
        chunks = _chunk_text(section_text)
        if chunks:
            try:
                results = analyze_batch(chunks)
                scores = [r["sentiment"] for r in results]
                section_scores[section_name] = float(np.mean(scores)) if scores else 0.0
            except Exception as exc:
                logger.warning("Filing sentiment failed for section %s: %s", section_name, exc)

    return {
        "filing_sentiment": section_scores.get("material_event", 0.0),
        "risk_factor_change": section_scores.get("risk_factors", 0.0),
    }
