"""News and sentiment feature computation."""
import logging
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Financial-domain sentiment lexicon (replaces TextBlob)
# ---------------------------------------------------------------------------
# Scores: +2 = strongly bullish, +1 = mildly bullish,
#         -1 = mildly bearish, -2 = strongly bearish
_FINANCIAL_LEXICON: dict[str, int] = {
    # Strongly bullish
    "soar": 2, "soars": 2, "soaring": 2, "surge": 2, "surges": 2, "surging": 2,
    "skyrocket": 2, "skyrockets": 2, "breakout": 2, "record high": 2,
    "blowout": 2, "crushes": 2, "smashes": 2, "blockbuster": 2,
    # Mildly bullish
    "beat": 1, "beats": 1, "beating": 1, "exceeded": 1, "exceeds": 1,
    "rally": 1, "rallies": 1, "upgrade": 1, "upgrades": 1, "upgraded": 1,
    "raises": 1, "raised": 1, "outperform": 1, "bullish": 1, "growth": 1,
    "strong": 1, "strength": 1, "boom": 1, "profit": 1, "profitable": 1,
    "dividend": 1, "buyback": 1, "acquisition": 1, "partnership": 1,
    "approval": 1, "approved": 1, "expand": 1, "expansion": 1,
    "accelerate": 1, "momentum": 1, "upbeat": 1, "positive": 1,
    "optimistic": 1, "jumps": 1, "gains": 1, "recover": 1, "recovery": 1,
    "outpace": 1, "rebound": 1, "rebounds": 1,
    # Mildly bearish
    "miss": -1, "misses": -1, "missed": -1, "decline": -1, "declines": -1,
    "drop": -1, "drops": -1, "fall": -1, "falls": -1, "weak": -1,
    "weakness": -1, "downgrade": -1, "downgrades": -1, "downgraded": -1,
    "cut": -1, "cuts": -1, "layoff": -1, "layoffs": -1, "slowdown": -1,
    "concern": -1, "concerns": -1, "risk": -1, "warning": -1, "warns": -1,
    "delay": -1, "delays": -1, "delayed": -1, "disappointing": -1,
    "below": -1, "underperform": -1, "slump": -1, "slumps": -1,
    "bearish": -1, "sell": -1, "selloff": -1, "losses": -1, "loss": -1,
    # Strongly bearish
    "crash": -2, "crashes": -2, "plunge": -2, "plunges": -2, "plunging": -2,
    "collapse": -2, "collapses": -2, "bankruptcy": -2, "bankrupt": -2,
    "default": -2, "defaults": -2, "fraud": -2, "investigation": -2,
    "lawsuit": -2, "indictment": -2, "halt": -2, "halted": -2,
    "delisted": -2, "recession": -2, "crisis": -2,
}

# Negation words that flip the next sentiment word
_NEGATION = {"not", "no", "never", "neither", "nor", "without", "barely", "hardly"}

_WORD_RE = re.compile(r"[a-z]+(?:\s+[a-z]+)?")


def _financial_sentiment(text: str) -> float:
    """Compute sentiment for one headline using the financial lexicon.

    Returns a value in [-1.0, 1.0].
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    words = text_lower.split()
    total = 0.0
    count = 0
    negated = False

    for word in words:
        if word in _NEGATION:
            negated = True
            continue

        score = _FINANCIAL_LEXICON.get(word, 0)
        if score != 0:
            if negated:
                score = -score
            total += score
            count += 1
            negated = False
        else:
            negated = False

    # Also check two-word phrases (e.g. "record high")
    for phrase, score in _FINANCIAL_LEXICON.items():
        if " " in phrase and phrase in text_lower:
            total += score
            count += 1

    if count == 0:
        return 0.0
    # Normalise to [-1, 1]; max realistic score ~4-6
    return float(np.clip(total / max(count * 2, 1), -1.0, 1.0))


def compute_news_features(news_df: pd.DataFrame, return_1d: float) -> dict:
    """Compute news and sentiment features."""
    defaults = {
        "news_count_24h": 0,
        "news_sentiment_score": 0.0,
        "public_catalyst_strength": 0.0,
        "price_news_divergence_score": 0.0,
    }
    if news_df is None or news_df.empty:
        divergence = float(np.clip(abs(return_1d), 0.0, 1.0)) if abs(return_1d) > 0.02 else 0.0
        defaults["price_news_divergence_score"] = divergence
        return defaults

    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(hours=24)

    news_24h = []
    if "published_at" in news_df.columns:
        for _, row in news_df.iterrows():
            pub = row.get("published_at")
            if pub is None:
                continue
            if isinstance(pub, str):
                try:
                    pub = datetime.fromisoformat(pub)
                except ValueError:
                    continue
            if isinstance(pub, datetime):
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                if pub >= cutoff:
                    news_24h.append(row.get("title", ""))
    else:
        news_24h = news_df.get("title", pd.Series()).tolist()

    news_count_24h = len(news_24h)

    news_sentiment_score = 0.0
    if news_count_24h > 0:
        try:
            polarities = [_financial_sentiment(t) for t in news_24h if t]
            if polarities:
                news_sentiment_score = float(np.mean(polarities))
        except Exception as exc:
            logger.warning("Sentiment computation failed: %s", exc)

    public_catalyst_strength = float(np.clip(news_count_24h / 5.0, 0.0, 1.0))

    if news_count_24h == 0 and abs(return_1d) > 0.02:
        price_news_divergence_score = float(np.clip(abs(return_1d), 0.0, 1.0))
    else:
        price_news_divergence_score = 0.0

    return {
        "news_count_24h": news_count_24h,
        "news_sentiment_score": news_sentiment_score,
        "public_catalyst_strength": public_catalyst_strength,
        "price_news_divergence_score": price_news_divergence_score,
    }
