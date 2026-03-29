"""News divergence signal computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_news_divergence_signal(features: dict) -> dict:
    """Compute news divergence signal from news features."""
    flags = []

    divergence = features.get("price_news_divergence_score", 0.0)
    sentiment = features.get("news_sentiment_score", 0.0)
    news_count = features.get("news_count_24h", 0)

    divergence_component = min(divergence / 0.05, 1.0) * 60
    if news_count > 0:
        sentiment_conflict = (1 - abs(sentiment)) * 40
    else:
        sentiment_conflict = 0.0

    if divergence_component > 30:
        flags.append(f"Price moved without news catalyst: divergence={divergence:.3f}")
    if sentiment_conflict > 20 and news_count > 0:
        flags.append(f"Weak/neutral news sentiment during price move: {sentiment:.2f}")

    score = float(np.clip(divergence_component + sentiment_conflict, 0.0, 100.0))

    return {
        "signal_type": "news_divergence",
        "score": score,
        "flags": flags,
    }
