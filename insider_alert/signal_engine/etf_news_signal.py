"""Direction-aware news sentiment signal for leveraged ETFs."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

_SENTIMENT_MAX = 50
_NEWS_COUNT_MAX = 30
_DIVERGENCE_MAX = 20


def compute_etf_news_sentiment_signal(news_features: dict, direction: str = "long") -> dict:
    """Compute a news sentiment signal adapted for leveraged-ETF direction.

    For **long** ETFs a positive sentiment is favourable (high score).
    For **short** ETFs a negative sentiment is favourable (high score).

    Components
    ----------
    - ``news_sentiment_score`` → up to 50 pts (direction-aware)
    - ``news_count_24h``       → up to 30 pts (more news = higher conviction)
    - ``price_news_divergence_score`` → up to 20 pts
    """
    sentiment = float(news_features.get("news_sentiment_score", 0.0))
    news_count = int(news_features.get("news_count_24h", 0))
    divergence = float(news_features.get("price_news_divergence_score", 0.0))

    flags: list[str] = []

    # --- Sentiment component (direction-aware) ---
    if direction == "short":
        effective_sentiment = -sentiment
    else:
        effective_sentiment = sentiment

    # Map [-1, +1] → [0, 1] so that +1 → 1.0, 0 → 0.5, -1 → 0.0
    sentiment_norm = (effective_sentiment + 1.0) / 2.0
    sentiment_score = float(np.clip(sentiment_norm * _SENTIMENT_MAX, 0, _SENTIMENT_MAX))

    if abs(sentiment) > 0.3:
        label = "bullish" if sentiment > 0 else "bearish"
        flags.append(f"News sentiment {label}: {sentiment:+.2f} (direction={direction})")

    # --- News count component ---
    count_norm = min(news_count / 5.0, 1.0)
    count_score = count_norm * _NEWS_COUNT_MAX

    if news_count >= 3:
        flags.append(f"Active news flow: {news_count} items in 24h")

    # --- Divergence component ---
    divergence_score = float(np.clip(divergence, 0, 1)) * _DIVERGENCE_MAX

    if divergence > 0.3:
        flags.append(f"Price-news divergence: {divergence:.2f}")

    total = float(np.clip(sentiment_score + count_score + divergence_score, 0, 100))

    return {
        "signal_type": "news_sentiment",
        "score": total,
        "flags": flags,
    }
