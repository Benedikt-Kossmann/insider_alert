"""News and sentiment feature computation."""
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False
    logger.warning("textblob not available; sentiment will be 0")


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
    if news_count_24h > 0 and _TEXTBLOB_AVAILABLE:
        try:
            polarities = []
            for title in news_24h:
                if title:
                    blob = TextBlob(str(title))
                    polarities.append(blob.sentiment.polarity)
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
