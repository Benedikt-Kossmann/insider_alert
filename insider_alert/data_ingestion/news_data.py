"""News/sentiment data using yfinance news endpoint."""
import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_news(ticker: str, max_items: int = 20) -> pd.DataFrame:
    """Return recent news items with columns: title, published_at, source."""
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news:
            return pd.DataFrame(columns=["title", "published_at", "source"])
        rows = []
        for item in news[:max_items]:
            title = item.get("title", "")
            pub_ts = item.get("providerPublishTime", 0)
            if pub_ts:
                published_at = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
            else:
                published_at = None
            source = item.get("publisher", "")
            rows.append({"title": title, "published_at": published_at, "source": source})
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("fetch_news failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=["title", "published_at", "source"])
