"""Shared SEC EDGAR utilities (CIK lookup, HTTP headers, rate limits)."""
import logging
import os

import requests

logger = logging.getLogger(__name__)

EDGAR_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "insider_alert_bot contact@example.com",
)
EDGAR_HEADERS = {"User-Agent": EDGAR_USER_AGENT}
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_REQUEST_DELAY = 0.12

_CIK_CACHE: dict[str, str | None] = {}


def get_cik_for_ticker(ticker: str) -> str | None:
    """Resolve ticker to SEC CIK using EDGAR company search (cached)."""
    ticker_upper = ticker.upper()
    if ticker_upper in _CIK_CACHE:
        return _CIK_CACHE[ticker_upper]

    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"])
                _CIK_CACHE[ticker_upper] = cik
                return cik
    except Exception as exc:
        logger.warning("CIK lookup failed for %s: %s", ticker, exc)

    _CIK_CACHE[ticker_upper] = None
    return None
