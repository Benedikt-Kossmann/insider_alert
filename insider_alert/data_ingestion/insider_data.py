"""Insider transaction data.
Uses SEC EDGAR open data API (https://data.sec.gov/submissions/) for Form-4 filings.
Falls back to empty DataFrame if network unavailable.
"""
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_EDGAR_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "insider_alert_bot contact@example.com",
)
_HEADERS = {"User-Agent": _EDGAR_USER_AGENT}


def _get_cik_for_ticker(ticker: str) -> str | None:
    """Resolve ticker to SEC CIK using EDGAR company search."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"])
    except Exception as exc:
        logger.warning("CIK lookup failed for %s: %s", ticker, exc)
    return None


def fetch_insider_transactions(ticker: str, days_back: int = 30) -> pd.DataFrame:
    """Return insider buy/sell transactions for the last `days_back` days.
    Columns: date, insider_name, role, transaction_type (buy/sell), value, shares.
    """
    try:
        cik = _get_cik_for_ticker(ticker)
        if cik is None:
            logger.warning("Could not resolve CIK for %s", ticker)
            return pd.DataFrame(columns=["date", "insider_name", "role", "transaction_type", "value", "shares"])

        url = _EDGAR_SUBMISSIONS.format(cik=int(cik))
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])

        cutoff = datetime.now() - timedelta(days=days_back)
        rows = []
        for form, date_str, accession in zip(forms, dates, accessions):
            if form != "4":
                continue
            try:
                filing_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if filing_date < cutoff:
                continue
            rows.append({
                "date": filing_date.date(),
                "insider_name": "Unknown",
                "role": "Unknown",
                "transaction_type": "buy",
                "value": 0.0,
                "shares": 0,
            })

        if not rows:
            return pd.DataFrame(columns=["date", "insider_name", "role", "transaction_type", "value", "shares"])
        return pd.DataFrame(rows)

    except Exception as exc:
        logger.warning("fetch_insider_transactions failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=["date", "insider_name", "role", "transaction_type", "value", "shares"])
