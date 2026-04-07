"""Corporate event data (earnings dates, 8-K filings) using yfinance and SEC EDGAR."""
import logging
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from insider_alert.data_ingestion.sec_utils import (
    get_cik_for_ticker as _get_cik_for_ticker,
    EDGAR_HEADERS as _EDGAR_HEADERS,
    EDGAR_SUBMISSIONS as _EDGAR_SUBMISSIONS,
)

logger = logging.getLogger(__name__)

# 8-K item codes that indicate material corporate events
_MATERIAL_8K_ITEMS = {
    "1.01": "Material Definitive Agreement",
    "1.02": "Termination of Material Definitive Agreement",
    "1.03": "Bankruptcy/Receivership",
    "2.01": "Completion of Acquisition or Disposition",
    "2.04": "Triggering Events for Acceleration",
    "5.01": "Changes in Control",
    "5.02": "Departure/Appointment of Officers or Directors",
    "5.07": "Shareholder Vote Results",
    "8.01": "Other Events",
}


def fetch_earnings_dates(ticker: str) -> pd.DataFrame:
    """Return upcoming and recent earnings dates."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty):
            return pd.DataFrame(columns=["earnings_date"])
        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date", [])
            if not earnings_date:
                return pd.DataFrame(columns=["earnings_date"])
            if not isinstance(earnings_date, list):
                earnings_date = [earnings_date]
            return pd.DataFrame({"earnings_date": earnings_date})
        if "Earnings Date" in cal.columns:
            return pd.DataFrame({"earnings_date": cal["Earnings Date"].dropna().tolist()})
        return pd.DataFrame(columns=["earnings_date"])
    except Exception as exc:
        logger.warning("fetch_earnings_dates failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=["earnings_date"])


def days_to_next_earnings(ticker: str) -> int | None:
    """Return number of calendar days to the next earnings date, or None."""
    try:
        df = fetch_earnings_dates(ticker)
        if df.empty:
            return None
        today = datetime.now().date()
        future_dates = []
        for val in df["earnings_date"]:
            try:
                if isinstance(val, (datetime,)):
                    d = val.date()
                elif isinstance(val, date):
                    d = val
                else:
                    d = pd.Timestamp(val).date()
                if d >= today:
                    future_dates.append(d)
            except Exception:
                continue
        if not future_dates:
            return None
        next_date = min(future_dates)
        return (next_date - today).days
    except Exception as exc:
        logger.warning("days_to_next_earnings failed for %s: %s", ticker, exc)
        return None


def fetch_recent_corporate_events(ticker: str, days_back: int = 30) -> pd.DataFrame:
    """Return recent material 8-K filings for the ticker from SEC EDGAR.

    Fetches the EDGAR submissions JSON and filters for 8-K and 8-K/A filings
    within the last ``days_back`` calendar days that contain material item codes.

    Returns a DataFrame with columns:
    - date (datetime.date): filing date
    - form_type (str): '8-K' or '8-K/A'
    - items (str): comma-separated item numbers from the filing
    - description (str): human-readable description of the primary item
    """
    _empty = pd.DataFrame(columns=["date", "form_type", "items", "description"])
    try:
        cik = _get_cik_for_ticker(ticker)
        if cik is None:
            logger.debug("Could not resolve CIK for %s; skipping 8-K fetch", ticker)
            return _empty

        url = _EDGAR_SUBMISSIONS.format(cik=int(cik))
        resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        items_list = recent.get("items", [""] * len(forms))

        cutoff = datetime.now() - timedelta(days=days_back)
        rows: list[dict] = []
        for form, date_str, items_str in zip(forms, dates, items_list):
            if form not in ("8-K", "8-K/A"):
                continue
            try:
                filing_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if filing_date < cutoff:
                continue

            items_str = str(items_str or "")
            # Build a human-readable description from the first recognised item
            description = ""
            for item_code, item_desc in _MATERIAL_8K_ITEMS.items():
                if item_code in items_str:
                    description = item_desc
                    break

            rows.append({
                "date": filing_date.date(),
                "form_type": form,
                "items": items_str,
                "description": description,
            })

        if not rows:
            return _empty
        return pd.DataFrame(rows)[["date", "form_type", "items", "description"]]

    except Exception as exc:
        logger.warning("fetch_recent_corporate_events failed for %s: %s", ticker, exc)
        return _empty
