"""Corporate event data (earnings dates etc.) using yfinance."""
import logging
from datetime import date, datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


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
