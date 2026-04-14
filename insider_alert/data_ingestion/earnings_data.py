"""Earnings data from yfinance for PEAD (Post-Earnings Announcement Drift) analysis."""
import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_earnings_data(ticker: str) -> dict:
    """Fetch earnings-related data for PEAD analysis.

    Returns
    -------
    dict
        last_earnings_date : str | None — ISO date string of most recent earnings
        days_since_earnings : int — days elapsed since last earnings
        earnings_day_return : float — price change on earnings day (%)
        post_earnings_drift_3d : float — 3-day return after earnings (%)
        post_earnings_drift_10d : float — 10-day return after earnings (%)
        next_earnings_date : str | None — ISO date of next expected earnings
    """
    defaults: dict = {
        "last_earnings_date": None,
        "days_since_earnings": 999,
        "earnings_day_return": 0.0,
        "post_earnings_drift_3d": 0.0,
        "post_earnings_drift_10d": 0.0,
        "next_earnings_date": None,
    }

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)

        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is None or earnings_dates.empty:
                return defaults
        except Exception:
            return defaults

        now = datetime.now()
        past = earnings_dates[earnings_dates.index <= pd.Timestamp(now)]
        future = earnings_dates[earnings_dates.index > pd.Timestamp(now)]

        if past.empty:
            return defaults

        last_date = past.index[0]  # Most recent past earnings
        last_dt = last_date.to_pydatetime().replace(tzinfo=None)
        days_since = max(0, (now - last_dt).days)

        result = defaults.copy()
        result["last_earnings_date"] = str(last_date.date())
        result["days_since_earnings"] = days_since

        if not future.empty:
            result["next_earnings_date"] = str(future.index[-1].date())

        # Price action around earnings
        try:
            start_str = (last_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            end_str = (last_date + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
            hist = stock.history(start=start_str, end=end_str)

            if hist.empty or len(hist) < 3:
                return result

            # Find index closest to earnings date
            earn_idx = hist.index.get_indexer([last_date], method="nearest")[0]

            if 0 < earn_idx < len(hist):
                prev_close = float(hist["Close"].iloc[earn_idx - 1])
                earn_close = float(hist["Close"].iloc[earn_idx])
                if prev_close > 0:
                    result["earnings_day_return"] = round(
                        (earn_close / prev_close - 1) * 100, 2
                    )

            if earn_idx + 3 < len(hist):
                base = float(hist["Close"].iloc[earn_idx])
                if base > 0:
                    result["post_earnings_drift_3d"] = round(
                        (float(hist["Close"].iloc[earn_idx + 3]) / base - 1) * 100, 2
                    )

            if earn_idx + 10 < len(hist):
                base = float(hist["Close"].iloc[earn_idx])
                if base > 0:
                    result["post_earnings_drift_10d"] = round(
                        (float(hist["Close"].iloc[earn_idx + 10]) / base - 1) * 100, 2
                    )

        except Exception as exc:
            logger.debug("Earnings price data failed for %s: %s", ticker, exc)

        return result

    except Exception as exc:
        logger.warning("Earnings data fetch failed for %s: %s", ticker, exc)
        return defaults
