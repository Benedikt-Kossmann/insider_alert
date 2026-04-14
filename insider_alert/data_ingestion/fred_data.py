"""FRED API data ingestion for macro indicators."""
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# FRED Series IDs
_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi_yoy": "CPIAUCSL",
    "unemployment": "UNRATE",
    "yield_10y2y": "T10Y2Y",
    "hy_spread": "BAMLH0A0HYM2",
    "initial_claims": "ICSA",
    "real_gdp": "GDPC1",
    "consumer_sentiment": "UMCSENT",
}

_fred_client = None


def _get_client():
    """Lazy-load FRED client. Returns None when unavailable."""
    global _fred_client
    if _fred_client is not None:
        return _fred_client
    try:
        from fredapi import Fred
        from insider_alert.config import get_config

        cfg = get_config()
        # Support both Config attribute and env variable
        api_key = getattr(cfg, "fred_api_key", "") or os.environ.get("FRED_API_KEY", "")
        if not api_key:
            logger.warning(
                "FRED API key not configured. Set fred_api_key in config.yaml "
                "or FRED_API_KEY in .env. Macro signal will use defaults."
            )
            return None
        _fred_client = Fred(api_key=api_key)
        logger.info("FRED API client initialized")
    except ImportError:
        logger.warning("fredapi not installed; run: pip install 'fredapi>=0.5'")
        return None
    except Exception as exc:
        logger.warning("FRED client init failed: %s", exc)
        return None
    return _fred_client


def fetch_fred_series(series_id: str, lookback_days: int = 365) -> Optional[pd.Series]:
    """Fetch a single FRED time series.

    Returns a pd.Series with DatetimeIndex, or None on failure.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        start = datetime.now() - timedelta(days=lookback_days)
        data = client.get_series(series_id, observation_start=start)
        if data is not None and not data.empty:
            return data.dropna()
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
    return None


def fetch_all_macro_data() -> dict:
    """Fetch all relevant macro data from FRED.

    Returns a dict with pre-processed feature values.  Falls back to
    sensible defaults when the API is unavailable or a series cannot
    be fetched.

    Keys
    ----
    hy_spread : float
        High-Yield spread in % (ICE BofA).
    hy_spread_change_1m : float
        1-month change in HY spread.
    fed_funds_rate : float
        Current effective federal funds rate.
    fed_policy_direction : str
        "tightening" | "easing" | "hold"
    cpi_yoy_change : float
        CPI year-over-year change in %.
    inflation_trend : str
        "rising" | "falling" | "stable"
    initial_claims_zscore : float
        Weekly initial jobless claims Z-score vs trailing 52-week mean.
    unemployment_rate : float
        Unemployment rate in %.
    consumer_sentiment : float
        UMich Consumer Sentiment index.
    """
    defaults: dict = {
        "hy_spread": 4.0,
        "hy_spread_change_1m": 0.0,
        "fed_funds_rate": 5.0,
        "fed_policy_direction": "hold",
        "cpi_yoy_change": 0.0,
        "inflation_trend": "stable",
        "initial_claims_zscore": 0.0,
        "unemployment_rate": 4.0,
        "consumer_sentiment": 70.0,
    }

    result = defaults.copy()

    # 1. High Yield Spread (credit stress)
    hy = fetch_fred_series("BAMLH0A0HYM2", lookback_days=60)
    if hy is not None and len(hy) > 20:
        result["hy_spread"] = float(hy.iloc[-1])
        result["hy_spread_change_1m"] = float(hy.iloc[-1] - hy.iloc[-20])

    # 2. Fed Funds Rate
    ffr = fetch_fred_series("FEDFUNDS", lookback_days=180)
    if ffr is not None and len(ffr) >= 3:
        result["fed_funds_rate"] = float(ffr.iloc[-1])
        recent = ffr.tail(3).values
        if recent[-1] > recent[0] + 0.1:
            result["fed_policy_direction"] = "tightening"
        elif recent[-1] < recent[0] - 0.1:
            result["fed_policy_direction"] = "easing"
        else:
            result["fed_policy_direction"] = "hold"

    # 3. CPI / Inflation
    cpi = fetch_fred_series("CPIAUCSL", lookback_days=400)
    if cpi is not None and len(cpi) >= 13:
        yoy = float((cpi.iloc[-1] / cpi.iloc[-12] - 1) * 100) if cpi.iloc[-12] > 0 else 0.0
        result["cpi_yoy_change"] = round(yoy, 2)
        yoy_prev = float((cpi.iloc[-2] / cpi.iloc[-13] - 1) * 100) if len(cpi) >= 14 else yoy
        if yoy > yoy_prev + 0.2:
            result["inflation_trend"] = "rising"
        elif yoy < yoy_prev - 0.2:
            result["inflation_trend"] = "falling"
        else:
            result["inflation_trend"] = "stable"

    # 4. Initial Claims (weekly)
    claims = fetch_fred_series("ICSA", lookback_days=365)
    if claims is not None and len(claims) > 26:
        import numpy as np
        mean_1y = float(claims.mean())
        std_1y = float(claims.std())
        if std_1y > 0:
            result["initial_claims_zscore"] = round(float((claims.iloc[-1] - mean_1y) / std_1y), 2)

    # 5. Unemployment Rate
    unemp = fetch_fred_series("UNRATE", lookback_days=90)
    if unemp is not None and len(unemp) > 0:
        result["unemployment_rate"] = float(unemp.iloc[-1])

    # 6. Consumer Sentiment (UMich)
    sent = fetch_fred_series("UMCSENT", lookback_days=90)
    if sent is not None and len(sent) > 0:
        result["consumer_sentiment"] = float(sent.iloc[-1])

    return result
