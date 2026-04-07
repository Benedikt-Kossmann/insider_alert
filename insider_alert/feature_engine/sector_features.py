"""Sector-relative strength computation.

Compares a ticker's recent performance to its sector ETF to determine
whether a move is stock-specific or market-driven.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping of popular tickers to sector ETFs
_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "QCOM": "XLK",
    "AMD": "XLK", "CRM": "XLK", "ORCL": "XLK", "PLTR": "XLK", "SNOW": "XLK",
    "ARM": "XLK",
    # Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY", "SHOP": "XLY", "UBER": "XLY", "NFLX": "XLY",
    "RIVN": "XLY", "LCID": "XLY",
    # Communication
    "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF", "WFC": "XLF",
    "C": "XLF", "V": "XLF", "MA": "XLF", "SOFI": "XLF", "COIN": "XLF",
    "HOOD": "XLF",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "OXY": "XLE", "SLB": "XLE",
    # Healthcare
    "UNH": "XLV", "LLY": "XLV", "JNJ": "XLV", "ABBV": "XLV", "MRK": "XLV",
    "ISRG": "XLV", "MRNA": "XLV",
    # Industrials
    "CAT": "XLI", "GE": "XLI",
    # Crypto-adjacent (use Bitcoin proxy)
    "MSTR": "BITO", "MARA": "BITO", "RIOT": "BITO",
}

# ETF → sector label
_SECTOR_LABELS: dict[str, str] = {
    "XLK": "Technology", "XLY": "Consumer Discretionary", "XLC": "Communication",
    "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLI": "Industrials", "BITO": "Crypto", "SPY": "S&P 500",
}


def get_sector_etf(ticker: str) -> str:
    """Return the sector ETF for a ticker, falling back to SPY."""
    return _SECTOR_MAP.get(ticker.upper(), "SPY")


def get_sector_label(ticker: str) -> str:
    """Return a human-readable sector label for a ticker."""
    etf = get_sector_etf(ticker)
    return _SECTOR_LABELS.get(etf, "Market")


def compute_relative_strength(
    ticker_ohlcv: pd.DataFrame,
    sector_ohlcv: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """Compute relative strength of a ticker vs its sector ETF.

    Returns
    -------
    dict
        ``rs_5d``, ``rs_10d``, ``rs_20d`` – relative return delta in pct.
        ``rs_score`` (0-100) – composite relative-strength score.
        ``relative_trend`` – "outperforming" / "inline" / "underperforming".
        ``sector_etf``, ``sector_label``.
    """
    defaults = {
        "rs_5d": 0.0,
        "rs_10d": 0.0,
        "rs_20d": 0.0,
        "rs_score": 50.0,
        "relative_trend": "inline",
        "sector_etf": "",
        "sector_label": "",
    }

    if (ticker_ohlcv is None or ticker_ohlcv.empty
            or sector_ohlcv is None or sector_ohlcv.empty):
        return defaults

    t = ticker_ohlcv.copy()
    s = sector_ohlcv.copy()
    t.columns = [c.lower() for c in t.columns]
    s.columns = [c.lower() for c in s.columns]

    if "close" not in t.columns or "close" not in s.columns:
        return defaults

    tc = t["close"].dropna()
    sc = s["close"].dropna()

    if len(tc) < max(windows) or len(sc) < max(windows):
        return defaults

    rs_values = {}
    for w in windows:
        t_ret = (float(tc.iloc[-1]) / float(tc.iloc[-w]) - 1) * 100
        s_ret = (float(sc.iloc[-1]) / float(sc.iloc[-w]) - 1) * 100
        rs_values[f"rs_{w}d"] = round(t_ret - s_ret, 2)

    defaults.update(rs_values)

    # Composite score: 50 = inline, > 50 = outperforming
    # Weight recent performance more
    raw = (rs_values.get("rs_5d", 0) * 0.5
           + rs_values.get("rs_10d", 0) * 0.3
           + rs_values.get("rs_20d", 0) * 0.2)
    # Map ±10% relative to 0-100 scale
    score = float(np.clip(50 + raw * 5, 0, 100))
    defaults["rs_score"] = round(score, 1)

    if score >= 65:
        defaults["relative_trend"] = "outperforming"
    elif score <= 35:
        defaults["relative_trend"] = "underperforming"
    else:
        defaults["relative_trend"] = "inline"

    return defaults
