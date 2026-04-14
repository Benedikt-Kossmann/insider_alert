"""Sector rotation signal — detects capital flow between sectors."""
import logging

import numpy as np
import pandas as pd

from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

logger = logging.getLogger(__name__)

_SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLY",
    "XLP", "XLI", "XLB", "XLRE", "XLU", "XLC",
]

_COMPONENTS = [
    SignalComponent(
        key="sector_momentum_rank",
        max_score=40,
        normaliser=1.0,       # already in [0, 1]
        flag_template="📊 Sector Momentum Rank: {value:.0%} (1.0 = top sector)",
        flag_threshold=0.8,
    ),
    SignalComponent(
        key="sector_reversal_score",
        max_score=30,
        normaliser=1.0,
        flag_template="🔄 Sector Reversal (10d oversold → 3d recovery): {value:.2f}",
        flag_threshold=0.5,
    ),
    SignalComponent(
        key="capital_flow_score",
        max_score=30,
        normaliser=1.0,
        flag_template="💰 Capital Flow (volume × direction): {value:.2f}",
        flag_threshold=0.5,
        abs_value=True,
    ),
]


def compute_sector_rotation_features(ticker_sector_etf: str = "XLK") -> dict:
    """Compute sector rotation features for a given sector ETF.

    Parameters
    ----------
    ticker_sector_etf : str
        SPDR sector ETF of the analysed ticker (e.g. "XLK" for tech stocks).

    Returns
    -------
    dict
        sector_momentum_rank  : float in [0, 1], 1 = best performing sector
        sector_reversal_score : float in [0, 1], oversold 10d but recovering 3d
        capital_flow_score    : float in [-1, 1], positive = inflow
    """
    defaults = {
        "sector_momentum_rank": 0.5,
        "sector_reversal_score": 0.0,
        "capital_flow_score": 0.0,
    }

    try:
        import yfinance as yf
        data = yf.download(_SECTOR_ETFS, period="30d", progress=False, auto_adjust=True)
        if data.empty:
            return defaults
    except Exception as exc:
        logger.warning("Sector data fetch failed: %s", exc)
        return defaults

    # yfinance returns multi-level columns when multiple tickers are passed
    close: pd.DataFrame = data["Close"] if "Close" in data.columns else data
    volume: pd.DataFrame | None = data["Volume"] if "Volume" in data.columns else None

    ret_10d = close.pct_change(10).iloc[-1]
    ret_3d = close.pct_change(3).iloc[-1]

    # 1. Sector momentum rank (percentile rank against all 11 sectors)
    ranks = ret_10d.rank(pct=True)
    sector_rank = float(ranks.get(ticker_sector_etf, 0.5)) if ticker_sector_etf in ranks.index else 0.5

    # 2. Sector reversal: weak 10d return but recovering over past 3d
    if ticker_sector_etf in ret_10d.index and ticker_sector_etf in ret_3d.index:
        r10 = float(ret_10d[ticker_sector_etf])
        r3 = float(ret_3d[ticker_sector_etf])
        # High reversal = prior weakness × current strength
        reversal = max(0.0, -r10) * max(0.0, r3) * 100
        sector_reversal = float(np.clip(reversal, 0.0, 1.0))
    else:
        sector_reversal = 0.0

    # 3. Capital flow proxy: above-average volume + positive direction
    capital_flow = 0.0
    if volume is not None and ticker_sector_etf in volume.columns:
        vol_series = volume[ticker_sector_etf].tail(20).dropna()
        if len(vol_series) >= 2:
            vol_mean = float(vol_series.mean())
            vol_latest = float(vol_series.iloc[-1])
            vol_ratio = vol_latest / max(vol_mean, 1.0)
            if ticker_sector_etf in ret_3d.index:
                direction = 1.0 if float(ret_3d[ticker_sector_etf]) > 0 else -1.0
                capital_flow = float(np.clip(direction * (vol_ratio - 1.0), -1.0, 1.0))

    return {
        "sector_momentum_rank": round(sector_rank, 3),
        "sector_reversal_score": round(sector_reversal, 3),
        "capital_flow_score": round(capital_flow, 3),
    }


def compute_sector_rotation_signal(features: dict) -> dict:
    """Compute sector rotation signal from sector features.

    Returns ``{"signal_type": "sector_rotation", "score": 0-100, "flags": [...]}``.
    """
    return compute_signal("sector_rotation", features, _COMPONENTS)
