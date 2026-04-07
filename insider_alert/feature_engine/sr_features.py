"""Support and resistance level detection from OHLCV data."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_support_resistance(
    ohlcv: pd.DataFrame,
    *,
    pivot_window: int = 5,
    cluster_pct: float = 0.015,
    max_levels: int = 4,
) -> dict:
    """Detect support/resistance levels using pivot points and volume clusters.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        OHLCV data (at least 20 bars recommended).
    pivot_window : int
        Number of bars on each side to confirm a pivot high/low.
    cluster_pct : float
        Price tolerance (%) to merge nearby levels.
    max_levels : int
        Maximum number of S/R levels returned per side.

    Returns
    -------
    dict
        ``support_levels``, ``resistance_levels`` (list[float]),
        ``nearest_support``, ``nearest_resistance`` (float),
        ``distance_to_support_pct``, ``distance_to_resistance_pct`` (float),
        ``sr_zone`` (str): "near_support", "near_resistance", "mid_range"
    """
    defaults = {
        "support_levels": [],
        "resistance_levels": [],
        "nearest_support": 0.0,
        "nearest_resistance": 0.0,
        "distance_to_support_pct": 0.0,
        "distance_to_resistance_pct": 0.0,
        "sr_zone": "mid_range",
    }

    if ohlcv is None or ohlcv.empty or len(ohlcv) < pivot_window * 2 + 1:
        return defaults

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    for col in ("high", "low", "close", "volume"):
        if col not in df.columns:
            return defaults

    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values
    close = float(df["close"].iloc[-1])

    if close <= 0:
        return defaults

    # --- Find pivot highs and lows ---
    pivot_highs = []
    pivot_lows = []

    for i in range(pivot_window, len(df) - pivot_window):
        window_h = highs[i - pivot_window: i + pivot_window + 1]
        if highs[i] == window_h.max():
            pivot_highs.append((float(highs[i]), float(volumes[i])))

        window_l = lows[i - pivot_window: i + pivot_window + 1]
        if lows[i] == window_l.min():
            pivot_lows.append((float(lows[i]), float(volumes[i])))

    # --- Cluster nearby levels (volume-weighted) ---
    def cluster_levels(levels: list[tuple[float, float]]) -> list[float]:
        if not levels:
            return []
        # Sort by price
        levels.sort(key=lambda x: x[0])
        clusters = []
        current = [levels[0]]

        for price, vol in levels[1:]:
            cluster_mid = np.mean([p for p, _ in current])
            if abs(price - cluster_mid) / cluster_mid < cluster_pct:
                current.append((price, vol))
            else:
                clusters.append(current)
                current = [(price, vol)]
        clusters.append(current)

        # Volume-weighted average for each cluster
        result = []
        for cluster in clusters:
            total_vol = sum(v for _, v in cluster) + 1e-9
            weighted_price = sum(p * v for p, v in cluster) / total_vol
            result.append(round(weighted_price, 2))

        return result

    resistance_raw = cluster_levels(pivot_highs)
    support_raw = cluster_levels(pivot_lows)

    # Separate into above/below current price
    support_levels = sorted([s for s in support_raw if s < close], reverse=True)[:max_levels]
    resistance_levels = sorted([r for r in resistance_raw if r > close])[:max_levels]

    # Nearest levels
    nearest_support = support_levels[0] if support_levels else 0.0
    nearest_resistance = resistance_levels[0] if resistance_levels else 0.0

    dist_support = (close - nearest_support) / close * 100 if nearest_support > 0 else 0.0
    dist_resistance = (nearest_resistance - close) / close * 100 if nearest_resistance > 0 else 0.0

    # Zone classification
    sr_zone = "mid_range"
    if dist_support < 2.0 and nearest_support > 0:
        sr_zone = "near_support"
    elif dist_resistance < 2.0 and nearest_resistance > 0:
        sr_zone = "near_resistance"

    return {
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "distance_to_support_pct": round(dist_support, 2),
        "distance_to_resistance_pct": round(dist_resistance, 2),
        "sr_zone": sr_zone,
    }
