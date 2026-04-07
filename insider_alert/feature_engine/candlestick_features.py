"""Candlestick pattern recognition from OHLCV data."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_candlestick_patterns(ohlcv: pd.DataFrame) -> dict:
    """Detect classic candlestick patterns from the last few bars.

    Returns a dict with boolean flags and a composite pattern score (0-100).
    """
    defaults = {
        "hammer": False,
        "inverted_hammer": False,
        "engulfing_bullish": False,
        "engulfing_bearish": False,
        "doji": False,
        "morning_star": False,
        "evening_star": False,
        "three_white_soldiers": False,
        "three_black_crows": False,
        "bullish_pattern_score": 0.0,
        "bearish_pattern_score": 0.0,
        "pattern_names": [],
    }
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 3:
        return defaults

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            return defaults

    c = df.iloc[-1]
    p = df.iloc[-2]
    pp = df.iloc[-3]

    o, h, lo, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    po, ph, plo, pcl = float(p["open"]), float(p["high"]), float(p["low"]), float(p["close"])
    ppo, pph, pplo, ppcl = float(pp["open"]), float(pp["high"]), float(pp["low"]), float(pp["close"])

    body = cl - o
    body_abs = abs(body)
    hl_range = h - lo + 1e-9
    body_ratio = body_abs / hl_range

    upper_shadow = h - max(o, cl)
    lower_shadow = min(o, cl) - lo

    p_body = pcl - po
    p_body_abs = abs(p_body)
    p_hl = ph - plo + 1e-9

    bull_score = 0.0
    bear_score = 0.0
    patterns = []

    # --- Hammer (bullish reversal at bottom) ---
    if lower_shadow > 2 * body_abs and upper_shadow < body_abs * 0.5 and body_ratio < 0.4:
        defaults["hammer"] = True
        patterns.append("Hammer 🔨")
        bull_score += 20

    # --- Inverted Hammer ---
    if upper_shadow > 2 * body_abs and lower_shadow < body_abs * 0.5 and body_ratio < 0.4:
        defaults["inverted_hammer"] = True
        patterns.append("Inverted Hammer")
        bull_score += 15

    # --- Doji ---
    if body_ratio < 0.1:
        defaults["doji"] = True
        patterns.append("Doji ✚")
        # Doji is neutral, slight points to both
        bull_score += 5
        bear_score += 5

    # --- Bullish Engulfing ---
    if p_body < 0 and body > 0 and cl > po and o < pcl and body_abs > p_body_abs:
        defaults["engulfing_bullish"] = True
        patterns.append("Bullish Engulfing 🟢")
        bull_score += 25

    # --- Bearish Engulfing ---
    if p_body > 0 and body < 0 and cl < po and o > pcl and body_abs > p_body_abs:
        defaults["engulfing_bearish"] = True
        patterns.append("Bearish Engulfing 🔴")
        bear_score += 25

    # --- Morning Star (3-bar bullish reversal) ---
    pp_bearish = ppcl < ppo
    p_small = p_body_abs / p_hl < 0.3
    curr_bullish = cl > o
    if pp_bearish and p_small and curr_bullish and cl > (ppo + ppcl) / 2:
        defaults["morning_star"] = True
        patterns.append("Morning Star ⭐")
        bull_score += 30

    # --- Evening Star (3-bar bearish reversal) ---
    pp_bullish = ppcl > ppo
    if pp_bullish and p_small and body < 0 and cl < (ppo + ppcl) / 2:
        defaults["evening_star"] = True
        patterns.append("Evening Star 🌙")
        bear_score += 30

    # --- Three White Soldiers ---
    if len(ohlcv) >= 3:
        bars = [df.iloc[-3], df.iloc[-2], df.iloc[-1]]
        all_bullish = all(float(b["close"]) > float(b["open"]) for b in bars)
        progressive = (float(bars[1]["close"]) > float(bars[0]["close"])
                       and float(bars[2]["close"]) > float(bars[1]["close"]))
        if all_bullish and progressive:
            defaults["three_white_soldiers"] = True
            patterns.append("Three White Soldiers 🟢🟢🟢")
            bull_score += 25

    # --- Three Black Crows ---
    if len(ohlcv) >= 3:
        bars = [df.iloc[-3], df.iloc[-2], df.iloc[-1]]
        all_bearish = all(float(b["close"]) < float(b["open"]) for b in bars)
        progressive = (float(bars[1]["close"]) < float(bars[0]["close"])
                       and float(bars[2]["close"]) < float(bars[1]["close"]))
        if all_bearish and progressive:
            defaults["three_black_crows"] = True
            patterns.append("Three Black Crows 🔴🔴🔴")
            bear_score += 25

    defaults["bullish_pattern_score"] = float(np.clip(bull_score, 0, 100))
    defaults["bearish_pattern_score"] = float(np.clip(bear_score, 0, 100))
    defaults["pattern_names"] = patterns

    return defaults
