"""Momentum signal – trend/momentum score for leveraged ETFs."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_momentum_signal(
    momentum_features: dict,
    *,
    direction: str = "long",
) -> dict:
    """Convert momentum features into a 0-100 score.

    Parameters
    ----------
    momentum_features : dict
        Output of ``compute_momentum_features()``.
    direction : str
        ``'long'`` or ``'short'``.  For short/inverse ETFs the RSI sweet
        spot is inverted.

    Returns
    -------
    dict
        ``{"signal_type": "momentum", "score": float, "flags": list[str]}``
    """
    score = 0.0
    flags: list[str] = []

    rsi = float(momentum_features.get("rsi_14", 50.0))
    macd_hist = float(momentum_features.get("macd_histogram", 0.0))
    ema_cross = int(momentum_features.get("ema_crossover", 0))
    adx = float(momentum_features.get("adx_14", 0.0))
    adx_trending = int(momentum_features.get("adx_trending", 0))

    # --- RSI sweet spot (up to 30 pts) ---
    if direction == "long":
        # Ideal RSI range 40-70 for long positions
        if 40 <= rsi <= 70:
            score += 30.0
            flags.append(f"RSI in bullish zone ({rsi:.0f})")
        elif 30 <= rsi < 40:
            score += 25.0  # approaching oversold bounce
            flags.append(f"RSI near oversold bounce ({rsi:.0f})")
        elif rsi < 30:
            score += 10.0  # deeply oversold, might not be momentum
            flags.append(f"RSI deeply oversold ({rsi:.0f})")
        elif 70 < rsi <= 80:
            score += 15.0
            flags.append(f"RSI overbought risk ({rsi:.0f})")
    else:
        # For short/inverse ETFs: bearish momentum = good
        if 30 <= rsi <= 60:
            score += 30.0
            flags.append(f"RSI in bearish zone ({rsi:.0f})")
        elif 60 < rsi <= 70:
            score += 25.0
            flags.append(f"RSI near overbought reversal ({rsi:.0f})")
        elif rsi > 70:
            score += 10.0
            flags.append(f"RSI extremely overbought ({rsi:.0f})")
        elif 20 <= rsi < 30:
            score += 15.0
            flags.append(f"RSI oversold, bearish exhaustion risk ({rsi:.0f})")

    # --- EMA crossover (up to 20 pts) ---
    if direction == "long" and ema_cross == 1:
        score += 20.0
        flags.append("EMA bullish crossover (fast > slow)")
    elif direction == "short" and ema_cross == -1:
        score += 20.0
        flags.append("EMA bearish crossover (fast < slow)")
    elif ema_cross == 0:
        score += 5.0

    # --- MACD histogram (up to 20 pts) ---
    if direction == "long":
        if macd_hist > 0:
            pts = min(20.0, 10.0 + abs(macd_hist) * 100)
            score += pts
            flags.append(f"MACD histogram positive ({macd_hist:.3f})")
    else:
        if macd_hist < 0:
            pts = min(20.0, 10.0 + abs(macd_hist) * 100)
            score += pts
            flags.append(f"MACD histogram negative ({macd_hist:.3f})")

    # --- ADX trend strength (up to 30 pts) ---
    if adx_trending:
        if adx >= 30:
            score += 30.0
            flags.append(f"Strong trend (ADX={adx:.0f})")
        else:
            score += 15.0
            flags.append(f"Trending (ADX={adx:.0f})")

    score = float(np.clip(score, 0.0, 100.0))

    return {
        "signal_type": "momentum",
        "score": score,
        "flags": flags,
    }
