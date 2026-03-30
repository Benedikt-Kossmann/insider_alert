"""Breakout & Trend alert detection.

Detects range breakouts on daily and intraday timeframes, requiring volume
confirmation and an impulsive price move.  An ATR-based stop level and a
basic reward:risk estimate are included in the returned alert dict.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default parameters (overridable via config)
DEFAULT_BREAKOUT_WINDOW = 20        # bars used to define the prior range
DEFAULT_VOLUME_CONFIRMATION = 1.5   # RVOL threshold for volume confirmation
DEFAULT_IMPULSE_FACTOR = 1.0        # breakout move must be >= 1× ATR
DEFAULT_ATR_WINDOW = 14
DEFAULT_RR_RATIO = 2.0              # typical reward:risk used in message hints


def detect_breakout(
    ohlcv: pd.DataFrame,
    price_features: dict,
    volume_features: dict,
    *,
    breakout_window: int = DEFAULT_BREAKOUT_WINDOW,
    volume_confirmation: float = DEFAULT_VOLUME_CONFIRMATION,
    impulse_factor: float = DEFAULT_IMPULSE_FACTOR,
    atr_window: int = DEFAULT_ATR_WINDOW,
    rr_ratio: float = DEFAULT_RR_RATIO,
) -> dict | None:
    """Return a breakout alert dict or *None* if no breakout is detected.

    The alert dict contains:
    ``ticker`` (str), ``alert_type`` (str), ``setup_type`` (str),
    ``direction`` (str: 'bullish'|'bearish'),
    ``breakout_level`` (float), ``atr`` (float),
    ``stop_hint`` (float), ``target_hint`` (float), ``rr_ratio`` (float),
    ``score`` (float 0–100), ``flags`` (list[str]).
    """
    if ohlcv is None or ohlcv.empty or len(ohlcv) < breakout_window + 2:
        return None

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    if not {"high", "low", "close", "volume"}.issubset(df.columns):
        return None

    closes = df["close"].dropna()
    highs = df["high"].dropna()
    lows = df["low"].dropna()

    # Range defined over the prior *breakout_window* bars (excluding current)
    prior = df.iloc[-(breakout_window + 1):-1]
    range_high = float(prior["high"].max())
    range_low = float(prior["low"].min())
    current_close = float(closes.iloc[-1])
    current_high = float(highs.iloc[-1])
    current_low = float(lows.iloc[-1])

    from insider_alert.feature_engine.price_features import compute_atr
    atr = compute_atr(df, window=atr_window)

    rvol = volume_features.get("volume_rvol_20d", 1.0)
    volume_confirmed = rvol >= volume_confirmation

    flags: list[str] = []
    direction: str | None = None
    breakout_level: float = 0.0
    score: float = 0.0

    # Bullish breakout: close above range high by at least impulse_factor × ATR
    if current_close > range_high and (current_close - range_high) >= impulse_factor * atr:
        direction = "bullish"
        breakout_level = range_high
        score = 60.0
        if volume_confirmed:
            score += 20.0
            flags.append(f"Volume confirmed: RVOL={rvol:.2f}x")
        move_atr = (current_close - range_high) / (atr + 1e-9)
        score += min(move_atr / 2.0, 1.0) * 20.0
        flags.append(f"Bullish breakout above {range_high:.2f} (ATR={atr:.2f})")

    # Bearish breakout: close below range low by at least impulse_factor × ATR
    elif current_close < range_low and (range_low - current_close) >= impulse_factor * atr:
        direction = "bearish"
        breakout_level = range_low
        score = 60.0
        if volume_confirmed:
            score += 20.0
            flags.append(f"Volume confirmed: RVOL={rvol:.2f}x")
        move_atr = (range_low - current_close) / (atr + 1e-9)
        score += min(move_atr / 2.0, 1.0) * 20.0
        flags.append(f"Bearish breakdown below {range_low:.2f} (ATR={atr:.2f})")

    if direction is None:
        return None

    if direction == "bullish":
        stop_hint = current_close - atr
        target_hint = current_close + rr_ratio * atr
    else:
        stop_hint = current_close + atr
        target_hint = current_close - rr_ratio * atr

    return {
        "alert_type": "breakout",
        "setup_type": f"breakout_{direction}",
        "direction": direction,
        "breakout_level": round(breakout_level, 4),
        "atr": round(atr, 4),
        "stop_hint": round(stop_hint, 4),
        "target_hint": round(target_hint, 4),
        "rr_ratio": rr_ratio,
        "score": float(np.clip(score, 0.0, 100.0)),
        "flags": flags,
    }
