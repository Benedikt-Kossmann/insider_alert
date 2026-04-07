"""Volatility-regime signal – scores how favourable current volatility is for leverage."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_volatility_regime_signal(
    vol_regime_features: dict,
    leverage_features: dict,
) -> dict:
    """Score current volatility conditions for leveraged-ETF positioning.

    A **high** score means conditions are favourable for holding leveraged
    ETFs (low VIX, contracting ATR, low decay).

    Parameters
    ----------
    vol_regime_features : dict
        Output of ``compute_volatility_regime_features()``.
    leverage_features : dict
        Output of ``compute_leverage_features()``.

    Returns
    -------
    dict
        ``{"signal_type": "volatility_regime", "score": float, "flags": list[str]}``
    """
    score = 0.0
    flags: list[str] = []

    vix_regime = vol_regime_features.get("vix_regime", "unknown")
    vix_current = float(vol_regime_features.get("vix_current", 0.0))
    atr_regime = vol_regime_features.get("atr_regime", "normal")
    realized_vol = float(vol_regime_features.get("realized_vol_20d", 0.0))
    estimated_decay = float(leverage_features.get("estimated_daily_decay", 0.0))

    # --- VIX regime (up to 40 pts) ---
    if vix_regime == "low":
        score += 40.0
        flags.append(f"VIX low ({vix_current:.1f}) — ideal for leverage")
    elif vix_regime == "normal":
        score += 25.0
        flags.append(f"VIX normal ({vix_current:.1f})")
    elif vix_regime == "high":
        score += 5.0
        flags.append(f"VIX elevated ({vix_current:.1f}) — high leverage risk")
    else:
        # VIX data unavailable — neutral
        score += 20.0

    # --- ATR regime (up to 30 pts) ---
    if atr_regime == "contracting":
        score += 30.0
        flags.append("ATR contracting — decreasing volatility")
    elif atr_regime == "normal":
        score += 15.0
    elif atr_regime == "expanding":
        score += 0.0
        flags.append("ATR expanding — increasing volatility risk")

    # --- Decay estimate (up to 30 pts) ---
    if estimated_decay < 0.005:
        score += 30.0
        flags.append(f"Low estimated decay ({estimated_decay:.4f}/day)")
    elif estimated_decay < 0.01:
        score += 20.0
        flags.append(f"Moderate decay ({estimated_decay:.4f}/day)")
    elif estimated_decay < 0.02:
        score += 10.0
        flags.append(f"Elevated decay ({estimated_decay:.4f}/day)")
    else:
        score += 0.0
        flags.append(f"High decay warning ({estimated_decay:.4f}/day)")

    score = float(np.clip(score, 0.0, 100.0))

    return {
        "signal_type": "volatility_regime",
        "score": score,
        "flags": flags,
    }
