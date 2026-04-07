"""Leverage-health signal – scores the tracking quality and alignment of a leveraged ETF."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_leverage_health_signal(leverage_features: dict) -> dict:
    """Score the health/quality of a leveraged ETF position.

    Parameters
    ----------
    leverage_features : dict
        Output of ``compute_leverage_features()``.

    Returns
    -------
    dict
        ``{"signal_type": "leverage_health", "score": float, "flags": list[str]}``
    """
    score = 0.0
    flags: list[str] = []

    tracking_error = float(leverage_features.get("tracking_error_5d", 0.0))
    corr = float(leverage_features.get("underlying_vs_etf_corr_20d", 0.0))
    trend_aligned = int(leverage_features.get("underlying_trend_aligned", 0))
    decay = float(leverage_features.get("estimated_daily_decay", 0.0))

    # --- Tracking error (up to 25 pts) ---
    if tracking_error < 0.01:
        score += 25.0
        flags.append(f"Excellent tracking (error={tracking_error:.4f})")
    elif tracking_error < 0.03:
        score += 18.0
        flags.append(f"Good tracking (error={tracking_error:.4f})")
    elif tracking_error < 0.05:
        score += 10.0
        flags.append(f"Moderate tracking error ({tracking_error:.4f})")
    else:
        score += 0.0
        flags.append(f"High tracking error ({tracking_error:.4f})")

    # --- Underlying trend alignment (up to 30 pts) ---
    if trend_aligned:
        score += 30.0
        flags.append("Underlying trend aligned with ETF direction")
    else:
        flags.append("Underlying trend NOT aligned with ETF direction")

    # --- Correlation (up to 20 pts) ---
    if corr >= 0.95:
        score += 20.0
    elif corr >= 0.85:
        score += 15.0
        flags.append(f"Correlation slightly below ideal ({corr:.2f})")
    elif corr >= 0.70:
        score += 8.0
        flags.append(f"Weak correlation ({corr:.2f})")
    else:
        flags.append(f"Poor correlation ({corr:.2f}) — structural risk")

    # --- Decay (up to 25 pts) ---
    if decay < 0.005:
        score += 25.0
    elif decay < 0.01:
        score += 18.0
        flags.append(f"Moderate estimated decay ({decay:.4f}/day)")
    elif decay < 0.02:
        score += 8.0
        flags.append(f"Elevated estimated decay ({decay:.4f}/day)")
    else:
        score += 0.0
        flags.append(f"Dangerous decay level ({decay:.4f}/day)")

    score = float(np.clip(score, 0.0, 100.0))

    return {
        "signal_type": "leverage_health",
        "score": score,
        "flags": flags,
    }
