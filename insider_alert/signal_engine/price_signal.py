"""Price anomaly signal computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Scoring weights / normalisation constants for the price anomaly signal
_ZSCORE_NORMALISER = 3.0       # z-score that yields a maximum score contribution
_RETURN_5D_NORMALISER = 0.10   # 10 % five-day return maps to full weight
_ZSCORE_MAX_SCORE = 40         # maximum score points from the z-score component
_RETURN_5D_MAX_SCORE = 30      # maximum score points from the 5-day return component
_GAP_MAX_SCORE = 30            # maximum score points from the gap component
_GAP_NORMALISER = 3.0          # 3 gaps in 5 days yields maximum gap score


def compute_price_anomaly_signal(features: dict) -> dict:
    """Compute price anomaly signal from price features."""
    flags = []

    zscore = features.get("daily_return_zscore", 0.0)
    return_5d = features.get("return_5d", 0.0)
    gap_up = features.get("gap_up_count_5d", 0)
    gap_down = features.get("gap_down_count_5d", 0)

    zscore_component = min(abs(zscore) / _ZSCORE_NORMALISER, 1.0) * _ZSCORE_MAX_SCORE
    return5d_component = min(abs(return_5d) / _RETURN_5D_NORMALISER, 1.0) * _RETURN_5D_MAX_SCORE
    gap_component = min((gap_up + gap_down) / _GAP_NORMALISER, 1.0) * _GAP_MAX_SCORE

    if zscore_component > _ZSCORE_MAX_SCORE / 2:
        flags.append(f"High daily return z-score: {zscore:.2f}")
    if return5d_component > _RETURN_5D_MAX_SCORE / 2:
        flags.append(f"Significant 5d return: {return_5d * 100:.1f}%")
    if gap_component > _GAP_MAX_SCORE / 2:
        flags.append(f"Multiple gaps in last 5d: up={gap_up}, down={gap_down}")

    score = float(np.clip(zscore_component + return5d_component + gap_component, 0.0, 100.0))

    return {
        "signal_type": "price_anomaly",
        "score": score,
        "flags": flags,
    }
