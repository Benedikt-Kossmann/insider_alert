"""Price anomaly signal computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_price_anomaly_signal(features: dict) -> dict:
    """Compute price anomaly signal from price features."""
    flags = []

    zscore = features.get("daily_return_zscore", 0.0)
    return_5d = features.get("return_5d", 0.0)
    gap_up = features.get("gap_up_count_5d", 0)
    gap_down = features.get("gap_down_count_5d", 0)

    zscore_component = min(abs(zscore) / 3.0, 1.0) * 40
    return5d_component = min(abs(return_5d) / 0.10, 1.0) * 30
    gap_component = min((gap_up + gap_down) / 3.0, 1.0) * 30

    if zscore_component > 20:
        flags.append(f"High daily return z-score: {zscore:.2f}")
    if return5d_component > 15:
        flags.append(f"Significant 5d return: {return_5d * 100:.1f}%")
    if gap_component > 15:
        flags.append(f"Multiple gaps in last 5d: up={gap_up}, down={gap_down}")

    score = float(np.clip(zscore_component + return5d_component + gap_component, 0.0, 100.0))

    return {
        "signal_type": "price_anomaly",
        "score": score,
        "flags": flags,
    }
