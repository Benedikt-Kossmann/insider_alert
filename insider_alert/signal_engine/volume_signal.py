"""Volume anomaly signal computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Scoring weights / normalisation constants for the volume anomaly signal
_RVOL_NORMALISER = 2.0         # an RVOL excess of 2.0× yields maximum contribution
_RVOL_MAX_SCORE = 50           # maximum score points from RVOL component
_ZSCORE_NORMALISER = 3.0       # volume z-score that yields maximum contribution
_ZSCORE_MAX_SCORE = 30         # maximum score points from the z-score component
_TIGHT_RANGE_MAX_SCORE = 20    # maximum score points from the tight-range flag


def compute_volume_anomaly_signal(features: dict) -> dict:
    """Compute volume anomaly signal from volume features."""
    flags = []

    rvol = features.get("volume_rvol_20d", 1.0)
    zscore = features.get("volume_zscore_20d", 0.0)
    tight_flag = features.get("tight_range_high_volume_flag", 0)

    rvol_component = min((rvol - 1) / _RVOL_NORMALISER, 1.0) * _RVOL_MAX_SCORE
    rvol_component = max(rvol_component, 0.0)
    zscore_component = min(abs(zscore) / _ZSCORE_NORMALISER, 1.0) * _ZSCORE_MAX_SCORE
    tight_range_component = tight_flag * _TIGHT_RANGE_MAX_SCORE

    if rvol_component > _RVOL_MAX_SCORE / 2:
        flags.append(f"Elevated relative volume: {rvol:.2f}x")
    if zscore_component > _ZSCORE_MAX_SCORE / 2:
        flags.append(f"High volume z-score: {zscore:.2f}")
    if tight_flag:
        flags.append("Tight price range with high volume detected")

    score = float(np.clip(rvol_component + zscore_component + tight_range_component, 0.0, 100.0))

    return {
        "signal_type": "volume_anomaly",
        "score": score,
        "flags": flags,
    }
