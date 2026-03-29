"""Accumulation pattern signal computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_accumulation_signal(features: dict) -> dict:
    """Compute accumulation pattern signal from accumulation features."""
    flags = []

    wyckoff = features.get("wyckoff_accumulation_score", 0.0)
    higher_lows = features.get("higher_lows_score", 0.0)
    range_compression = features.get("range_compression_score", 0.0)

    wyckoff_component = wyckoff * 40
    higher_lows_component = higher_lows * 30
    range_compression_component = range_compression * 30

    if wyckoff_component > 20:
        flags.append(f"Wyckoff accumulation pattern detected: {wyckoff:.2f}")
    if higher_lows_component > 15:
        flags.append(f"Higher lows pattern: {higher_lows:.2f}")
    if range_compression_component > 15:
        flags.append(f"Range compression detected: {range_compression:.2f}")

    score = float(np.clip(wyckoff_component + higher_lows_component + range_compression_component, 0.0, 100.0))

    return {
        "signal_type": "accumulation_pattern",
        "score": score,
        "flags": flags,
    }
