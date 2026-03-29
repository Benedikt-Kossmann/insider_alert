"""Options anomaly signal computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_options_anomaly_signal(features: dict) -> dict:
    """Compute options anomaly signal from options features."""
    flags = []

    call_zscore = features.get("call_volume_zscore", 0.0)
    short_otm = features.get("short_dated_otm_call_score", 0.0)
    block = features.get("block_trade_score", 0.0)
    sweep = features.get("sweep_order_score", 0.0)

    call_component = min(max(call_zscore, 0.0) / 3.0, 1.0) * 25
    short_otm_component = short_otm * 25
    block_component = block * 25
    sweep_component = sweep * 25

    if call_component > 12:
        flags.append(f"Elevated call volume z-score: {call_zscore:.2f}")
    if short_otm_component > 12:
        flags.append(f"Short-dated OTM call activity: {short_otm:.2f}")
    if block_component > 12:
        flags.append(f"Block trades detected: {block:.2f}")
    if sweep_component > 12:
        flags.append(f"Sweep orders detected: {sweep:.2f}")

    score = float(np.clip(
        call_component + short_otm_component + block_component + sweep_component,
        0.0, 100.0
    ))

    return {
        "signal_type": "options_anomaly",
        "score": score,
        "flags": flags,
    }
