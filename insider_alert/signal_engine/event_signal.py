"""Corporate event leadup signal computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_event_leadup_signal(features: dict) -> dict:
    """Compute event leadup signal from event features."""
    flags = []

    pre_return = features.get("pre_event_return_score", 0.0)
    pre_volume = features.get("pre_event_volume_score", 0.0)
    pre_options = features.get("pre_event_options_score", 0.0)
    days_to_earnings = features.get("days_to_earnings", 999)

    return_component = pre_return * 35
    volume_component = pre_volume * 35
    options_component = pre_options * 30

    if days_to_earnings <= 10:
        flags.append(f"Earnings in {days_to_earnings} days")
    if return_component > 17:
        flags.append(f"Pre-event price movement: score={pre_return:.2f}")
    if volume_component > 17:
        flags.append(f"Pre-event volume surge: score={pre_volume:.2f}")
    if options_component > 15:
        flags.append(f"Pre-event options activity: score={pre_options:.2f}")

    score = float(np.clip(return_component + volume_component + options_component, 0.0, 100.0))

    return {
        "signal_type": "event_leadup",
        "score": score,
        "flags": flags,
    }
