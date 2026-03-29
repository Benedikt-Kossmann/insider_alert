"""Order-flow anomaly signal computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_orderflow_anomaly_signal(features: dict) -> dict:
    """Compute order-flow anomaly signal from order-flow features."""
    flags = []

    imbalance = features.get("bid_ask_imbalance", 0.0)
    aggressive_buy = features.get("aggressive_buy_ratio", 0.0)
    iceberg = features.get("iceberg_suspect_score", 0.0)
    absorption = features.get("absorption_score", 0)

    imbalance_component = abs(imbalance) * 25
    aggressive_component = aggressive_buy * 25
    iceberg_component = iceberg * 25
    absorption_component = absorption * 25

    if imbalance_component > 12:
        flags.append(f"High bid-ask imbalance: {imbalance:.2f}")
    if aggressive_component > 12:
        flags.append(f"Aggressive buying: {aggressive_buy:.2f}")
    if iceberg_component > 12:
        flags.append(f"Iceberg order suspected: {iceberg:.2f}")
    if absorption:
        flags.append("Absorption pattern detected")

    score = float(np.clip(
        imbalance_component + aggressive_component + iceberg_component + absorption_component,
        0.0, 100.0
    ))

    return {
        "signal_type": "orderflow_anomaly",
        "score": score,
        "flags": flags,
    }
