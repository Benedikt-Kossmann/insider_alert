"""Insider transaction signal computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_insider_signal(features: dict) -> dict:
    """Compute insider signal from insider transaction features."""
    flags = []

    buy_count = features.get("insider_buy_count_30d", 0)
    cluster = features.get("insider_cluster_score", 0.0)
    role_weighted = features.get("insider_role_weighted_score", 0.0)

    buy_count_component = min(buy_count / 5.0, 1.0) * 30
    cluster_component = cluster * 40
    role_component = role_weighted * 30

    if buy_count_component > 15:
        flags.append(f"Multiple insider buys in last 30d: {buy_count}")
    if cluster_component > 20:
        flags.append(f"Cluster buying detected: score={cluster:.2f}")
    if role_component > 15:
        flags.append(f"Senior insider buying: role-weighted score={role_weighted:.2f}")

    score = float(np.clip(buy_count_component + cluster_component + role_component, 0.0, 100.0))

    return {
        "signal_type": "insider_signal",
        "score": score,
        "flags": flags,
    }
