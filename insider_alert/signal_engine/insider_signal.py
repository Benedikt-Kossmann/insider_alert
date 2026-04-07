"""Insider transaction signal computation."""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    SignalComponent(
        key="insider_buy_count_30d", max_score=30, normaliser=5.0,
        flag_template="Multiple insider buys in last 30d: {value}",
    ),
    SignalComponent(
        key="insider_cluster_score", max_score=40, normaliser=1.0,
        flag_template="Cluster buying detected: score={value:.2f}",
    ),
    SignalComponent(
        key="insider_role_weighted_score", max_score=30, normaliser=1.0,
        flag_template="Senior insider buying: role-weighted score={value:.2f}",
    ),
]


def compute_insider_signal(features: dict) -> dict:
    """Compute insider signal from insider transaction features."""
    return compute_signal("insider_signal", features, _COMPONENTS)
