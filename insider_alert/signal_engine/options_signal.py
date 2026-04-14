"""Options flow signal based on real Black-Scholes Greeks."""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    SignalComponent(
        key="net_delta_exposure",
        max_score=30,
        normaliser=2.0,      # range approx [-2, 2] → [0, 1] after abs clipping
        flag_template="🔵 Net Delta Flow: {value:.2f}",
        flag_threshold=0.3,
    ),
    SignalComponent(
        key="gamma_imbalance",
        max_score=25,
        normaliser=1.0,
        flag_template="⚡ Gamma Imbalance: {value:.2f}",
        flag_threshold=0.3,
        abs_value=True,
    ),
    SignalComponent(
        key="iv_skew_25d",
        max_score=25,
        normaliser=1.0,
        flag_template="📊 IV Skew (25Δ): {value:.2f}",
        flag_threshold=0.2,
        abs_value=True,
    ),
    SignalComponent(
        key="iv_term_structure",
        max_score=20,
        normaliser=1.0,
        flag_template="📅 IV Term Structure: {value:.2f}",
        flag_threshold=0.15,
        abs_value=True,
    ),
]


def compute_options_anomaly_signal(features: dict) -> dict:
    """Compute options flow signal from Greek-based features.

    Returns ``{"signal_type": "options_anomaly", "score": 0-100, "flags": [...]}``.
    """
    return compute_signal("options_anomaly", features, _COMPONENTS)
