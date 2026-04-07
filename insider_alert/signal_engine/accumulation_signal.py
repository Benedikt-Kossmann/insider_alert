"""Accumulation pattern signal computation."""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    SignalComponent(
        key="wyckoff_accumulation_score", max_score=40, normaliser=1.0,
        flag_template="Wyckoff accumulation pattern detected: {value:.2f}",
    ),
    SignalComponent(
        key="higher_lows_score", max_score=30, normaliser=1.0,
        flag_template="Higher lows pattern: {value:.2f}",
    ),
    SignalComponent(
        key="range_compression_score", max_score=30, normaliser=1.0,
        flag_template="Range compression detected: {value:.2f}",
    ),
]


def compute_accumulation_signal(features: dict) -> dict:
    """Compute accumulation pattern signal from accumulation features."""
    return compute_signal("accumulation_pattern", features, _COMPONENTS)
