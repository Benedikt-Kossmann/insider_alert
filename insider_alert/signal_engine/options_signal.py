"""Options anomaly signal computation."""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    SignalComponent(
        key="call_volume_zscore", max_score=25, normaliser=3.0,
        flag_template="Elevated call volume z-score: {value:.2f}",
    ),
    SignalComponent(
        key="short_dated_otm_call_score", max_score=25, normaliser=1.0,
        flag_template="Short-dated OTM call activity: {value:.2f}",
    ),
    SignalComponent(
        key="block_trade_score", max_score=25, normaliser=1.0,
        flag_template="Block trades detected: {value:.2f}",
    ),
    SignalComponent(
        key="sweep_order_score", max_score=25, normaliser=1.0,
        flag_template="Sweep orders detected: {value:.2f}",
    ),
]


def compute_options_anomaly_signal(features: dict) -> dict:
    """Compute options anomaly signal from options features."""
    return compute_signal("options_anomaly", features, _COMPONENTS)
