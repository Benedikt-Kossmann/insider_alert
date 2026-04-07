"""Volume anomaly signal computation."""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    SignalComponent(
        key="volume_rvol_20d", max_score=50, normaliser=2.0, offset=1.0,
        flag_template="Elevated relative volume: {value:.2f}x",
    ),
    SignalComponent(
        key="volume_zscore_20d", max_score=30, normaliser=3.0,
        abs_value=True, flag_template="High volume z-score: {value:.2f}",
    ),
    SignalComponent(
        key="tight_range_high_volume_flag", max_score=20, binary=True,
        flag_template="Tight price range with high volume detected",
    ),
]


def compute_volume_anomaly_signal(features: dict) -> dict:
    """Compute volume anomaly signal from volume features."""
    return compute_signal("volume_anomaly", features, _COMPONENTS)
