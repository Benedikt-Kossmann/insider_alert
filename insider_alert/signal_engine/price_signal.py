"""Price anomaly signal computation."""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    SignalComponent(
        key="daily_return_zscore", max_score=40, normaliser=3.0,
        abs_value=True, flag_template="High daily return z-score: {value:.2f}",
    ),
    SignalComponent(
        key="return_5d", max_score=30, normaliser=0.10,
        abs_value=True, flag_template="Significant 5d return: {value:.1%}",
    ),
    SignalComponent(
        key="_gap_total", max_score=30, normaliser=3.0,
        flag_template="Multiple gaps in last 5d: {value:.0f}",
    ),
]


def compute_price_anomaly_signal(features: dict) -> dict:
    """Compute price anomaly signal from price features."""
    ext = {**features, "_gap_total": features.get("gap_up_count_5d", 0) + features.get("gap_down_count_5d", 0)}
    return compute_signal("price_anomaly", ext, _COMPONENTS)
