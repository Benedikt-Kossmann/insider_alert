"""Candle pattern signal (OHLCV-derived order-flow proxies)."""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    SignalComponent(
        key="bid_ask_imbalance", max_score=25, normaliser=1.0,
        abs_value=True, flag_template="Candle body imbalance: {value:.2f}",
    ),
    SignalComponent(
        key="aggressive_buy_ratio", max_score=25, normaliser=1.0,
        flag_template="Close near high (buying pressure proxy): {value:.2f}",
    ),
    SignalComponent(
        key="iceberg_suspect_score", max_score=25, normaliser=1.0,
        flag_template="Iceberg order suspected: {value:.2f}",
    ),
    SignalComponent(
        key="absorption_score", max_score=25, binary=True,
        flag_template="Absorption pattern detected",
    ),
]


def compute_orderflow_anomaly_signal(features: dict) -> dict:
    """Compute candle-pattern signal from OHLCV-derived order-flow proxies."""
    return compute_signal("candle_pattern", features, _COMPONENTS)
