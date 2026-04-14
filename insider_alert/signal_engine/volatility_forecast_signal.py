"""Volatility forecast signal based on GARCH(1,1)."""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    SignalComponent(
        key="vol_surprise_ratio",
        max_score=40,
        # Deviation above neutral (1.0): val = raw - 1.0; 2.0 → score=40, 1.0 → score=0
        normaliser=1.0,
        offset=1.0,
        flag_template="📈 Vol Surprise: {value:.2f}x expected",
        flag_threshold=0.375,  # fires when score > 15 (raw ~= 1.375x)
    ),
    SignalComponent(
        key="vol_of_vol",
        max_score=30,
        normaliser=0.5,  # CoV typically 0–0.5
        flag_template="⚡ Vol Instability (vol-of-vol): {value:.2f}",
        flag_threshold=0.5,
    ),
    SignalComponent(
        key="garch_forecast_1d",
        max_score=30,
        normaliser=0.5,  # annualised vol: 0.50 = 50% → max score
        flag_template="🔮 GARCH 1d Vol Forecast: {value:.1%}",
        flag_threshold=0.5,  # fires when forecast > 25% ann. vol
    ),
]


def compute_volatility_forecast_signal(features: dict) -> dict:
    """Compute volatility forecast signal from GARCH-based features.

    Returns ``{"signal_type": "volatility_forecast", "score": 0-100, "flags": [...]}``.
    """
    return compute_signal("volatility_forecast", features, _COMPONENTS)
