"""Short squeeze signal from FINRA RegSHO short volume data."""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    SignalComponent(
        key="short_ratio_zscore",
        max_score=40,
        normaliser=3.0,       # Z-scores typically [-3, 3]
        flag_template="🩳 Short Ratio Z-Score: {value:+.1f}σ",
        flag_threshold=1.5,
        abs_value=True,       # Extremes in either direction are interesting
    ),
    SignalComponent(
        key="short_ratio_trend_5d",
        max_score=30,
        normaliser=0.2,       # Typical range ±0.05–0.15
        flag_template="📈 Short Ratio Trend 5d: {value:+.3f}",
        flag_threshold=0.05,
        abs_value=True,
    ),
    SignalComponent(
        key="short_squeeze_score",
        max_score=30,
        normaliser=1.0,
        flag_template="💥 Short Squeeze Score: {value:.0%}",
        flag_threshold=0.3,
    ),
]


def short_squeeze_signal(features: dict) -> dict:
    """Compute short squeeze signal from FINRA short volume features.

    Returns ``{"signal_type": "short_squeeze", "score": 0-100, "flags": [...]}``.
    """
    return compute_signal("short_squeeze", features, _COMPONENTS)
