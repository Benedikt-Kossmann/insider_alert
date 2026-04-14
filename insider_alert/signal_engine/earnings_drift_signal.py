"""Post-Earnings Announcement Drift (PEAD) signal."""
import numpy as np

from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal


def compute_pead_features(earnings_data: dict) -> dict:
    """Derive PEAD features from raw earnings data.

    Parameters
    ----------
    earnings_data : dict
        Output of :func:`~insider_alert.data_ingestion.earnings_data.fetch_earnings_data`.

    Returns
    -------
    dict
        earnings_surprise_magnitude : float — surprise strength, normalised 0–1
        drift_remaining : float — time-decay factor indicating remaining drift potential (0–1)
        pead_direction : float — +1.0 = positive drift expected, −1.0 = negative
    """
    defaults = {
        "earnings_surprise_magnitude": 0.0,
        "drift_remaining": 0.0,
        "pead_direction": 0.0,
    }

    day_return = float(earnings_data.get("earnings_day_return", 0.0))
    days_since = int(earnings_data.get("days_since_earnings", 999))

    # Only relevant within 60 days of earnings and when there was a meaningful move
    if days_since > 60 or abs(day_return) < 1.0:
        return defaults

    # Surprise magnitude: 3% = mild, 5% = moderate, 10%+ = extreme
    magnitude = float(np.clip(abs(day_return) / 10.0, 0.0, 1.0))

    # Direction of the initial move
    direction = 1.0 if day_return > 0 else -1.0

    # Time-decay: 100% on day 0, ~60% on day 10, ~22% on day 30, ~5% on day 60
    remaining = float(np.exp(-days_since / 20.0))

    return {
        "earnings_surprise_magnitude": round(magnitude, 3),
        "drift_remaining": round(remaining, 3),
        "pead_direction": direction,
    }


_COMPONENTS = [
    SignalComponent(
        key="earnings_surprise_magnitude",
        max_score=40,
        normaliser=1.0,
        flag_template="📊 Earnings Surprise: {value:.0%}",
        flag_threshold=0.3,
    ),
    SignalComponent(
        key="drift_remaining",
        max_score=30,
        normaliser=1.0,
        flag_template="🎯 PEAD Drift Remaining: {value:.0%}",
        flag_threshold=0.3,
    ),
    SignalComponent(
        key="pead_direction",
        max_score=30,
        normaliser=1.0,
        flag_template="📈 PEAD Direction: {value:+.0f}",
        flag_threshold=0.5,
        abs_value=True,
    ),
]


def earnings_drift_signal(features: dict) -> dict:
    """Compute PEAD signal.

    Returns ``{"signal_type": "earnings_drift", "score": 0-100, "flags": [...]}``.
    """
    return compute_signal("earnings_drift", features, _COMPONENTS)
