"""Macro-regime signal – adjusts alert sensitivity based on market environment."""
import logging

import numpy as np

from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extended macro signal (Phase-6) – uses FRED-derived features
# ---------------------------------------------------------------------------

_COMPONENTS = [
    # Existing market-data components (reduced weights)
    SignalComponent(
        key="vix_value",
        max_score=25,
        normaliser=40.0,
        flag_template="🌡️ VIX: {value:.1f}",
        flag_threshold=25.0,
        offset=-15.0,
    ),
    SignalComponent(
        key="yield_spread",
        max_score=15,
        normaliser=3.0,
        flag_template="📈 Yield Curve: {value:.2f}%",
        flag_threshold=-0.3,
        abs_value=True,
    ),
    SignalComponent(
        key="dxy_change_5d",
        max_score=10,
        normaliser=3.0,
        flag_template="💵 DXY 5d: {value:+.1f}%",
        flag_threshold=1.5,
        abs_value=True,
    ),
    # New FRED-based components
    SignalComponent(
        key="credit_stress_score",
        max_score=25,
        normaliser=1.0,
        flag_template="🏦 Credit Stress: {value:.0%}",
        flag_threshold=0.5,
    ),
    SignalComponent(
        key="fed_policy_score",
        max_score=15,
        normaliser=1.0,
        flag_template="🏛️ Fed Policy: {value:+.1f}",
        flag_threshold=0.5,
        abs_value=True,
    ),
    SignalComponent(
        key="labor_market_score",
        max_score=10,
        normaliser=1.0,
        flag_template="👷 Labor Market: {value:+.2f}",
        flag_threshold=0.5,
        abs_value=True,
    ),
]


def macro_signal(features: dict) -> dict:
    """Extended macro signal (6 components, FRED-aware).

    Returns ``{"signal_type": "macro_regime", "score": 0-100, "flags": [...]}``
    """
    return compute_signal("macro_regime", features, _COMPONENTS)


def compute_macro_regime_signal(macro_features: dict) -> dict:
    """Score how favourable the macro environment is for risk-taking.

    A **high** score = risk-on (VIX low, yield curve normal, dollar weak).
    A **low** score  = risk-off (VIX high, inverted curve, strong dollar).

    Returns
    -------
    dict
        ``{"signal_type": "macro_regime", "score": float, "flags": list[str]}``
    """
    score = 0.0
    flags: list[str] = []

    # --- VIX component (up to 40 pts) ---
    vix_regime = macro_features.get("vix_regime", "unknown")
    vix_current = float(macro_features.get("vix_current", 0))

    if vix_regime == "low":
        score += 40.0
        flags.append(f"VIX low ({vix_current:.1f}) — risk-on")
    elif vix_regime == "normal":
        score += 25.0
        flags.append(f"VIX normal ({vix_current:.1f})")
    elif vix_regime == "high":
        score += 5.0
        flags.append(f"VIX elevated ({vix_current:.1f}) — risk-off")
    else:
        score += 20.0

    # --- Yield curve component (up to 35 pts) ---
    yc_regime = macro_features.get("yield_curve_regime", "unknown")
    spread = macro_features.get("yield_spread", 0)

    if yc_regime == "normal":
        score += 35.0
        flags.append(f"Yield curve normal (spread={spread:.2f}%)")
    elif yc_regime == "flat":
        score += 15.0
        flags.append(f"Yield curve flat (spread={spread:.2f}%)")
    elif yc_regime == "inverted":
        score += 0.0
        flags.append(f"Yield curve INVERTED (spread={spread:.2f}%) — recession risk")
    else:
        score += 15.0

    # --- Dollar trend component (up to 25 pts) ---
    dxy_trend = macro_features.get("dxy_trend", "flat")
    dxy_ret = macro_features.get("dxy_return_20d", 0)

    if dxy_trend == "falling":
        score += 25.0
        flags.append(f"Dollar weakening ({dxy_ret:+.1%}) — risk-on")
    elif dxy_trend == "flat":
        score += 15.0
        flags.append(f"Dollar stable ({dxy_ret:+.1%})")
    elif dxy_trend == "rising":
        score += 5.0
        flags.append(f"Dollar strengthening ({dxy_ret:+.1%}) — headwind")

    score = float(np.clip(score, 0.0, 100.0))

    return {
        "signal_type": "macro_regime",
        "score": score,
        "flags": flags,
    }
