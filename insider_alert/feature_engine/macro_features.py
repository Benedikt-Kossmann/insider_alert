"""Macro-regime feature computation from cross-asset data."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_macro_features(macro_data: dict[str, pd.DataFrame]) -> dict:
    """Derive macro-regime features from VIX, yield-curve, and Dollar data.

    Parameters
    ----------
    macro_data : dict
        Output of ``fetch_macro_data()`` with keys ``vix``, ``tnx``, ``irx``, ``dxy``.

    Returns
    -------
    dict
        Macro features including regime classification.
    """
    defaults = {
        "vix_current": 0.0,
        "vix_sma_20": 0.0,
        "vix_regime": "unknown",
        "yield_spread": 0.0,
        "yield_curve_regime": "unknown",
        "dxy_current": 0.0,
        "dxy_return_20d": 0.0,
        "dxy_trend": "flat",
        "risk_regime": "neutral",
        "macro_score": 50.0,
    }

    # --- VIX ---
    vix_df = macro_data.get("vix", pd.DataFrame())
    if not vix_df.empty and "close" in vix_df.columns:
        vix_close = vix_df["close"].dropna()
        if len(vix_close) >= 1:
            defaults["vix_current"] = float(vix_close.iloc[-1])
        if len(vix_close) >= 20:
            defaults["vix_sma_20"] = float(vix_close.iloc[-20:].mean())
        else:
            defaults["vix_sma_20"] = float(vix_close.mean())

        vix = defaults["vix_current"]
        if vix < 15:
            defaults["vix_regime"] = "low"
        elif vix < 25:
            defaults["vix_regime"] = "normal"
        else:
            defaults["vix_regime"] = "high"

    # --- Yield Curve (10Y - 3M spread) ---
    tnx_df = macro_data.get("tnx", pd.DataFrame())
    irx_df = macro_data.get("irx", pd.DataFrame())
    if (
        not tnx_df.empty and "close" in tnx_df.columns
        and not irx_df.empty and "close" in irx_df.columns
    ):
        tnx_last = float(tnx_df["close"].dropna().iloc[-1]) if len(tnx_df) else 0.0
        irx_last = float(irx_df["close"].dropna().iloc[-1]) if len(irx_df) else 0.0
        spread = tnx_last - irx_last
        defaults["yield_spread"] = round(spread, 3)
        # Expose IRX as a decimal risk-free rate for Black-Scholes Greeks (^IRX is in %)
        defaults["irx_rate"] = round(irx_last / 100.0, 6)

        if spread < -0.5:
            defaults["yield_curve_regime"] = "inverted"
        elif spread < 0.25:
            defaults["yield_curve_regime"] = "flat"
        else:
            defaults["yield_curve_regime"] = "normal"

    # --- Dollar Index ---
    dxy_df = macro_data.get("dxy", pd.DataFrame())
    if not dxy_df.empty and "close" in dxy_df.columns:
        dxy_close = dxy_df["close"].dropna()
        if len(dxy_close) >= 1:
            defaults["dxy_current"] = float(dxy_close.iloc[-1])
        if len(dxy_close) >= 20:
            ret_20d = (float(dxy_close.iloc[-1]) / float(dxy_close.iloc[-20]) - 1)
            defaults["dxy_return_20d"] = round(ret_20d, 4)
            if ret_20d > 0.02:
                defaults["dxy_trend"] = "rising"
            elif ret_20d < -0.02:
                defaults["dxy_trend"] = "falling"
            else:
                defaults["dxy_trend"] = "flat"

    # --- Composite risk regime ---
    score = 50.0  # neutral baseline

    vr = defaults["vix_regime"]
    if vr == "low":
        score += 25.0
    elif vr == "normal":
        score += 10.0
    elif vr == "high":
        score -= 25.0

    yc = defaults["yield_curve_regime"]
    if yc == "normal":
        score += 15.0
    elif yc == "flat":
        score += 0.0
    elif yc == "inverted":
        score -= 20.0

    dt = defaults["dxy_trend"]
    if dt == "falling":
        score += 10.0  # weaker dollar = risk-on
    elif dt == "rising":
        score -= 10.0

    score = float(np.clip(score, 0.0, 100.0))
    defaults["macro_score"] = score

    if score >= 65:
        defaults["risk_regime"] = "risk_on"
    elif score <= 35:
        defaults["risk_regime"] = "risk_off"
    else:
        defaults["risk_regime"] = "neutral"

    # --- Aliases used by the extended macro_signal (SignalComponent keys) ---
    defaults["vix_value"] = defaults["vix_current"]
    # dxy_change_5d: convert decimal 20d return to %-points (best proxy available)
    defaults["dxy_change_5d"] = round(defaults["dxy_return_20d"] * 100.0, 3)

    return defaults


def compute_fred_macro_features(fred_data: dict, market_data: dict) -> dict:
    """Derive normalised macro features from FRED data.

    Parameters
    ----------
    fred_data : dict
        Output of :func:`~insider_alert.data_ingestion.fred_data.fetch_all_macro_data`.
    market_data : dict
        Existing macro features dict (output of :func:`compute_macro_features`).
        Used only for context; not mutated.

    Returns
    -------
    dict
        Normalised FRED-derived features ready to merge into the macro context.
    """
    features: dict = {}

    # --- Credit Stress Score (0 = calm, 1 = extreme stress) ---
    hy_spread = float(fred_data.get("hy_spread", 4.0))
    hy_change = float(fred_data.get("hy_spread_change_1m", 0.0))
    # Normal: 3-4 %, elevated: >5 %, crisis: >8 %
    credit_stress = float(np.clip((hy_spread - 3.0) / 5.0, 0.0, 1.0))
    if hy_change > 0.5:
        credit_stress = min(1.0, credit_stress + 0.2)
    features["credit_stress_score"] = round(credit_stress, 3)

    # --- Fed Policy Score (0 = neutral, +1 = hawkish, -1 = dovish) ---
    direction = fred_data.get("fed_policy_direction", "hold")
    if direction == "tightening":
        features["fed_policy_score"] = 0.7
    elif direction == "easing":
        features["fed_policy_score"] = -0.7
    else:
        features["fed_policy_score"] = 0.0

    # --- Inflation Score ---
    cpi_yoy = float(fred_data.get("cpi_yoy_change", 2.0))
    # Target band 2-3 %. Below 1 % = deflation risk, above 5 % = problematic.
    features["inflation_score"] = round(float(np.clip((cpi_yoy - 2.0) / 4.0, -1.0, 1.0)), 3)

    # --- Labor Market Score (-1 = weak, +1 = strong) ---
    claims_z = float(fred_data.get("initial_claims_zscore", 0.0))
    # Negative Z-score = fewer claims = strong labor market
    features["labor_market_score"] = round(float(np.clip(-claims_z * 0.5, -1.0, 1.0)), 3)

    # --- Consumer Sentiment (normalised 0-1) ---
    sentiment = float(fred_data.get("consumer_sentiment", 70.0))
    features["consumer_sentiment_norm"] = round(float(np.clip((sentiment - 50.0) / 50.0, 0.0, 1.0)), 3)

    # --- Macro Regime (qualitative) ---
    if credit_stress > 0.6 or cpi_yoy > 6.0:
        features["macro_regime"] = "stress"
    elif credit_stress < 0.2 and features["labor_market_score"] > 0.3:
        features["macro_regime"] = "expansion"
    else:
        features["macro_regime"] = "neutral"

    return features
