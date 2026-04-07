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

    return defaults
